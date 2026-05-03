from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest import _compute_daily_returns_drifted_buy_and_hold, compute_turnover_one_way
from src.costs import apply_rebalance_costs_to_daily_returns, bps_to_rate, build_rebalance_cost_series
from src.metrics import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    expected_shortfall_historical,
    max_drawdown,
    sharpe_ratio,
)
from src.risk_overlay import make_overlay_decision


@dataclass
class OverlayBacktestResult:
    daily_returns: pd.Series
    net_daily_returns: pd.Series
    weights_history: pd.DataFrame
    decisions: pd.DataFrame
    turnover: pd.DataFrame
    summary: pd.DataFrame


def _sortino_ratio(daily_returns: pd.Series, annualization_factor: float = 252.0) -> float:
    downside = daily_returns[daily_returns < 0.0]
    if len(downside) < 2:
        return np.nan
    downside_vol = downside.std() * np.sqrt(float(annualization_factor))
    if np.isclose(downside_vol, 0.0):
        return np.nan
    return float(annualised_return(daily_returns, annualization_factor) / downside_vol)


def _forecast_lookup(forecasts: pd.DataFrame, date: pd.Timestamp) -> dict[str, float | None]:
    if forecasts is None or forecasts.empty:
        return {"forecast_vol": None, "crash_probability": None}

    if date in forecasts.index:
        row = forecasts.loc[date]
    else:
        pos = forecasts.index.searchsorted(date)
        if pos == 0:
            return {"forecast_vol": None, "crash_probability": None}
        row = forecasts.iloc[pos - 1]

    return {
        "forecast_vol": float(row["forecast_vol"]) if "forecast_vol" in row and pd.notna(row["forecast_vol"]) else None,
        "crash_probability": float(row["crash_probability"]) if "crash_probability" in row and pd.notna(row["crash_probability"]) else None,
    }


def run_overlay_backtest(
    returns: pd.DataFrame,
    base_weights_history: pd.DataFrame,
    forecasts: pd.DataFrame,
    regime_data: pd.DataFrame | None,
    config: dict[str, Any],
    strategy_name: str = "overlay",
) -> OverlayBacktestResult:
    """Run drifted monthly overlay backtest on top of baseline target weights."""
    rebalance_dates = [pd.Timestamp(d) for d in base_weights_history.index if pd.Timestamp(d) in returns.index]
    if not rebalance_dates:
        raise ValueError("No overlapping rebalance dates between base weights and returns index")

    portfolio_segments: list[pd.Series] = []
    weight_rows: list[dict[str, Any]] = []
    turnover_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    previous_pretrade: pd.Series | None = None

    for i, date in enumerate(rebalance_dates):
        base_weights = base_weights_history.loc[date].astype(float)

        fcst = _forecast_lookup(forecasts, date)
        regime_state = None
        if regime_data is not None and not regime_data.empty:
            if date in regime_data.index:
                regime_state = regime_data.loc[date].to_dict()
            else:
                pos = regime_data.index.searchsorted(date)
                if pos > 0:
                    regime_state = regime_data.iloc[pos - 1].to_dict()

        decision = make_overlay_decision(
            date=date,
            base_weights=base_weights,
            forecasts=fcst,
            regime_state=regime_state,
            config=config,
        )
        target_weights = decision["adjusted_weights"]

        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
            holding_returns = returns.loc[date:next_date].iloc[:-1]
        else:
            holding_returns = returns.loc[date:]

        if holding_returns.empty:
            continue

        daily_ret, arriving_pretrade = _compute_daily_returns_drifted_buy_and_hold(
            holding_returns=holding_returns[target_weights.index],
            target_weights=target_weights,
        )
        daily_ret.name = strategy_name
        portfolio_segments.append(daily_ret)

        w_row = {"rebalance_date": date}
        w_row.update(target_weights.to_dict())
        weight_rows.append(w_row)

        if previous_pretrade is None:
            turnover = 0.0
            is_initial = True
        else:
            turnover = compute_turnover_one_way(target_weights, previous_pretrade)
            is_initial = False

        turnover_rows.append(
            {
                "rebalance_date": date,
                "turnover_one_way": float(turnover),
                "is_initial_rebalance": bool(is_initial),
            }
        )

        d_row = {
            "date": date,
            "risk_scale": decision["risk_scale"],
            "crypto_cap": decision["crypto_cap"],
            "de_risk_flag": decision["de_risk_flag"],
            "reason": decision["reason"],
            "base_crypto_weight": decision["base_crypto_weight"],
            "adjusted_crypto_weight": decision["adjusted_crypto_weight"],
            "forecast_vol": decision["model_inputs"]["forecast_vol"],
            "crash_probability": decision["model_inputs"]["crash_probability"],
        }
        decision_rows.append(d_row)

        previous_pretrade = arriving_pretrade

    gross_returns = pd.concat(portfolio_segments).sort_index() if portfolio_segments else pd.Series(dtype=float)

    turnover_df = pd.DataFrame(turnover_rows).set_index("rebalance_date") if turnover_rows else pd.DataFrame()
    costs_cfg = config.get("costs", {})
    apply_costs = bool(costs_cfg.get("apply_costs", True))
    cost_rate = bps_to_rate(float(costs_cfg.get("cost_bps", 10.0)))

    if apply_costs and not turnover_df.empty:
        rebal_costs = build_rebalance_cost_series(turnover_df["turnover_one_way"], cost_rate)
        net_returns, _ = apply_rebalance_costs_to_daily_returns(gross_returns, rebal_costs)
        total_cost = float(rebal_costs.sum())
    else:
        net_returns = gross_returns.copy()
        total_cost = 0.0

    weights_df = pd.DataFrame(weight_rows).set_index("rebalance_date") if weight_rows else pd.DataFrame()
    decisions_df = pd.DataFrame(decision_rows).set_index("date") if decision_rows else pd.DataFrame()

    ann = float(config.get("calendar", {}).get("annualization_factor", 252.0))
    turnover_operational = (
        turnover_df.loc[~turnover_df["is_initial_rebalance"], "turnover_one_way"] if not turnover_df.empty else pd.Series(dtype=float)
    )

    crypto_cols = [c for c in ["BTC-USD", "ETH-USD"] if c in weights_df.columns]
    crypto_weight = weights_df[crypto_cols].sum(axis=1) if crypto_cols else pd.Series(dtype=float)

    summary = pd.DataFrame(
        [
            {
                "strategy": strategy_name,
                "ann_return": annualised_return(net_returns, ann) if len(net_returns) > 0 else np.nan,
                "annualized_return": annualised_return(net_returns, ann) if len(net_returns) > 0 else np.nan,
                "ann_volatility": annualised_volatility(net_returns, ann) if len(net_returns) > 1 else np.nan,
                "annualized_volatility": annualised_volatility(net_returns, ann) if len(net_returns) > 1 else np.nan,
                "sharpe": sharpe_ratio(net_returns, annualization_factor=ann) if len(net_returns) > 1 else np.nan,
                "sortino": _sortino_ratio(net_returns, ann) if len(net_returns) > 1 else np.nan,
                "calmar": calmar_ratio(net_returns, ann) if len(net_returns) > 1 else np.nan,
                "max_drawdown": max_drawdown(net_returns) if len(net_returns) > 0 else np.nan,
                "es95": expected_shortfall_historical(net_returns, beta=0.95) if len(net_returns) > 0 else np.nan,
                "avg_turnover": float(turnover_operational.mean()) if len(turnover_operational) > 0 else np.nan,
                "turnover": float(turnover_operational.mean()) if len(turnover_operational) > 0 else np.nan,
                "total_costs": total_cost,
                "crypto_avg_weight": float(crypto_weight.mean()) if len(crypto_weight) > 0 else np.nan,
                "avg_crypto_weight": float(crypto_weight.mean()) if len(crypto_weight) > 0 else np.nan,
                "crypto_median_weight": float(crypto_weight.median()) if len(crypto_weight) > 0 else np.nan,
                "crypto_max_weight": float(crypto_weight.max()) if len(crypto_weight) > 0 else np.nan,
                "max_crypto_weight": float(crypto_weight.max()) if len(crypto_weight) > 0 else np.nan,
                "n_rebalances_crypto_gt_2pct": int((crypto_weight > 0.02).sum()) if len(crypto_weight) > 0 else 0,
                "n_crypto_rebalances_gt_2pct": int((crypto_weight > 0.02).sum()) if len(crypto_weight) > 0 else 0,
                "n_de_risking_events": int(decisions_df["de_risk_flag"].sum()) if not decisions_df.empty else 0,
                "avg_crypto_cap_applied": float(decisions_df["crypto_cap"].mean()) if not decisions_df.empty else np.nan,
            }
        ]
    )

    return OverlayBacktestResult(
        daily_returns=gross_returns,
        net_daily_returns=net_returns,
        weights_history=weights_df,
        decisions=decisions_df,
        turnover=turnover_df,
        summary=summary,
    )
