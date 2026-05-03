from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_forward_realized_vol(
    returns: pd.Series | pd.DataFrame,
    horizon: int,
    annualization_factor: float = 252,
) -> pd.Series | pd.DataFrame:
    """Forward realized volatility target indexed at t using returns t+1:t+h."""
    shifted = returns.shift(-1)
    # rolling() is right-aligned by default, so shift back (h-1) to index at t.
    vol = shifted.rolling(window=int(horizon), min_periods=int(horizon)).std().shift(-(int(horizon) - 1))
    return vol * np.sqrt(float(annualization_factor))


def _forward_drawdown_window(window_returns: pd.Series) -> float:
    wealth = (1.0 + window_returns.astype(float)).cumprod()
    wealth = pd.concat([pd.Series([1.0], index=[-1]), wealth])
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def make_forward_drawdown(
    returns: pd.Series,
    horizon: int,
) -> pd.Series:
    """Forward max drawdown target indexed at t using returns t+1:t+h."""
    horizon = int(horizon)
    out = pd.Series(np.nan, index=returns.index, dtype=float)
    values = returns.astype(float)

    for i in range(len(values)):
        window = values.iloc[i + 1 : i + 1 + horizon]
        if len(window) < horizon:
            continue
        out.iloc[i] = _forward_drawdown_window(window)

    return out


def make_forward_drawdown_event(
    returns: pd.Series,
    horizon: int,
    threshold: float,
) -> pd.Series:
    """Binary event where 1 indicates forward drawdown <= threshold."""
    dd = make_forward_drawdown(returns=returns, horizon=horizon)
    event = (dd <= float(threshold)).astype(float)
    event[dd.isna()] = np.nan
    return event


def make_forward_loss_quantile_event(
    returns: pd.Series,
    horizon: int,
    quantile: float,
) -> pd.Series:
    """Binary event based on lower-tail quantile of forward cumulative returns.

    Note: this uses a global threshold and is safe for descriptive targets.
    During model training, quantile thresholds should be estimated on train folds only.
    """
    fwd_cum = (1.0 + returns.shift(-1)).rolling(int(horizon), min_periods=int(horizon)).apply(
        np.prod,
        raw=True,
    ).shift(-(int(horizon) - 1)) - 1.0
    threshold = float(fwd_cum.dropna().quantile(float(quantile))) if fwd_cum.notna().any() else np.nan
    event = (fwd_cum <= threshold).astype(float)
    event[fwd_cum.isna()] = np.nan
    return event


def _pick_baseline_series(baseline_portfolio_returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(baseline_portfolio_returns, pd.DataFrame):
        if "min_variance" in baseline_portfolio_returns.columns:
            return baseline_portfolio_returns["min_variance"].astype(float)
        return baseline_portfolio_returns.iloc[:, 0].astype(float)
    return baseline_portfolio_returns.astype(float)


def make_supervised_targets(
    returns: pd.DataFrame,
    baseline_portfolio_returns: pd.Series | pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build Chapter 5 supervised targets in leakage-safe forward form."""
    cfg = config.get("targets", {})
    ann = float(config.get("calendar", {}).get("annualization_factor", 252.0))

    portfolio = _pick_baseline_series(baseline_portfolio_returns).reindex(returns.index)
    spy = returns["SPY"].astype(float)
    btc = returns["BTC-USD"].astype(float)

    targets = pd.DataFrame(index=returns.index)

    vol_cfg = cfg.get("volatility", {})
    vol_horizons = [int(h) for h in vol_cfg.get("horizons", [21, 63])]
    targets["target_portfolio_vol_21d"] = make_forward_realized_vol(portfolio, 21, ann)
    targets["target_portfolio_vol_63d"] = make_forward_realized_vol(portfolio, 63, ann)
    targets["target_spy_vol_21d"] = make_forward_realized_vol(spy, 21, ann)
    targets["target_btc_vol_21d"] = make_forward_realized_vol(btc, 21, ann)

    if 63 in vol_horizons:
        targets["target_spy_vol_63d"] = make_forward_realized_vol(spy, 63, ann)
        targets["target_btc_vol_63d"] = make_forward_realized_vol(btc, 63, ann)

    dd_cfg = cfg.get("drawdown_event", {})
    thresholds = dd_cfg.get("thresholds", {})
    p21 = float(thresholds.get("portfolio_21d", -0.05))
    p63 = float(thresholds.get("portfolio_63d", -0.10))
    b21 = float(thresholds.get("btc_21d", -0.15))

    targets["target_portfolio_dd_21d"] = make_forward_drawdown(portfolio, 21)
    targets["target_portfolio_dd_63d"] = make_forward_drawdown(portfolio, 63)
    targets["target_portfolio_dd_event_21d"] = make_forward_drawdown_event(portfolio, 21, p21)
    targets["target_portfolio_dd_event_63d"] = make_forward_drawdown_event(portfolio, 63, p63)
    targets["target_btc_dd_event_21d"] = make_forward_drawdown_event(btc, 21, b21)

    stress_cfg = cfg.get("stress_event", {})
    stress_h = int(stress_cfg.get("horizon", 21))
    stress_q = float(stress_cfg.get("quantile", 0.10))
    targets["target_stress_event_21d"] = make_forward_loss_quantile_event(
        portfolio,
        horizon=stress_h,
        quantile=stress_q,
    )

    targets.index.name = returns.index.name or "Date"
    return targets
