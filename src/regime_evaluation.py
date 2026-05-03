"""
Chapter 4 — Regime Evaluation (Phase 3)

Diagnostic, post-hoc conditional analysis by detected regime.

This module does not produce trading signals and does not use labels to build
forward-looking allocations. It only attributes already-realised strategy
returns and crypto sleeve behaviour to historical regimes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from costs import (
    apply_rebalance_costs_to_daily_returns,
    bps_to_rate,
    build_rebalance_cost_series,
)
from metrics import (
    annualised_return,
    annualised_volatility,
    expected_shortfall_historical,
    max_drawdown,
    return_over_es,
    sharpe_ratio,
)

logger = logging.getLogger(__name__)

COND_PERF_COLUMNS: list[str] = [
    "strategy",
    "regime_id",
    "regime_name",
    "ann_return",
    "ann_volatility",
    "sharpe",
    "max_drawdown",
    "expected_shortfall",
    "return_over_es",
    "hit_rate",
    "avg_negative_return",
    "n_obs",
]

CRYPTO_EXPOSURE_COLUMNS: list[str] = [
    "strategy",
    "regime_id",
    "regime_name",
    "mean_crypto_weight",
    "median_crypto_weight",
    "p90_crypto_weight",
    "max_crypto_weight",
    "share_crypto_gt_2pct",
    "share_crypto_le_1pct",
    "n_obs",
]

STRESS_MAP_COLUMNS: list[str] = [
    "window_id",
    "window_label",
    "start_date",
    "end_date",
    "strategy",
    "scope",
    "regime_id",
    "regime_name",
    "n_obs",
    "share_obs_in_window",
    "mapped_window_n_obs",
    "n_days_reported",
    "ann_return",
    "ann_volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
    "expected_shortfall",
    "return_over_es",
]


@dataclass
class RegimeEvaluationResult:
    """Container for all Phase 3 output tables."""

    conditional_performance: pd.DataFrame
    conditional_performance_net: pd.DataFrame
    crypto_exposure: pd.DataFrame
    drawdown_tail_summary: pd.DataFrame



def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(set(df.columns)))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")



def _prepare_labels(labels_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(labels_df, {"date", "regime_id", "regime_name"}, "regime_labels")
    out = labels_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["regime_id"] = out["regime_id"].astype(int)
    out["regime_name"] = out["regime_name"].astype(str)
    out = out[["date", "regime_id", "regime_name"]].drop_duplicates(subset=["date"])
    out = out.sort_values("date").reset_index(drop=True)
    return out



def _prepare_returns(returns_df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    _require_columns(
        returns_df,
        {"date", "strategy", "portfolio_return"},
        "tail_risk_returns",
    )
    out = returns_df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out[out["strategy"].isin(strategies)].copy()
    out["strategy"] = out["strategy"].astype(str)
    out["portfolio_return"] = out["portfolio_return"].astype(float)
    out = out[["date", "strategy", "portfolio_return"]].drop_duplicates(subset=["date", "strategy"])
    out = out.sort_values(["strategy", "date"]).reset_index(drop=True)
    return out



def _prepare_turnover(turnover_df: pd.DataFrame | None, strategies: list[str]) -> pd.DataFrame:
    if turnover_df is None:
        return pd.DataFrame(columns=["rebalance_date", "strategy", "turnover_one_way"])

    _require_columns(
        turnover_df,
        {"rebalance_date", "strategy", "turnover_one_way"},
        "tail_risk_turnover_panel",
    )
    out = turnover_df.copy()
    out["rebalance_date"] = pd.to_datetime(out["rebalance_date"])
    out = out[out["strategy"].isin(strategies)].copy()
    out["strategy"] = out["strategy"].astype(str)
    out["turnover_one_way"] = out["turnover_one_way"].astype(float)
    out = out[["rebalance_date", "strategy", "turnover_one_way"]].drop_duplicates(
        subset=["rebalance_date", "strategy"]
    )
    out = out.sort_values(["strategy", "rebalance_date"]).reset_index(drop=True)
    return out



def _build_net_returns_panel(
    gross_returns: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """Reconstruct net daily returns by subtracting rebalance costs.

    Cost model is the same used in prior chapters:
        cost_t = turnover_one_way_t * (cost_bps / 10000)

    and cost is applied on rebalance dates only.
    """
    cost_rate = bps_to_rate(cost_bps)
    rows: list[pd.DataFrame] = []

    for strategy, grp in gross_returns.groupby("strategy"):
        gross_series = grp.set_index("date")["portfolio_return"].sort_index()
        turn_series = (
            turnover.loc[turnover["strategy"] == strategy, ["rebalance_date", "turnover_one_way"]]
            .set_index("rebalance_date")["turnover_one_way"]
            .sort_index()
        )
        rebalance_costs = build_rebalance_cost_series(turn_series, cost_rate=cost_rate)
        net_series, _ = apply_rebalance_costs_to_daily_returns(gross_series, rebalance_costs)

        frame = net_series.rename("portfolio_return").to_frame().reset_index()
        frame = frame.rename(columns={"index": "date"})
        frame["strategy"] = strategy
        rows.append(frame[["date", "strategy", "portfolio_return"]])

    if not rows:
        return pd.DataFrame(columns=["date", "strategy", "portfolio_return"])

    return pd.concat(rows, ignore_index=True).sort_values(["strategy", "date"]).reset_index(drop=True)



def _align_returns_with_labels(returns_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    # Strict inner merge avoids any implicit forward fill / look-ahead.
    aligned = returns_df.merge(labels_df, on="date", how="inner", validate="many_to_one")
    aligned = aligned.sort_values(["strategy", "date"]).reset_index(drop=True)
    return aligned



def _regime_metrics_table(
    panel: pd.DataFrame,
    beta: float,
    annualization_factor: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for (strategy, regime_id, regime_name), grp in panel.groupby(["strategy", "regime_id", "regime_name"]):
        returns = grp["portfolio_return"].astype(float)
        negatives = returns[returns < 0.0]

        row = {
            "strategy": str(strategy),
            "regime_id": int(regime_id),
            "regime_name": str(regime_name),
            "ann_return": annualised_return(returns, annualization_factor=annualization_factor),
            "ann_volatility": annualised_volatility(returns, annualization_factor=annualization_factor),
            "sharpe": sharpe_ratio(returns, annualization_factor=annualization_factor),
            "max_drawdown": max_drawdown(returns),
            "expected_shortfall": expected_shortfall_historical(returns, beta=beta),
            "return_over_es": return_over_es(
                returns,
                beta=beta,
                annualization_factor=annualization_factor,
            ),
            "hit_rate": float((returns > 0.0).mean()),
            "avg_negative_return": float(negatives.mean()) if len(negatives) > 0 else float("nan"),
            "n_obs": int(len(returns)),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=COND_PERF_COLUMNS)

    return out[COND_PERF_COLUMNS].sort_values(["strategy", "regime_id"]).reset_index(drop=True)



def _prepare_weights(weights_df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    _require_columns(
        weights_df,
        {"rebalance_date", "strategy", "ticker", "weight"},
        "tail_risk_weights_panel",
    )
    out = weights_df.copy()
    out["rebalance_date"] = pd.to_datetime(out["rebalance_date"])
    out["strategy"] = out["strategy"].astype(str)
    out["ticker"] = out["ticker"].astype(str)
    out["weight"] = out["weight"].astype(float)
    out = out[out["strategy"].isin(strategies)].copy()
    return out[["rebalance_date", "strategy", "ticker", "weight"]]



def _crypto_exposure_by_regime(
    weights_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    crypto_tickers: list[str],
) -> pd.DataFrame:
    all_dates = (
        weights_df[["rebalance_date", "strategy"]]
        .drop_duplicates()
        .rename(columns={"rebalance_date": "date"})
    )

    crypto = weights_df[weights_df["ticker"].isin(crypto_tickers)].copy()
    crypto = (
        crypto.groupby(["rebalance_date", "strategy"], as_index=False)["weight"]
        .sum()
        .rename(columns={"rebalance_date": "date", "weight": "crypto_weight"})
    )

    panel = all_dates.merge(crypto, on=["date", "strategy"], how="left")
    panel["crypto_weight"] = panel["crypto_weight"].fillna(0.0)

    panel = panel.merge(labels_df, on="date", how="inner", validate="many_to_one")

    rows: list[dict[str, Any]] = []
    for (strategy, regime_id, regime_name), grp in panel.groupby(["strategy", "regime_id", "regime_name"]):
        s = grp["crypto_weight"].astype(float)
        rows.append(
            {
                "strategy": str(strategy),
                "regime_id": int(regime_id),
                "regime_name": str(regime_name),
                "mean_crypto_weight": float(s.mean()),
                "median_crypto_weight": float(s.median()),
                "p90_crypto_weight": float(s.quantile(0.90)),
                "max_crypto_weight": float(s.max()),
                "share_crypto_gt_2pct": float((s > 0.02).mean()),
                "share_crypto_le_1pct": float((s <= 0.01).mean()),
                "n_obs": int(len(s)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=CRYPTO_EXPOSURE_COLUMNS)

    return out[CRYPTO_EXPOSURE_COLUMNS].sort_values(["strategy", "regime_id"]).reset_index(drop=True)



def _stress_window_regime_map(
    stress_summary_df: pd.DataFrame,
    gross_panel: pd.DataFrame,
    net_panel: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        stress_summary_df,
        {
            "window_id",
            "window_label",
            "start_date",
            "end_date",
            "strategy",
            "scope",
            "n_days",
            "ann_return",
            "ann_volatility",
            "sharpe",
            "max_drawdown",
            "calmar",
            "expected_shortfall",
            "return_over_es",
        },
        "stress_summary",
    )

    stress = stress_summary_df.copy()
    stress["start_date"] = pd.to_datetime(stress["start_date"])
    stress["end_date"] = pd.to_datetime(stress["end_date"])

    rows: list[dict[str, Any]] = []

    for _, row in stress.iterrows():
        scope = str(row["scope"])
        strategy = str(row["strategy"])
        start = pd.to_datetime(row["start_date"])
        end = pd.to_datetime(row["end_date"])

        panel = gross_panel if scope == "gross" else net_panel
        win = panel.loc[
            (panel["strategy"] == strategy)
            & (panel["date"] >= start)
            & (panel["date"] <= end)
        ].copy()

        mapped_window_n_obs = int(len(win))
        if mapped_window_n_obs == 0:
            continue

        grp = (
            win.groupby(["regime_id", "regime_name"], as_index=False)
            .size()
            .rename(columns={"size": "n_obs"})
        )

        for _, grow in grp.iterrows():
            n_obs = int(grow["n_obs"])
            rows.append(
                {
                    "window_id": str(row["window_id"]),
                    "window_label": str(row["window_label"]),
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "strategy": strategy,
                    "scope": scope,
                    "regime_id": int(grow["regime_id"]),
                    "regime_name": str(grow["regime_name"]),
                    "n_obs": n_obs,
                    "share_obs_in_window": float(n_obs / mapped_window_n_obs),
                    "mapped_window_n_obs": mapped_window_n_obs,
                    "n_days_reported": int(row["n_days"]),
                    "ann_return": float(row["ann_return"]),
                    "ann_volatility": float(row["ann_volatility"]),
                    "sharpe": float(row["sharpe"]),
                    "max_drawdown": float(row["max_drawdown"]),
                    "calmar": float(row["calmar"]),
                    "expected_shortfall": float(row["expected_shortfall"]),
                    "return_over_es": float(row["return_over_es"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=STRESS_MAP_COLUMNS)

    out = out[STRESS_MAP_COLUMNS].sort_values(["window_id", "strategy", "scope", "regime_id"]).reset_index(drop=True)
    return out



def evaluate_regimes_from_frames(
    *,
    labels_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    stress_summary_df: pd.DataFrame,
    cfg: dict[str, Any],
    turnover_df: pd.DataFrame | None = None,
) -> RegimeEvaluationResult:
    """Compute all Phase 3 outputs from in-memory DataFrames."""
    eval_cfg = cfg["evaluation"]
    strategies = [str(s) for s in eval_cfg["strategies"]]
    beta = float(eval_cfg.get("es_beta", 0.95))
    annualization_factor = float(
        cfg.get("dataset_metadata", {}).get("annualization_factor", 252.0)
    )
    cost_bps = float(eval_cfg.get("cost_bps", 10.0))
    min_obs_warn = int(eval_cfg.get("min_obs_warning_threshold", 63))
    crypto_tickers = [str(t) for t in eval_cfg.get("crypto_tickers", ["BTC-USD", "ETH-USD"])]
    tolerance = float(eval_cfg.get("crypto_weight_tolerance", 1e-6))

    labels = _prepare_labels(labels_df)
    gross_returns = _prepare_returns(returns_df, strategies)
    turnover = _prepare_turnover(turnover_df, strategies)
    weights = _prepare_weights(weights_df, strategies)

    gross_panel = _align_returns_with_labels(gross_returns, labels)
    net_returns = _build_net_returns_panel(gross_returns, turnover, cost_bps=cost_bps)
    net_panel = _align_returns_with_labels(net_returns, labels)

    cond_gross = _regime_metrics_table(
        gross_panel,
        beta=beta,
        annualization_factor=annualization_factor,
    )
    cond_net = _regime_metrics_table(
        net_panel,
        beta=beta,
        annualization_factor=annualization_factor,
    )

    for _, row in cond_gross.iterrows():
        if int(row["n_obs"]) < min_obs_warn:
            logger.warning(
                "Low observations for regime slice: strategy=%s regime=%s n_obs=%d < %d",
                row["strategy"],
                row["regime_name"],
                int(row["n_obs"]),
                min_obs_warn,
            )

    crypto_exposure = _crypto_exposure_by_regime(weights, labels, crypto_tickers=crypto_tickers)

    no_crypto_controls = {"minvar_no_crypto_control", "cvar_no_crypto_control"}
    offenders = crypto_exposure.loc[
        crypto_exposure["strategy"].isin(no_crypto_controls)
        & (crypto_exposure["max_crypto_weight"] > tolerance)
    ]
    if len(offenders) > 0:
        logger.warning(
            "No-crypto controls exceed tolerance %.2e in %d regime slices.",
            tolerance,
            len(offenders),
        )

    drawdown_tail_summary = _stress_window_regime_map(
        stress_summary_df=stress_summary_df,
        gross_panel=gross_panel,
        net_panel=net_panel,
    )

    # Diagnostic caveat: stress windows overlap with regime labels in time, but this
    # mapping is descriptive only and must not be interpreted as causal evidence.

    return RegimeEvaluationResult(
        conditional_performance=cond_gross,
        conditional_performance_net=cond_net,
        crypto_exposure=crypto_exposure,
        drawdown_tail_summary=drawdown_tail_summary,
    )



def load_and_evaluate(cfg: dict[str, Any], project_root: Path) -> RegimeEvaluationResult:
    """Load configured input CSVs and run Phase 3 regime evaluation."""
    eval_cfg = cfg["evaluation"]
    inputs = eval_cfg["inputs"]

    labels = pd.read_csv(project_root / inputs["regime_labels"])
    returns = pd.read_csv(project_root / inputs["tail_risk_returns"])
    weights = pd.read_csv(project_root / inputs["tail_risk_weights_panel"])
    stress_summary = pd.read_csv(project_root / inputs["stress_summary"])

    turnover_df: pd.DataFrame | None = None
    if "tail_risk_turnover_panel" in inputs:
        turnover_path = project_root / inputs["tail_risk_turnover_panel"]
        if turnover_path.exists():
            turnover_df = pd.read_csv(turnover_path)

    return evaluate_regimes_from_frames(
        labels_df=labels,
        returns_df=returns,
        weights_df=weights,
        stress_summary_df=stress_summary,
        cfg=cfg,
        turnover_df=turnover_df,
    )



def save_evaluation_results(
    result: RegimeEvaluationResult,
    cfg: dict[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    """Persist Phase 3 output CSVs and return written paths."""
    out_cfg = cfg["evaluation"]["outputs"]
    output_dir = project_root / cfg["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    p1 = project_root / out_cfg["regime_conditional_performance"]
    result.conditional_performance.to_csv(p1, index=False)
    paths["regime_conditional_performance"] = p1

    p2 = project_root / out_cfg["regime_conditional_performance_net"]
    result.conditional_performance_net.to_csv(p2, index=False)
    paths["regime_conditional_performance_net"] = p2

    p3 = project_root / out_cfg["regime_crypto_exposure"]
    result.crypto_exposure.to_csv(p3, index=False)
    paths["regime_crypto_exposure"] = p3

    p4 = project_root / out_cfg["regime_drawdown_tail_summary"]
    result.drawdown_tail_summary.to_csv(p4, index=False)
    paths["regime_drawdown_tail_summary"] = p4

    return paths
