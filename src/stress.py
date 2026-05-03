"""Historical stress-testing helpers for pre-registered windows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.metrics import compute_all_metrics, expected_shortfall_historical, return_over_es


STRESS_COLUMNS: list[str] = [
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
]


def run_historical_stress_windows(
    strategy_returns: dict[str, pd.Series],
    windows: list[dict[str, Any]],
    *,
    beta: float = 0.95,
    risk_free_rate: float = 0.0,
    scope: str = "gross",
    annualization_factor: float = 252.0,
) -> pd.DataFrame:
    """Compute stress-period metrics for all strategies on registered windows.

    Args:
        strategy_returns: Mapping of strategy name to daily return series.
        windows: List of dicts with keys: window_id, label, start_date, end_date.
        beta: ES confidence level.
        risk_free_rate: Annualized risk-free rate for Sharpe.
        scope: Label for output rows (for example "gross" or "net").

    Returns:
        DataFrame with one row per (window, strategy).
    """
    rows: list[dict[str, Any]] = []

    for window in windows:
        window_id = str(window["window_id"])
        label = str(window.get("label", window_id))
        start = pd.to_datetime(window["start_date"])
        end = pd.to_datetime(window["end_date"])

        if start > end:
            raise ValueError(f"Stress window {window_id} has start_date > end_date.")

        for strategy_name, series in strategy_returns.items():
            clipped = series.loc[(series.index >= start) & (series.index <= end)].sort_index()
            if clipped.empty:
                continue

            perf = compute_all_metrics(
                clipped,
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
            )
            es = expected_shortfall_historical(clipped, beta=beta)
            ratio = return_over_es(clipped, beta=beta, annualization_factor=annualization_factor)

            rows.append(
                {
                    "window_id": window_id,
                    "window_label": label,
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "strategy": strategy_name,
                    "scope": scope,
                    "n_days": int(len(clipped)),
                    "ann_return": perf["ann_return"],
                    "ann_volatility": perf["ann_volatility"],
                    "sharpe": perf["sharpe"],
                    "max_drawdown": perf["max_drawdown"],
                    "calmar": perf["calmar"],
                    "expected_shortfall": es,
                    "return_over_es": ratio,
                }
            )

    if not rows:
        return pd.DataFrame(columns=STRESS_COLUMNS)

    out = pd.DataFrame(rows)
    return out[STRESS_COLUMNS].sort_values(["window_id", "scope", "strategy"]).reset_index(drop=True)
