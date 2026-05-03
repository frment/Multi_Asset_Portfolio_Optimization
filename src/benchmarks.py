"""Benchmark portfolio construction.

Primary benchmark methodology matches the optimized backtest engine:
- same rebalance dates,
- same lookback warm-up,
- same holding return method,
- turnover computed against drifted pre-trade weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.backtest import (
    _compute_daily_returns_constant_mix,
    _compute_daily_returns_drifted_buy_and_hold,
    compute_turnover_one_way,
    get_rebalance_dates,
)
from src.config import load_settings


def _load_benchmark_weights(settings_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    benchmarks_cfg = settings_cfg.get("benchmarks", {})
    weights: dict[str, dict[str, float]] = {}

    sixty_forty = benchmarks_cfg.get("sixty_forty", {})
    if sixty_forty.get("weights"):
        weights["sixty_forty"] = {str(k): float(v) for k, v in sixty_forty["weights"].items()}

    fixed_crypto = benchmarks_cfg.get("fixed_small_crypto", {})
    if fixed_crypto.get("weights"):
        weights["fixed_small_crypto"] = {
            str(k): float(v) for k, v in fixed_crypto["weights"].items()
        }

    return weights


def _validate_weights(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    missing = [ticker for ticker in weights if ticker not in returns.columns]
    if missing:
        raise ValueError(
            "Benchmark weight dict references tickers not found in returns: "
            + ", ".join(missing)
        )

    ordered = pd.Series(weights, dtype=float)
    total = float(ordered.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Benchmark weights must sum to 1.0, got {total:.6f}.")
    return ordered


def _equal_weight_target(returns: pd.DataFrame) -> pd.Series:
    n_assets = len(returns.columns)
    return pd.Series(1.0 / n_assets, index=returns.columns, dtype=float)


def run_benchmark_backtest(
    returns: pd.DataFrame,
    *,
    strategy_name: str,
    target_weights: pd.Series,
    lookback_window: int,
    rebalance_frequency: str,
    holding_return_method: str,
    allow_weekend_rebalances: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run one benchmark strategy using same rebalance/holding conventions as optimizer."""
    method = str(holding_return_method).strip().lower()
    rebalance_dates = get_rebalance_dates(
        returns.index,
        lookback_window,
        rebalance_frequency=rebalance_frequency,
        allow_weekend_rebalances=allow_weekend_rebalances,
    )
    if not rebalance_dates:
        raise ValueError(
            f"No valid benchmark rebalance dates found with lookback={lookback_window}."
        )

    segments: list[pd.Series] = []
    weights_records: list[dict[str, Any]] = []
    turnover_records: list[dict[str, Any]] = []

    previous_pretrade: pd.Series | None = None

    for i, rebal_date in enumerate(rebalance_dates):
        if i + 1 < len(rebalance_dates):
            next_rebal = rebalance_dates[i + 1]
            holding_returns = returns.loc[rebal_date:next_rebal].iloc[:-1]
        else:
            holding_returns = returns.loc[rebal_date:]

        if holding_returns.empty:
            continue

        tw = target_weights.reindex(returns.columns, fill_value=0.0)
        tw = tw / float(tw.sum())

        if method == "drifted_buy_and_hold":
            daily_returns, pretrade = _compute_daily_returns_drifted_buy_and_hold(holding_returns, tw)
        elif method == "constant_target_weights":
            daily_returns, pretrade = _compute_daily_returns_constant_mix(holding_returns, tw)
        else:
            raise ValueError(
                "Unsupported holding_return_method: "
                f"{holding_return_method}. Use 'drifted_buy_and_hold' or 'constant_target_weights'."
            )

        daily_returns.name = strategy_name
        segments.append(daily_returns)

        row = {"rebalance_date": rebal_date}
        row.update(tw.to_dict())
        weights_records.append(row)

        is_initial = previous_pretrade is None
        if is_initial:
            turnover = 0.0
            n_changed = 0
            max_abs_change = 0.0
        else:
            turnover = compute_turnover_one_way(tw, previous_pretrade)
            all_tickers = tw.index.union(previous_pretrade.index)
            abs_changes = (
                tw.reindex(all_tickers, fill_value=0.0)
                - previous_pretrade.reindex(all_tickers, fill_value=0.0)
            ).abs()
            n_changed = int((abs_changes > 1e-6).sum())
            max_abs_change = float(abs_changes.max())

        turnover_records.append(
            {
                "rebalance_date": rebal_date,
                "strategy": strategy_name,
                "turnover_one_way": turnover,
                "is_initial_rebalance": bool(is_initial),
                "n_assets_changed": int(n_changed),
                "max_abs_weight_change": max_abs_change,
                "holding_return_method": method,
            }
        )

        previous_pretrade = pretrade

    strategy_returns = pd.concat(segments).sort_index()
    strategy_returns.name = strategy_name

    weights_history = pd.DataFrame(weights_records).set_index("rebalance_date")
    weights_history.index.name = "rebalance_date"

    turnover_history = pd.DataFrame(turnover_records).set_index("rebalance_date")
    turnover_history.index.name = "rebalance_date"

    return strategy_returns, weights_history, turnover_history


def run_benchmark_suite(
    returns: pd.DataFrame,
    *,
    lookback_window: int,
    rebalance_frequency: str,
    holding_return_method: str,
    allow_weekend_rebalances: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all benchmark series and their turnover/weights histories."""
    settings_cfg = load_settings()
    all_weights = _load_benchmark_weights(settings_cfg)

    sixty_forty = all_weights.get("sixty_forty", {"SPY": 0.60, "TLT": 0.40})
    fixed_small = all_weights.get(
        "fixed_small_crypto",
        {
            "BTC-USD": 0.05,
            "ETH-USD": 0.05,
            "SPY": 0.225,
            "QQQ": 0.225,
            "GLD": 0.225,
            "TLT": 0.225,
        },
    )

    strategies = {
        "equal_weight": _equal_weight_target(returns),
        "sixty_forty": _validate_weights(returns, sixty_forty),
        "fixed_small_crypto": _validate_weights(returns, fixed_small),
    }

    returns_list: list[pd.Series] = []
    turnover_list: list[pd.DataFrame] = []
    weights_list: list[pd.DataFrame] = []

    for name, target in strategies.items():
        strategy_returns, strategy_weights, strategy_turnover = run_benchmark_backtest(
            returns=returns,
            strategy_name=name,
            target_weights=target,
            lookback_window=lookback_window,
            rebalance_frequency=rebalance_frequency,
            holding_return_method=holding_return_method,
            allow_weekend_rebalances=allow_weekend_rebalances,
        )
        returns_list.append(strategy_returns)

        sw = strategy_weights.reset_index().copy()
        sw["strategy"] = name
        weights_list.append(sw)

        turnover_list.append(strategy_turnover.reset_index())

    benchmark_returns = pd.concat(returns_list, axis=1).sort_index()
    benchmark_turnover = pd.concat(turnover_list, ignore_index=True).sort_values(
        ["strategy", "rebalance_date"]
    )
    benchmark_weights = pd.concat(weights_list, ignore_index=True).sort_values(
        ["strategy", "rebalance_date"]
    )

    return benchmark_returns, benchmark_weights, benchmark_turnover


def build_all_benchmarks(
    returns: pd.DataFrame,
    *,
    lookback_window: int = 252,
    rebalance_frequency: str = "monthly",
    holding_return_method: str = "drifted_buy_and_hold",
    allow_weekend_rebalances: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper that returns only benchmark return series."""
    benchmark_returns, _, _ = run_benchmark_suite(
        returns=returns,
        lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency,
        holding_return_method=holding_return_method,
        allow_weekend_rebalances=allow_weekend_rebalances,
    )
    return benchmark_returns
