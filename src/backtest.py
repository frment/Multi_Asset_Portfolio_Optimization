"""Rolling walk-forward backtest engine.

Baseline methodology:
- Rebalance at configured frequency (monthly by default).
- Compute target weights at rebalance date using only prior observations.
- Hold between rebalances with drifted buy-and-hold weights.

Legacy approximation is still available via `holding_return_method="constant_target_weights"`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from src.optimizer import load_optimizer_config, minimise_variance


def get_rebalance_dates(
    index: pd.DatetimeIndex,
    lookback_window: int,
    rebalance_frequency: str = "monthly",
    allow_weekend_rebalances: bool = True,
) -> list[pd.Timestamp]:
    """Return the first available date of each period after lookback warm-up."""
    if len(index) == 0:
        return []

    frequency = rebalance_frequency.lower().strip()
    if frequency == "monthly":
        period_index = index.to_period("M")
    elif frequency == "quarterly":
        period_index = index.to_period("Q")
    elif frequency == "weekly":
        period_index = index.to_period("W-FRI")
    else:
        raise ValueError(
            "Unsupported rebalance_frequency: "
            f"{rebalance_frequency}. Use 'monthly', 'quarterly', or 'weekly'."
        )

    first_of_period = index.to_series().groupby(period_index).first().sort_values().values

    rebalance_dates: list[pd.Timestamp] = []
    for raw_date in first_of_period:
        date = pd.Timestamp(raw_date)
        if not allow_weekend_rebalances and date.dayofweek >= 5:
            raise ValueError(
                f"Weekend rebalance date detected ({date.date()}) while "
                "allow_weekend_rebalances=False."
            )

        pos = index.get_loc(date)
        if int(pos) >= int(lookback_window):
            rebalance_dates.append(date)

    return rebalance_dates


def compute_pretrade_weights(
    target_weights: pd.Series,
    holding_returns: pd.DataFrame,
) -> pd.Series:
    """Compute drifted pre-trade weights arriving at the next rebalance."""
    tickers = target_weights.index.tolist()

    gross_returns = (1.0 + holding_returns[tickers]).prod()
    drifted_values = target_weights * gross_returns
    total_value = float(drifted_values.sum())
    if total_value <= 0.0:
        return pd.Series(1.0 / len(tickers), index=tickers)
    return drifted_values / total_value


def compute_turnover_one_way(
    target_weights: pd.Series,
    pretrade_weights: pd.Series,
) -> float:
    """Compute one-way turnover from drifted pre-trade to new target weights."""
    all_tickers = target_weights.index.union(pretrade_weights.index)
    target = target_weights.reindex(all_tickers, fill_value=0.0)
    pretrade = pretrade_weights.reindex(all_tickers, fill_value=0.0)
    return float(0.5 * np.abs(target - pretrade).sum())


def _compute_daily_returns_constant_mix(
    holding_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    daily_port_return = (holding_returns[target_weights.index] * target_weights.values).sum(axis=1)
    pretrade = compute_pretrade_weights(target_weights, holding_returns)
    return daily_port_return, pretrade


def _compute_daily_returns_drifted_buy_and_hold(
    holding_returns: pd.DataFrame,
    target_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    current_weights = target_weights.astype(float).copy()
    daily_values: list[float] = []

    for _, row in holding_returns[target_weights.index].iterrows():
        portfolio_return = float((current_weights * row).sum())
        daily_values.append(portfolio_return)

        denom = 1.0 + portfolio_return
        if np.isclose(denom, 0.0):
            raise ValueError("Portfolio value collapsed to zero during holding period.")

        current_weights = current_weights * (1.0 + row)
        current_weights = current_weights / denom
        current_weights = current_weights / float(current_weights.sum())

    daily_port_return = pd.Series(daily_values, index=holding_returns.index)
    return daily_port_return, current_weights


def run_rolling_backtest(
    returns: pd.DataFrame,
    optimizer_fn: Callable[..., pd.Series],
    optimizer_config: dict[str, Any] | None = None,
    lookback_window: int = 252,
    rebalance_frequency: str = "monthly",
    strategy_name: str = "strategy",
    optimizer_kwargs: dict[str, Any] | None = None,
    holding_return_method: str = "drifted_buy_and_hold",
    allow_weekend_rebalances: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run a generic walk-forward backtest with pluggable optimizer.

    holding_return_method:
    - drifted_buy_and_hold: baseline monthly-rebalance methodology.
    - constant_target_weights: legacy daily constant-mix approximation.
    """
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
    method = str(holding_return_method).strip().lower()

    if method not in {"drifted_buy_and_hold", "constant_target_weights"}:
        raise ValueError(
            "Unsupported holding_return_method: "
            f"{holding_return_method}. Use 'drifted_buy_and_hold' or 'constant_target_weights'."
        )

    rebalance_dates = get_rebalance_dates(
        returns.index,
        lookback_window,
        rebalance_frequency=rebalance_frequency,
        allow_weekend_rebalances=allow_weekend_rebalances,
    )
    if not rebalance_dates:
        raise ValueError(
            "No valid rebalance dates found. "
            f"Need at least {lookback_window} rows of history before first rebalance."
        )

    portfolio_segments: list[pd.Series] = []
    weights_records: list[dict[str, Any]] = []
    turnover_records: list[dict[str, Any]] = []

    previous_pretrade_weights: pd.Series | None = None

    for i, rebal_date in enumerate(rebalance_dates):
        rebal_pos = returns.index.get_loc(rebal_date)
        training_window = returns.iloc[rebal_pos - lookback_window : rebal_pos]

        try:
            target_weights = optimizer_fn(training_window, config=optimizer_config, **optimizer_kwargs)
        except ValueError as exc:
            print(f"  Warning: optimisation failed on {rebal_date.date()} - {exc}")
            continue

        if i + 1 < len(rebalance_dates):
            next_rebal_date = rebalance_dates[i + 1]
            holding_returns = returns.loc[rebal_date:next_rebal_date].iloc[:-1]
        else:
            holding_returns = returns.loc[rebal_date:]

        if holding_returns.empty:
            continue

        if method == "drifted_buy_and_hold":
            daily_port_return, arriving_pretrade_weights = _compute_daily_returns_drifted_buy_and_hold(
                holding_returns=holding_returns,
                target_weights=target_weights,
            )
        else:
            daily_port_return, arriving_pretrade_weights = _compute_daily_returns_constant_mix(
                holding_returns=holding_returns,
                target_weights=target_weights,
            )

        daily_port_return.name = strategy_name
        portfolio_segments.append(daily_port_return)

        weight_record = {"rebalance_date": rebal_date}
        weight_record.update(target_weights.to_dict())
        weights_records.append(weight_record)

        is_initial = previous_pretrade_weights is None
        if is_initial:
            turnover = 0.0
            n_changed = 0
            max_abs_change = 0.0
            pretrade_for_record = target_weights
        else:
            pretrade_for_record = previous_pretrade_weights
            turnover = compute_turnover_one_way(target_weights, pretrade_for_record)
            all_tickers = target_weights.index.union(pretrade_for_record.index)
            abs_changes = (
                target_weights.reindex(all_tickers, fill_value=0.0)
                - pretrade_for_record.reindex(all_tickers, fill_value=0.0)
            ).abs()
            n_changed = int((abs_changes > 1e-6).sum())
            max_abs_change = float(abs_changes.max())

        turnover_records.append(
            {
                "rebalance_date": rebal_date,
                "turnover_one_way": turnover,
                "is_initial_rebalance": bool(is_initial),
                "n_assets_changed": int(n_changed),
                "max_abs_weight_change": max_abs_change,
                "holding_return_method": method,
            }
        )

        previous_pretrade_weights = arriving_pretrade_weights

    portfolio_returns = pd.concat(portfolio_segments).sort_index()
    portfolio_returns.name = strategy_name

    weights_history = pd.DataFrame(weights_records).set_index("rebalance_date")
    weights_history.index.name = "rebalance_date"

    turnover_history = pd.DataFrame(turnover_records).set_index("rebalance_date")
    turnover_history.index.name = "rebalance_date"

    return portfolio_returns, weights_history, turnover_history


def run_min_variance_backtest(
    returns: pd.DataFrame,
    optimizer_config: dict[str, Any] | None = None,
    lookback_window: int = 252,
    rebalance_frequency: str = "monthly",
    covariance_method: str = "sample",
    holding_return_method: str = "drifted_buy_and_hold",
    allow_weekend_rebalances: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run rolling minimum variance backtest with configurable holding method."""
    if optimizer_config is None:
        optimizer_config = load_optimizer_config()

    return run_rolling_backtest(
        returns=returns,
        optimizer_fn=minimise_variance,
        optimizer_config=optimizer_config,
        lookback_window=lookback_window,
        rebalance_frequency=rebalance_frequency,
        strategy_name="min_variance",
        optimizer_kwargs={"covariance_method": covariance_method},
        holding_return_method=holding_return_method,
        allow_weekend_rebalances=allow_weekend_rebalances,
    )
