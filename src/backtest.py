"""Rolling walk-forward backtest engine.

This module implements a monthly-rebalanced, rolling lookback backtest for the
constrained minimum variance portfolio.  The design principle is strict temporal
separation: at every rebalance date, only returns observed *before* that date
are used to estimate the covariance and compute weights.  This guarantees there
is no look-ahead bias.

How it works (step by step):
1. Identify the first trading day of every calendar month (rebalance dates).
2. Skip months where we don't have at least `lookback_window` days of history.
3. On each rebalance date t:
     a. Extract the trailing lookback window: returns[t - lookback_window : t]
     b. Run the constrained minimum variance optimiser on that window.
     c. Apply the resulting weights to every day in the holding period
        [t, next_rebalance_date).  The holding period return for each day is
        simply the dot product of fixed weights × asset returns.
4. Concatenate all holding-period return series to form the full backtest history.

Turnover methodology (one-way, professional definition):
    turnover_one_way_t = 0.5 * sum_i( |w_target_t - w_pretrade_t| )

where w_pretrade_t are the weights *after* drift through the previous holding
period (i.e. the actual portfolio weights arriving at the new rebalance date,
before any trading).  This captures real economic turnover rather than a naive
target-to-target difference.  The first rebalance carries zero turnover because
there is no prior holding period.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.optimizer import load_optimizer_config, minimise_variance


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------

def get_rebalance_dates(
    index: pd.DatetimeIndex,
    lookback_window: int,
    rebalance_frequency: str = "monthly",
) -> list[pd.Timestamp]:
    """Return the first available trading day for each rebalance period.

    Supports monthly, quarterly, and weekly schedules.

    Only periods where at least `lookback_window` trading days of prior history
    exist are included, because the optimiser needs a full training window.

    Args:
        index           : DatetimeIndex of the full returns DataFrame.
        lookback_window : Minimum number of daily observations required before
                          a rebalance date to fit the covariance estimator.
        rebalance_frequency: Rebalance schedule. One of:
                          - "monthly"
                          - "quarterly"
                          - "weekly"

    Returns:
        Sorted list of rebalance timestamps (subset of `index`).

    Raises:
        ValueError: If `rebalance_frequency` is unsupported.
    """
    frequency = rebalance_frequency.lower().strip()

    # Group by period and take the first available trading day in each period.
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

    first_of_period = (
        index.to_series()
        .groupby(period_index)
        .first()
        .sort_values()
        .values
    )

    rebalance_dates = []
    for date in first_of_period:
        # How many rows of data exist *before* this date?
        pos = index.get_loc(date)
        if pos >= lookback_window:
            rebalance_dates.append(pd.Timestamp(date))

    return rebalance_dates


# ---------------------------------------------------------------------------
# Turnover helpers
# ---------------------------------------------------------------------------

def compute_pretrade_weights(
    target_weights: pd.Series,
    holding_returns: pd.DataFrame,
) -> pd.Series:
    """Compute the drifted (pre-trade) weights arriving at the next rebalance.

    Starting from ``target_weights`` applied at the opening of a holding period,
    propagate daily returns through the full holding period to find the actual
    portfolio weights just before the next rebalance trade.

    Args:
        target_weights : Weight vector (indexed by ticker) at the start of the
                         holding period.
        holding_returns: Daily return DataFrame for the holding period.  The
                         row index must cover every day from the first day of
                         the period through the day *before* the next rebalance.

    Returns:
        Drifted weight Series (same ticker index as ``target_weights``),
        normalised to sum to 1.
    """
    tickers = target_weights.index.tolist()
    # Gross return of each asset over the holding period: product of (1 + r_t)
    gross_returns = (1.0 + holding_returns[tickers]).prod()
    # Nominal value of each position (starting from weight as unit portfolio)
    drifted_values = target_weights * gross_returns
    total_value = drifted_values.sum()
    if total_value <= 0:
        # Degenerate: return equal weights as a safe fallback.
        return pd.Series(1.0 / len(tickers), index=tickers)
    return drifted_values / total_value


def compute_turnover_one_way(
    target_weights: pd.Series,
    pretrade_weights: pd.Series,
) -> float:
    """Compute one-way portfolio turnover at a single rebalance.

    Definition:
        turnover_one_way = 0.5 * sum_i( |w_target_i - w_pretrade_i| )

    This measures the fraction of the portfolio that must be traded to move
    from the drifted pre-trade allocation to the new target allocation.

    Args:
        target_weights  : New target weights from the optimiser.
        pretrade_weights: Drifted weights arriving at this rebalance date
                          (output of ``compute_pretrade_weights``).

    Returns:
        Scalar in [0, 1].
    """
    aligned_target   = target_weights.reindex(pretrade_weights.index, fill_value=0.0)
    aligned_pretrade = pretrade_weights.reindex(target_weights.index, fill_value=0.0)
    # Use the union of all tickers to handle any index mismatch gracefully.
    all_tickers = aligned_target.index.union(aligned_pretrade.index)
    t = aligned_target.reindex(all_tickers, fill_value=0.0)
    p = aligned_pretrade.reindex(all_tickers, fill_value=0.0)
    return float(0.5 * np.abs(t - p).sum())


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_min_variance_backtest(
    returns: pd.DataFrame,
    optimizer_config: dict[str, Any] | None = None,
    lookback_window: int = 252,
    rebalance_frequency: str = "monthly",
    covariance_method: str = "sample",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Run a rolling minimum variance backtest.

    At each rebalance date the optimiser sees only the trailing lookback window
    (strictly before the rebalance date).  The resulting weights are then held
    fixed until the next rebalance date.

    Turnover is computed against pre-trade drifted weights (not naive
    target-to-target differences).  The first rebalance always has
    ``turnover_one_way = 0.0`` and ``is_initial_rebalance = True``.

    Args:
        returns          : Daily returns DataFrame.  Columns are asset tickers.
        optimizer_config : Constraint config dict from ``load_optimizer_config()``.
                           If None, loaded from YAML automatically.
        lookback_window  : Number of trading days used as the estimation window.
        rebalance_frequency: Rebalance schedule. One of ``monthly``, ``quarterly``,
                   or ``weekly``.
        covariance_method: Covariance estimator for the optimiser. One of
               ``sample`` or ``ledoit_wolf``.

    Returns:
        A tuple of:
        - portfolio_returns  : Daily portfolio return Series over the OOS period.
        - weights_history    : DataFrame with one row per rebalance date; columns
                               are asset tickers.
        - turnover_history   : DataFrame with one row per rebalance date and
                               columns: ``turnover_one_way``, ``is_initial_rebalance``,
                               ``n_assets_changed``, ``max_abs_weight_change``.
    """
    if optimizer_config is None:
        optimizer_config = load_optimizer_config()

    rebalance_dates = get_rebalance_dates(
        returns.index,
        lookback_window,
        rebalance_frequency=rebalance_frequency,
    )

    if not rebalance_dates:
        raise ValueError(
            f"No valid rebalance dates found.  "
            f"Need at least {lookback_window} rows of history before first rebalance."
        )

    portfolio_return_segments: list[pd.Series] = []
    weights_records: list[dict] = []
    turnover_records: list[dict] = []

    # Track the (target_weights, holding_returns) of the previous period so we
    # can compute pre-trade drifted weights at the start of the next period.
    prev_target_weights: pd.Series | None = None
    prev_holding_returns: pd.DataFrame | None = None

    for i, rebal_date in enumerate(rebalance_dates):
        # --- Training window: strictly before the rebalance date ---------------
        rebal_pos = returns.index.get_loc(rebal_date)
        training_window = returns.iloc[rebal_pos - lookback_window : rebal_pos]

        # --- Optimise weights on the training window ---------------------------
        try:
            weights = minimise_variance(
                training_window,
                config=optimizer_config,
                covariance_method=covariance_method,
            )
        except ValueError as exc:
            print(f"  Warning: optimisation failed on {rebal_date.date()} — {exc}")
            continue

        # --- Determine holding period ------------------------------------------
        if i + 1 < len(rebalance_dates):
            next_rebal_date = rebalance_dates[i + 1]
            holding_returns = returns.loc[rebal_date:next_rebal_date].iloc[:-1]
        else:
            holding_returns = returns.loc[rebal_date:]

        if holding_returns.empty:
            continue

        # --- Compute portfolio return for the holding period -------------------
        tickers = weights.index.tolist()
        daily_port_return = (holding_returns[tickers] * weights.values).sum(axis=1)
        daily_port_return.name = "min_variance"
        portfolio_return_segments.append(daily_port_return)

        # --- Record the target weights for this rebalance date ----------------
        record = {"rebalance_date": rebal_date}
        record.update(weights.to_dict())
        weights_records.append(record)

        # --- Compute turnover --------------------------------------------------
        is_initial = (i == 0) or (prev_target_weights is None)

        if is_initial:
            turnover = 0.0
            n_changed = 0
            max_abs_change = 0.0
        else:
            # Drift prev_target_weights through the previous holding period to
            # get the actual portfolio composition arriving at this rebalance.
            pretrade = compute_pretrade_weights(prev_target_weights, prev_holding_returns)
            turnover = compute_turnover_one_way(weights, pretrade)
            abs_changes = (weights.reindex(pretrade.index, fill_value=0.0)
                           - pretrade.reindex(weights.index, fill_value=0.0)).abs()
            n_changed = int((abs_changes > 1e-6).sum())
            max_abs_change = float(abs_changes.max())

        turnover_records.append({
            "rebalance_date":       rebal_date,
            "turnover_one_way":     turnover,
            "is_initial_rebalance": is_initial,
            "n_assets_changed":     n_changed,
            "max_abs_weight_change": max_abs_change,
        })

        # Keep for the next iteration.
        prev_target_weights = weights
        prev_holding_returns = holding_returns

    # --- Assemble final outputs ------------------------------------------------
    portfolio_returns = pd.concat(portfolio_return_segments).sort_index()
    portfolio_returns.name = "min_variance"

    weights_history = pd.DataFrame(weights_records).set_index("rebalance_date")
    weights_history.index.name = "rebalance_date"

    turnover_history = pd.DataFrame(turnover_records).set_index("rebalance_date")
    turnover_history.index.name = "rebalance_date"

    return portfolio_returns, weights_history, turnover_history
