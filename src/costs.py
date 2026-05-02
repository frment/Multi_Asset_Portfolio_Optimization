"""Transaction-cost helpers for simple net-of-cost evaluation.

Current model is intentionally minimal and strategy-level:
    cost_t = turnover_one_way_t * cost_rate

where cost_rate is a decimal (for example 10 bps = 0.0010).
"""

from __future__ import annotations

import pandas as pd


def bps_to_rate(cost_bps: float) -> float:
    """Convert basis points to decimal cost rate.

    Example: 10 bps -> 0.0010
    """
    return float(cost_bps) / 10_000.0


def build_rebalance_cost_series(
    turnover_by_rebalance: pd.Series,
    cost_rate: float,
) -> pd.Series:
    """Return per-rebalance cost amounts from one-way turnover.

    Args:
        turnover_by_rebalance: Series indexed by rebalance date with one-way turnover.
        cost_rate: Decimal transaction cost rate.

    Returns:
        Series indexed by rebalance date with cost amounts.
    """
    if turnover_by_rebalance.empty:
        return turnover_by_rebalance.copy()
    return turnover_by_rebalance.astype(float) * float(cost_rate)


def apply_rebalance_costs_to_daily_returns(
    gross_daily_returns: pd.Series,
    rebalance_costs: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Apply rebalance costs on matching dates in a daily return series.

    Costs are subtracted on the rebalance date itself.

    Returns:
        net_daily_returns: gross returns minus costs on rebalance dates.
        aligned_daily_costs: daily series of applied costs aligned to gross index.
    """
    net = gross_daily_returns.copy().astype(float)
    aligned_costs = pd.Series(0.0, index=net.index, name="transaction_cost")

    if rebalance_costs.empty:
        return net, aligned_costs

    # Apply only where dates overlap.
    common_idx = net.index.intersection(rebalance_costs.index)
    if len(common_idx) > 0:
        applied = rebalance_costs.reindex(common_idx).astype(float)
        net.loc[common_idx] = net.loc[common_idx] - applied.values
        aligned_costs.loc[common_idx] = applied.values

    return net, aligned_costs
