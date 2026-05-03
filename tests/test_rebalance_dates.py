from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import get_rebalance_dates


def _make_returns() -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=320)
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0, 0.01, size=(len(idx), 2)), index=idx, columns=["A", "B"])


def test_rebalance_dates_first_available_of_month() -> None:
    returns = _make_returns()
    dates = get_rebalance_dates(returns.index, lookback_window=60, rebalance_frequency="monthly")
    for date in dates:
        month_days = returns.index[(returns.index.year == date.year) & (returns.index.month == date.month)]
        assert date == month_days[0]


def test_no_weekend_rebalances_when_disabled() -> None:
    returns = _make_returns()
    dates = get_rebalance_dates(
        returns.index,
        lookback_window=60,
        rebalance_frequency="monthly",
        allow_weekend_rebalances=False,
    )
    assert all(d.dayofweek < 5 for d in dates)


def test_lookback_counts_valid_observations() -> None:
    returns = _make_returns()
    lookback = 120
    dates = get_rebalance_dates(returns.index, lookback_window=lookback)
    for date in dates:
        assert returns.index.get_loc(date) >= lookback


def test_first_rebalance_has_sufficient_history() -> None:
    returns = _make_returns()
    lookback = 126
    dates = get_rebalance_dates(returns.index, lookback_window=lookback)
    assert len(dates) > 0
    assert returns.index.get_loc(dates[0]) >= lookback


def test_weekend_date_raises_when_disabled() -> None:
    idx = pd.to_datetime(["2024-01-06", "2024-02-05", "2024-03-04", "2024-04-01"])
    with pytest.raises(ValueError, match="Weekend rebalance date"):
        get_rebalance_dates(idx, lookback_window=0, allow_weekend_rebalances=False)
