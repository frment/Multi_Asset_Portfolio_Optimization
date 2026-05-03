from __future__ import annotations

import numpy as np
import pandas as pd

from src.benchmarks import run_benchmark_backtest


def _returns() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(7)
    data = rng.normal(0.0005, 0.01, size=(len(idx), 2))
    return pd.DataFrame(data, index=idx, columns=["SPY", "TLT"])


def test_sixty_forty_resets_monthly_and_drifts_in_between() -> None:
    returns = _returns()
    strategy_returns, weights_history, turnover = run_benchmark_backtest(
        returns=returns,
        strategy_name="sixty_forty",
        target_weights=pd.Series({"SPY": 0.6, "TLT": 0.4}),
        lookback_window=20,
        rebalance_frequency="monthly",
        holding_return_method="drifted_buy_and_hold",
        allow_weekend_rebalances=False,
    )
    assert len(weights_history) >= 2
    assert (weights_history[["SPY", "TLT"]].round(6) == pd.Series({"SPY": 0.6, "TLT": 0.4})).all(axis=1).all()
    assert (turnover["turnover_one_way"].iloc[1:] >= 0.0).all()
    assert len(strategy_returns) > 0


def test_equal_weight_drifted_not_equal_constant_mix_non_trivial() -> None:
    returns = _returns()
    drifted, _, _ = run_benchmark_backtest(
        returns=returns,
        strategy_name="eq",
        target_weights=pd.Series({"SPY": 0.5, "TLT": 0.5}),
        lookback_window=20,
        rebalance_frequency="monthly",
        holding_return_method="drifted_buy_and_hold",
        allow_weekend_rebalances=False,
    )
    constant, _, _ = run_benchmark_backtest(
        returns=returns,
        strategy_name="eq",
        target_weights=pd.Series({"SPY": 0.5, "TLT": 0.5}),
        lookback_window=20,
        rebalance_frequency="monthly",
        holding_return_method="constant_target_weights",
        allow_weekend_rebalances=False,
    )
    assert not np.allclose(drifted.values, constant.values)
