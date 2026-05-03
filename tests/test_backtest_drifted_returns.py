from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import (
    _compute_daily_returns_constant_mix,
    _compute_daily_returns_drifted_buy_and_hold,
    compute_pretrade_weights,
    compute_turnover_one_way,
)


def test_drifted_weights_change_when_one_asset_outperforms() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.10, 0.10, 0.10],
            "B": [0.00, 0.00, 0.00],
        },
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    start = pd.Series({"A": 0.5, "B": 0.5})
    _, end_weights = _compute_daily_returns_drifted_buy_and_hold(returns, start)
    assert end_weights["A"] > start["A"]
    assert end_weights["B"] < start["B"]


def test_drifted_path_differs_from_constant_mix() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.15, -0.05, 0.10],
            "B": [0.00, 0.00, 0.00],
        },
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    start = pd.Series({"A": 0.5, "B": 0.5})
    drifted, _ = _compute_daily_returns_drifted_buy_and_hold(returns, start)
    constant, _ = _compute_daily_returns_constant_mix(returns, start)
    assert not np.allclose(drifted.values, constant.values)


def test_turnover_uses_drifted_pretrade_weights() -> None:
    prev_target = pd.Series({"A": 0.5, "B": 0.5})
    holding = pd.DataFrame(
        {"A": [0.2, 0.1], "B": [0.0, 0.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    pretrade = compute_pretrade_weights(prev_target, holding)
    new_target = pd.Series({"A": 0.5, "B": 0.5})

    turnover = compute_turnover_one_way(new_target, pretrade)
    naive = compute_turnover_one_way(new_target, prev_target)

    assert turnover > 0.0
    assert turnover > naive
