from __future__ import annotations

import numpy as np
import pandas as pd

from src.supervised_targets import (
    make_forward_drawdown,
    make_forward_drawdown_event,
    make_forward_realized_vol,
)


def test_forward_vol_no_leakage_and_terminal_nans() -> None:
    idx = pd.bdate_range("2024-01-01", periods=10)
    r = pd.Series(np.arange(10, dtype=float) / 100.0, index=idx)

    vol = make_forward_realized_vol(r, horizon=3, annualization_factor=1.0)

    expected_first = pd.Series([0.01, 0.02, 0.03]).std()
    assert np.isclose(vol.iloc[0], expected_first)
    assert vol.iloc[-3:].isna().all()


def test_forward_drawdown_known_path() -> None:
    idx = pd.bdate_range("2024-01-01", periods=6)
    # At t=0, forward window is [-10%, +1%, +1%], so max DD is -10%
    r = pd.Series([0.0, -0.10, 0.01, 0.01, 0.0, 0.0], index=idx)

    dd = make_forward_drawdown(r, horizon=3)
    assert np.isclose(dd.iloc[0], -0.10)


def test_drawdown_event_and_terminal_nans() -> None:
    idx = pd.bdate_range("2024-01-01", periods=7)
    r = pd.Series([0.0, -0.10, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)

    event = make_forward_drawdown_event(r, horizon=3, threshold=-0.05)
    assert event.iloc[0] == 1.0
    assert event.iloc[-3:].isna().all()
