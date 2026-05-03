"""Tests for historical stress-window analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.stress import STRESS_COLUMNS, run_historical_stress_windows


def _returns(seed: int = 9) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=220)
    return {
        "minvar_baseline_ch1": pd.Series(rng.normal(0.0002, 0.01, size=len(idx)), index=idx),
        "cvar_baseline": pd.Series(rng.normal(0.00015, 0.009, size=len(idx)), index=idx),
    }


def test_stress_summary_schema_and_rows():
    windows = [
        {
            "window_id": "w1",
            "label": "Window 1",
            "start_date": "2020-03-01",
            "end_date": "2020-05-15",
        },
        {
            "window_id": "w2",
            "label": "Window 2",
            "start_date": "2020-06-01",
            "end_date": "2020-08-31",
        },
    ]

    out = run_historical_stress_windows(_returns(), windows, beta=0.95, scope="gross")

    assert list(out.columns) == STRESS_COLUMNS
    assert len(out) == 4
    assert set(out["window_id"]) == {"w1", "w2"}
    assert set(out["scope"]) == {"gross"}


def test_stress_raises_on_invalid_window_dates():
    windows = [
        {
            "window_id": "bad",
            "label": "Bad",
            "start_date": "2021-02-01",
            "end_date": "2021-01-01",
        }
    ]

    with pytest.raises(ValueError, match="start_date > end_date"):
        run_historical_stress_windows(_returns(), windows, beta=0.95)


def test_stress_handles_empty_overlap_by_skipping_rows():
    windows = [
        {
            "window_id": "future",
            "label": "Future",
            "start_date": "2030-01-01",
            "end_date": "2030-02-01",
        }
    ]

    out = run_historical_stress_windows(_returns(), windows, beta=0.95)
    assert out.empty
    assert list(out.columns) == STRESS_COLUMNS
