"""
Tests for src/regime_features.py — Chapter 4, Phase 1.

Covers:
1. Expected output shape (10 feature columns).
2. Monotonically increasing date index.
3. No NaNs anywhere in the output.
4. No look-ahead in realized volatility (causal check).
5. No look-ahead in momentum (causal check).
6. Rolling correlations bounded to [-1, 1].
7. Drawdowns are non-positive (≤ 0).
8. Deterministic generation (same output on repeated calls).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable when running tests from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regime_features import build_regime_features  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "realized_vol_spy_63d",
    "realized_vol_btc_usd_63d",
    "drawdown_spy_126d",
    "drawdown_btc_usd_126d",
    "corr_spy_tlt_126d",
    "corr_spy_btc_usd_126d",
    "corr_btc_usd_eth_usd_126d",
    "momentum_spy_126d",
    "momentum_btc_usd_126d",
    "momentum_tlt_126d",
]

# Minimal config that mirrors config/regime_analysis.yaml
_SAMPLE_CFG = {
    "features": {
        "realized_volatility": {
            "window": 63,
            "min_periods": 21,
            "annualization_factor": 252,
            "tickers": ["SPY", "BTC-USD"],
        },
        "drawdown": {
            "window": 126,
            "min_periods": 63,
            "tickers": ["SPY", "BTC-USD"],
        },
        "correlation": {
            "window": 126,
            "min_periods": 63,
            "pairs": [["SPY", "TLT"], ["SPY", "BTC-USD"], ["BTC-USD", "ETH-USD"]],
        },
        "momentum": {
            "window": 126,
            "min_periods": 63,
            "tickers": ["SPY", "BTC-USD", "TLT"],
        },
    },
    "nan_handling": {"strategy": "dropna"},
}

# Number of rows required before the longest rolling window (126d, min 63) starts
# producing valid values — used to size synthetic data.
_MIN_ROWS = 300

TICKERS = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD", "TLT"]


def _make_returns(n_rows: int = _MIN_ROWS, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily simple-return DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_rows, freq="B")
    data = rng.normal(0.0, 0.01, size=(n_rows, len(TICKERS)))
    return pd.DataFrame(data, index=dates, columns=TICKERS)


@pytest.fixture(scope="module")
def sample_returns() -> pd.DataFrame:
    return _make_returns()


@pytest.fixture(scope="module")
def features(sample_returns: pd.DataFrame) -> pd.DataFrame:
    return build_regime_features(sample_returns, _SAMPLE_CFG)


# ---------------------------------------------------------------------------
# 1. Expected shape
# ---------------------------------------------------------------------------

def test_output_has_ten_feature_columns(features: pd.DataFrame) -> None:
    """Output must have exactly 10 feature columns."""
    assert features.shape[1] == 10, (
        f"Expected 10 columns, got {features.shape[1]}: {list(features.columns)}"
    )


def test_output_column_names(features: pd.DataFrame) -> None:
    """Column names must match the expected feature catalogue."""
    assert list(features.columns) == EXPECTED_COLUMNS, (
        f"Column mismatch.\nExpected: {EXPECTED_COLUMNS}\nGot: {list(features.columns)}"
    )


def test_output_has_rows(features: pd.DataFrame) -> None:
    """After NaN drop, the output must still contain rows."""
    assert len(features) > 0, "Feature panel is empty after NaN drop."


# ---------------------------------------------------------------------------
# 2. Monotonically increasing date index
# ---------------------------------------------------------------------------

def test_index_is_monotonic(features: pd.DataFrame) -> None:
    """Date index must be strictly increasing (no duplicates, no reversals)."""
    assert features.index.is_monotonic_increasing, "Date index is not monotonically increasing."


def test_index_has_no_duplicates(features: pd.DataFrame) -> None:
    """Date index must contain no duplicate dates."""
    assert features.index.is_unique, "Date index contains duplicate dates."


# ---------------------------------------------------------------------------
# 3. No NaNs in output
# ---------------------------------------------------------------------------

def test_no_nans_in_output(features: pd.DataFrame) -> None:
    """The final output must have zero NaN values in any column."""
    nan_counts = features.isna().sum()
    assert nan_counts.sum() == 0, (
        f"NaNs found in output:\n{nan_counts[nan_counts > 0]}"
    )


# ---------------------------------------------------------------------------
# 4. No look-ahead in realized volatility
# ---------------------------------------------------------------------------

def test_realized_vol_no_leakage(sample_returns: pd.DataFrame) -> None:
    """Removing the last row of returns must not change any earlier vol value.

    If vol at t used data beyond t, removing the last observation would alter
    the value computed at t-1.
    """
    full = build_regime_features(sample_returns, _SAMPLE_CFG)
    truncated = build_regime_features(sample_returns.iloc[:-1], _SAMPLE_CFG)

    # Intersect dates that exist in both outputs.
    common_idx = full.index.intersection(truncated.index)
    assert len(common_idx) > 0, "No common dates after truncation."

    vol_col = "realized_vol_spy_63d"
    diff = (full.loc[common_idx, vol_col] - truncated.loc[common_idx, vol_col]).abs()
    assert diff.max() < 1e-12, (
        f"Realized vol changed after removing last row (max diff={diff.max():.2e}). "
        "Possible look-ahead in rolling window."
    )


# ---------------------------------------------------------------------------
# 5. No look-ahead in momentum
# ---------------------------------------------------------------------------

def test_momentum_no_leakage(sample_returns: pd.DataFrame) -> None:
    """Removing the last return row must not alter any earlier momentum value."""
    full = build_regime_features(sample_returns, _SAMPLE_CFG)
    truncated = build_regime_features(sample_returns.iloc[:-1], _SAMPLE_CFG)

    common_idx = full.index.intersection(truncated.index)
    mom_col = "momentum_spy_126d"
    diff = (full.loc[common_idx, mom_col] - truncated.loc[common_idx, mom_col]).abs()
    assert diff.max() < 1e-12, (
        f"Momentum changed after removing last row (max diff={diff.max():.2e}). "
        "Possible look-ahead."
    )


# ---------------------------------------------------------------------------
# 6. Rolling correlations within [-1, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", [
    "corr_spy_tlt_126d",
    "corr_spy_btc_usd_126d",
    "corr_btc_usd_eth_usd_126d",
])
def test_correlations_bounded(features: pd.DataFrame, col: str) -> None:
    """All rolling correlation values must lie in [-1, 1]."""
    series = features[col]
    assert series.min() >= -1.0 - 1e-9, (
        f"{col}: min value {series.min():.6f} is below -1."
    )
    assert series.max() <= 1.0 + 1e-9, (
        f"{col}: max value {series.max():.6f} is above +1."
    )


# ---------------------------------------------------------------------------
# 7. Drawdowns are non-positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", ["drawdown_spy_126d", "drawdown_btc_usd_126d"])
def test_drawdowns_non_positive(features: pd.DataFrame, col: str) -> None:
    """Drawdown values must be ≤ 0 (cannot represent a gain above the rolling peak)."""
    max_val = features[col].max()
    assert max_val <= 1e-9, (
        f"{col}: max drawdown value {max_val:.6f} is positive. "
        "Drawdown should always be ≤ 0."
    )


# ---------------------------------------------------------------------------
# 8. Deterministic generation
# ---------------------------------------------------------------------------

def test_deterministic_generation(sample_returns: pd.DataFrame) -> None:
    """Calling build_regime_features twice on the same data must yield identical results."""
    first = build_regime_features(sample_returns, _SAMPLE_CFG)
    second = build_regime_features(sample_returns, _SAMPLE_CFG)
    pd.testing.assert_frame_equal(first, second)
