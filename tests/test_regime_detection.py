"""
Tests for src/regime_detection.py — Chapter 4, Phase 2.

Covers:
1. Reproducibility: same labels on two calls with identical random_state.
2. Number of labels equals number of feature rows.
3. Labels contain no NaNs.
4. Transition matrix is square and row-stochastic.
5. Rows of the transition matrix sum to 1 (within float tolerance).
6. Each regime has share > 0.
7. Stress-score ordering is stable: state 0 ≤ state 1 by stress score.
8. Explicit fallback to KMeans when HMM is requested but unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from regime_detection import (  # noqa: E402
    RegimeResult,
    _compute_stress_scores,
    _empirical_transition_matrix,
    _reorder_states_by_stress,
    _standardise,
    detect_regimes,
)

# ---------------------------------------------------------------------------
# Shared test config (mirrors config/regime_analysis.yaml)
# ---------------------------------------------------------------------------

_CFG = {
    "paths": {
        "output_dir": "data/processed/regime_analysis",
        "regime_features": "data/processed/regime_analysis/regime_features.csv",
    },
    "detection": {
        "random_state": 42,
        "primary_model": "hmm",
        "n_states_default": 2,
        "hmm": {
            "covariance_type": "full",
            "n_iter": 100,
            "tol": 1e-4,
        },
        "candidate_models": [
            {"model": "kmeans", "n_states": 2},
            {"model": "kmeans", "n_states": 3},
            {"model": "hmm",    "n_states": 2},
            {"model": "hmm",    "n_states": 3},
        ],
        "stress_score_features": {
            "realized_vol_spy_63d":      1.0,
            "realized_vol_btc_usd_63d":  1.0,
            "drawdown_spy_126d":        -1.0,
            "drawdown_btc_usd_126d":    -1.0,
            "corr_spy_btc_usd_126d":    -1.0,
            "corr_spy_tlt_126d":         1.0,
        },
        "state_names": {
            2: {0: "Low-stress / Risk-on", 1: "High-stress / Risk-off"},
            3: {0: "Low-stress / Risk-on", 1: "High-stress / Risk-off", 2: "Mixed / Transition"},
        },
        "outputs": {
            "regime_labels":            "data/processed/regime_analysis/regime_labels.csv",
            "regime_model_summary":     "data/processed/regime_analysis/regime_model_summary.csv",
            "regime_transition_matrix": "data/processed/regime_analysis/regime_transition_matrix.csv",
        },
    },
}

# Fallback config that forces KMeans as primary (for test 8)
_CFG_KMEANS_PRIMARY = {
    **_CFG,
    "detection": {**_CFG["detection"], "primary_model": "kmeans"},
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
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


def _make_features(n_rows: int = 400, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic feature DataFrame with realistic-ish structure."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n_rows, freq="B")
    # Vols: positive, ~0.1–0.5 range
    spy_vol  = 0.15 + 0.10 * rng.standard_normal(n_rows) ** 2
    btc_vol  = 0.60 + 0.30 * rng.standard_normal(n_rows) ** 2
    # Drawdowns: negative
    spy_dd   = -np.abs(rng.uniform(0.0, 0.20, n_rows))
    btc_dd   = -np.abs(rng.uniform(0.0, 0.50, n_rows))
    # Correlations: in [-1, 1]
    corr_st  = np.clip(rng.normal(-0.10, 0.30, n_rows), -1, 1)
    corr_sb  = np.clip(rng.normal(0.20, 0.25, n_rows), -1, 1)
    corr_be  = np.clip(rng.normal(0.70, 0.15, n_rows), -1, 1)
    # Momenta
    mom_spy  = rng.normal(0.05, 0.10, n_rows)
    mom_btc  = rng.normal(0.10, 0.30, n_rows)
    mom_tlt  = rng.normal(0.02, 0.06, n_rows)

    data = np.column_stack([spy_vol, btc_vol, spy_dd, btc_dd,
                             corr_st, corr_sb, corr_be,
                             mom_spy, mom_btc, mom_tlt])
    return pd.DataFrame(data, index=dates, columns=_FEATURE_COLS)


@pytest.fixture(scope="module")
def sample_features() -> pd.DataFrame:
    return _make_features()


@pytest.fixture(scope="module")
def result(sample_features: pd.DataFrame) -> RegimeResult:
    return detect_regimes(sample_features, _CFG)


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------

def test_reproducibility(sample_features: pd.DataFrame) -> None:
    """Two calls with the same config must produce identical labels."""
    r1 = detect_regimes(sample_features, _CFG)
    r2 = detect_regimes(sample_features, _CFG)
    pd.testing.assert_series_equal(r1.labels, r2.labels)


# ---------------------------------------------------------------------------
# 2. Label count matches feature row count
# ---------------------------------------------------------------------------

def test_label_count_matches_rows(sample_features: pd.DataFrame, result: RegimeResult) -> None:
    """Number of daily labels must equal number of feature rows."""
    assert len(result.labels) == len(sample_features), (
        f"Labels: {len(result.labels)}, features: {len(sample_features)}"
    )


# ---------------------------------------------------------------------------
# 3. No NaNs in labels
# ---------------------------------------------------------------------------

def test_labels_no_nans(result: RegimeResult) -> None:
    """Labels series must contain no NaN values."""
    assert result.labels.isna().sum() == 0, "Labels contain NaN values."


# ---------------------------------------------------------------------------
# 4 & 5. Transition matrix: square and row-stochastic
# ---------------------------------------------------------------------------

def test_transition_matrix_is_square(result: RegimeResult) -> None:
    n = result.n_states
    assert result.transition_matrix.shape == (n, n), (
        f"Expected ({n}, {n}), got {result.transition_matrix.shape}"
    )


def test_transition_matrix_rows_sum_to_one(result: RegimeResult) -> None:
    """Every row of the transition matrix must sum to ~1.0."""
    row_sums = result.transition_matrix.values.sum(axis=1)
    np.testing.assert_allclose(
        row_sums, np.ones(result.n_states), atol=1e-9,
        err_msg=f"Row sums: {row_sums}"
    )


def test_transition_matrix_non_negative(result: RegimeResult) -> None:
    """All entries in the transition matrix must be ≥ 0."""
    assert (result.transition_matrix.values >= 0).all(), (
        "Transition matrix contains negative entries."
    )


# ---------------------------------------------------------------------------
# 6. Each regime has share > 0
# ---------------------------------------------------------------------------

def test_each_regime_has_positive_share(result: RegimeResult) -> None:
    """Every state must appear at least once in the label series."""
    for s in range(result.n_states):
        count = int((result.labels == s).sum())
        assert count > 0, f"State {s} has zero observations."


# ---------------------------------------------------------------------------
# 7. Stress-score ordering is stable (state 0 ≤ state 1)
# ---------------------------------------------------------------------------

def test_stress_score_ordering(sample_features: pd.DataFrame, result: RegimeResult) -> None:
    """State 0 must have a lower stress score than state 1 after reordering."""
    X_std, _ = _standardise(sample_features)
    feature_names = list(sample_features.columns)
    stress_weights = _CFG["detection"]["stress_score_features"]

    per_state_means = np.array([
        X_std[result.labels.values == s].mean(axis=0)
        if (result.labels.values == s).any() else np.zeros(len(feature_names))
        for s in range(result.n_states)
    ])
    scores = _compute_stress_scores(per_state_means, feature_names, stress_weights)
    assert scores[0] <= scores[1], (
        f"State 0 stress score ({scores[0]:.4f}) is NOT ≤ state 1 ({scores[1]:.4f}). "
        "Stress ordering failed."
    )


# ---------------------------------------------------------------------------
# 8. Explicit KMeans fallback when HMM is unavailable
# ---------------------------------------------------------------------------

def test_kmeans_fallback_when_hmm_unavailable(sample_features: pd.DataFrame) -> None:
    """When hmmlearn is not available, the model must fall back to KMeans gracefully."""
    with patch("regime_detection._HMM_AVAILABLE", False):
        r = detect_regimes(sample_features, _CFG)
    assert "kmeans" in r.model_id, (
        f"Expected kmeans fallback, got model_id='{r.model_id}'"
    )
    assert r.labels.isna().sum() == 0, "Fallback KMeans produced NaN labels."
    assert len(r.labels) == len(sample_features)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_empirical_transition_matrix_row_stochastic() -> None:
    """_empirical_transition_matrix must always produce row-stochastic output."""
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    T = _empirical_transition_matrix(labels, n_states=2)
    row_sums = T.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-12)


def test_reorder_states_by_stress_ascending() -> None:
    """After reordering, states must be assigned new IDs in ascending stress order."""
    raw_labels = np.array([0, 0, 1, 1, 0])
    # Old state 1 has higher stress than old state 0.
    stress_scores = np.array([2.0, 5.0])
    new_labels, sorted_ids = _reorder_states_by_stress(raw_labels, stress_scores)
    # Old state 0 (low stress) should map to new id 0.
    # Old state 1 (high stress) should map to new id 1.
    assert list(sorted_ids) == [0, 1]
    np.testing.assert_array_equal(new_labels, raw_labels)


def test_reorder_states_by_stress_swap() -> None:
    """Reordering must swap labels when the higher-stress state has a lower raw id."""
    raw_labels = np.array([0, 0, 1, 1, 0])
    # Old state 0 is high stress, old state 1 is low stress.
    stress_scores = np.array([5.0, 2.0])
    new_labels, sorted_ids = _reorder_states_by_stress(raw_labels, stress_scores)
    # Old state 1 (low stress) → new id 0; old state 0 (high stress) → new id 1.
    assert list(sorted_ids) == [1, 0]
    expected = np.array([1, 1, 0, 0, 1])
    np.testing.assert_array_equal(new_labels, expected)
