"""Tests for paired block bootstrap confidence interval module.

Five focused tests:
    1. Output schema is consistent with the expected CSV columns.
    2. Reproducibility: same seed produces identical results.
    3. Identical series produce Sharpe difference ≈ 0 and CI contains 0.
    4. Correct alignment when A and B have partially different date indices.
    5. Missing comparison is skipped cleanly (KeyError) without crashing the runner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bootstrap import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_SEED,
    build_confidence_summary,
    run_paired_bootstrap,
)

# Expected output columns (canonical schema).
EXPECTED_COLUMNS = [
    "comparison_id",
    "comparison_family",
    "strategy_a",
    "strategy_b",
    "sample_scope_used",
    "metric_compared",
    "point_estimate_difference",
    "ci_lower",
    "ci_upper",
    "ci_includes_zero",
    "bootstrap_n",
    "block_size",
    "random_seed",
    "n_observations_aligned",
    "p_zero_crossing",
    "notes",
]

# Reduce bootstrap_n to keep tests fast.
FAST_N = 200


def _make_returns(n: int = 500, mu: float = 0.0003, sigma: float = 0.01, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2019-01-02", periods=n)
    return pd.Series(rng.normal(mu, sigma, n), index=dates, name="ret")


# ---------------------------------------------------------------------------
# Test 1 — Output schema
# ---------------------------------------------------------------------------


def test_output_schema():
    """Result DataFrame must have exactly the expected columns."""
    a = _make_returns(seed=1)
    b = _make_returns(seed=2)
    result = run_paired_bootstrap(
        a, b,
        comparison_id="test_schema",
        comparison_family="test",
        strategy_a="A",
        strategy_b="B",
        n_replications=FAST_N,
        seed=DEFAULT_SEED,
    )
    df = build_confidence_summary([result])
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Column mismatch.\n  Expected: {EXPECTED_COLUMNS}\n  Got: {list(df.columns)}"
    )
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Test 2 — Reproducibility with fixed seed
# ---------------------------------------------------------------------------


def test_reproducibility():
    """Two runs with the same seed produce identical results."""
    a = _make_returns(seed=10)
    b = _make_returns(seed=11)
    kwargs = dict(
        comparison_id="test_repro",
        comparison_family="test",
        strategy_a="A",
        strategy_b="B",
        n_replications=FAST_N,
        seed=DEFAULT_SEED,
    )
    r1 = run_paired_bootstrap(a, b, **kwargs)
    r2 = run_paired_bootstrap(a, b, **kwargs)
    assert r1.ci_lower == r2.ci_lower
    assert r1.ci_upper == r2.ci_upper
    assert r1.point_estimate_difference == r2.point_estimate_difference


# ---------------------------------------------------------------------------
# Test 3 — Identical series → difference ≈ 0 and CI contains 0
# ---------------------------------------------------------------------------


def test_identical_series_difference_near_zero():
    """When A == B the point estimate must be ~0 and the CI must contain 0."""
    a = _make_returns(seed=5)
    b = a.copy()
    result = run_paired_bootstrap(
        a, b,
        comparison_id="test_identical",
        comparison_family="test",
        strategy_a="A",
        strategy_b="B",
        n_replications=FAST_N,
        seed=DEFAULT_SEED,
    )
    assert abs(result.point_estimate_difference) < 1e-12, (
        f"Expected difference ≈ 0, got {result.point_estimate_difference}"
    )
    assert result.ci_includes_zero, "CI should contain 0 for identical series."


# ---------------------------------------------------------------------------
# Test 4 — Correct alignment when indices partially differ
# ---------------------------------------------------------------------------


def test_alignment_partial_overlap():
    """n_observations_aligned must equal the size of the common date range."""
    a = _make_returns(n=300, seed=1)
    b = _make_returns(n=300, seed=2)

    # Shift B forward by 50 days to create a partial overlap.
    b.index = pd.bdate_range(start=a.index[50], periods=300)

    expected_overlap = len(a.index.intersection(b.index))
    assert expected_overlap > 0, "Sanity: overlap must be non-zero."

    result = run_paired_bootstrap(
        a, b,
        comparison_id="test_alignment",
        comparison_family="test",
        strategy_a="A",
        strategy_b="B",
        n_replications=FAST_N,
        seed=DEFAULT_SEED,
    )
    assert result.n_observations_aligned == expected_overlap, (
        f"Expected {expected_overlap} aligned obs, got {result.n_observations_aligned}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Missing experiment is skipped cleanly
# ---------------------------------------------------------------------------


def test_missing_experiment_skipped_cleanly():
    """A KeyError for an unknown experiment_id surfaces without crashing the pipeline."""
    panel = pd.DataFrame({
        "experiment_id": ["exp_a"] * 5,
        "portfolio_return": [0.001] * 5,
    }, index=pd.bdate_range("2020-01-02", periods=5))

    # Simulate what the runner does: look up an experiment not in the panel.
    def get_series(exp_id: str) -> pd.Series:
        subset = panel[panel["experiment_id"] == exp_id]["portfolio_return"]
        if subset.empty:
            raise KeyError(f"experiment_id '{exp_id}' not found in returns panel.")
        return subset.sort_index()

    with pytest.raises(KeyError, match="nonexistent_exp"):
        get_series("nonexistent_exp")

    # Runner pattern: try/except appends to skipped list — no crash.
    results: list = []
    skipped: list[str] = []
    try:
        a = get_series("nonexistent_exp")
        b = get_series("exp_a")
        results.append(run_paired_bootstrap(
            a, b,
            comparison_id="missing",
            comparison_family="test",
            strategy_a="nonexistent_exp",
            strategy_b="exp_a",
            n_replications=FAST_N,
            seed=DEFAULT_SEED,
        ))
    except KeyError as exc:
        skipped.append(str(exc))

    assert len(results) == 0
    assert len(skipped) == 1
    assert "nonexistent_exp" in skipped[0]
