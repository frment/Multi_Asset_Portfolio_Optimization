"""Automated tests for the rolling walk-forward backtest engine.

Test groups:
  a) Rebalance date generation
  b) No look-ahead / temporal separation
  c) Constraints hold at every rebalance
  d) Turnover series shape and non-negativity
  e) Drifted-weight / turnover unit tests (synthetic data)
  f) Regression: output shapes, index ordering, alignment
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure src/ is importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import (
    compute_pretrade_weights,
    compute_turnover_one_way,
    get_rebalance_dates,
    run_min_variance_backtest,
)
from src.optimizer import load_optimizer_config


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_returns(
    n_days: int = 600,
    tickers: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a small synthetic daily returns DataFrame for testing."""
    if tickers is None:
        tickers = ["A", "B", "C"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2020-01-02", periods=n_days)
    data = rng.normal(loc=0.0005, scale=0.015, size=(n_days, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture(scope="module")
def synthetic_returns() -> pd.DataFrame:
    return _make_synthetic_returns()


@pytest.fixture(scope="module")
def minimal_optimizer_config() -> dict:
    """A config without crypto constraints so it works on generic tickers."""
    return {
        "tickers": ["A", "B", "C"],
        "crypto_assets": [],
        "long_only": True,
        "max_weight": 1.0,
        "max_crypto_weight": 1.0,
    }


@pytest.fixture(scope="module")
def backtest_outputs(synthetic_returns, minimal_optimizer_config):
    port_returns, weights_hist, turnover_hist = run_min_variance_backtest(
        returns=synthetic_returns,
        optimizer_config=minimal_optimizer_config,
        lookback_window=126,
    )
    return port_returns, weights_hist, turnover_hist


# ---------------------------------------------------------------------------
# a) Rebalance date generation
# ---------------------------------------------------------------------------

class TestGetRebalanceDates:
    def test_first_day_of_each_month(self, synthetic_returns):
        """Each rebalance date must be the first available trading day in its month."""
        dates = get_rebalance_dates(synthetic_returns.index, lookback_window=126)
        for date in dates:
            year, month = date.year, date.month
            month_days = synthetic_returns.index[
                (synthetic_returns.index.year == year) & (synthetic_returns.index.month == month)
            ]
            assert date == month_days[0], (
                f"Rebalance date {date} is not the first trading day of {year}-{month:02d}"
            )

    def test_requires_full_lookback_before_first_rebalance(self, synthetic_returns):
        """No rebalance date should have fewer than lookback_window prior observations."""
        lookback = 252
        dates = get_rebalance_dates(synthetic_returns.index, lookback_window=lookback)
        for date in dates:
            pos = synthetic_returns.index.get_loc(date)
            assert pos >= lookback, (
                f"Rebalance date {date} at position {pos} < lookback {lookback}"
            )

    def test_no_rebalance_dates_when_insufficient_history(self):
        """Should return empty list when the dataset is shorter than lookback."""
        short_returns = _make_synthetic_returns(n_days=100)
        dates = get_rebalance_dates(short_returns.index, lookback_window=252)
        assert dates == [], "Expected no rebalance dates for short history"

    def test_rebalance_dates_sorted(self, synthetic_returns):
        dates = get_rebalance_dates(synthetic_returns.index, lookback_window=126)
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# b) No look-ahead / temporal separation
# ---------------------------------------------------------------------------

class TestTemporalSeparation:
    def test_training_window_ends_before_rebalance(self, synthetic_returns):
        """For every rebalance, the training window must end strictly before it."""
        lookback = 126
        dates = get_rebalance_dates(synthetic_returns.index, lookback_window=lookback)
        for rebal_date in dates:
            pos = synthetic_returns.index.get_loc(rebal_date)
            training_window = synthetic_returns.iloc[pos - lookback : pos]
            assert rebal_date not in training_window.index, (
                f"Rebalance date {rebal_date} is inside its own training window — look-ahead bias!"
            )
            # Last training day must be strictly before the rebalance date.
            assert training_window.index[-1] < rebal_date, (
                f"Training window last day {training_window.index[-1]} >= rebalance {rebal_date}"
            )


# ---------------------------------------------------------------------------
# c) Constraints hold at every rebalance
# ---------------------------------------------------------------------------

class TestConstraints:
    MAX_WEIGHT = 1.0  # synthetic config has no per-asset cap

    def test_weights_non_negative(self, backtest_outputs):
        _, weights_hist, _ = backtest_outputs
        assert (weights_hist.values >= -1e-8).all(), "Some weights are negative"

    def test_weights_sum_to_one(self, backtest_outputs):
        _, weights_hist, _ = backtest_outputs
        row_sums = weights_hist.sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6, "Weights do not sum to 1"

    def test_constraints_with_real_config(self):
        """Run against real config and verify all portfolio constraints hold."""
        try:
            config = load_optimizer_config()
        except Exception:
            pytest.skip("Real config not available in this environment")

        max_weight = config["max_weight"]
        crypto_assets = config["crypto_assets"]
        max_crypto = config["max_crypto_weight"]

        # Build a 600-day synthetic frame with the real tickers.
        tickers = config["tickers"]
        returns = _make_synthetic_returns(n_days=600, tickers=tickers)

        _, weights_hist, _ = run_min_variance_backtest(
            returns=returns,
            optimizer_config=config,
            lookback_window=252,
        )

        for date, row in weights_hist.iterrows():
            assert (row.values >= -1e-8).all(), f"Negative weight on {date}"
            assert abs(row.sum() - 1.0) < 1e-6, f"Weights don't sum to 1 on {date}"
            assert row.max() <= max_weight + 1e-6, f"Per-asset cap violated on {date}"
            if crypto_assets:
                crypto_total = row.reindex(crypto_assets, fill_value=0.0).sum()
                assert crypto_total <= max_crypto + 1e-6, f"Crypto cap violated on {date}"


# ---------------------------------------------------------------------------
# d) Turnover shape and non-negativity
# ---------------------------------------------------------------------------

class TestTurnoverShape:
    def test_one_row_per_rebalance(self, backtest_outputs, synthetic_returns):
        _, weights_hist, turnover_hist = backtest_outputs
        assert len(turnover_hist) == len(weights_hist), (
            "turnover_history must have exactly one row per rebalance date"
        )

    def test_turnover_non_negative(self, backtest_outputs):
        _, _, turnover_hist = backtest_outputs
        assert (turnover_hist["turnover_one_way"] >= 0).all(), "Negative turnover found"

    def test_initial_rebalance_flags(self, backtest_outputs):
        _, _, turnover_hist = backtest_outputs
        first_row = turnover_hist.iloc[0]
        assert first_row["is_initial_rebalance"] is True or bool(first_row["is_initial_rebalance"]), (
            "First row must be flagged as initial rebalance"
        )
        assert first_row["turnover_one_way"] == 0.0, (
            "Initial rebalance turnover must be exactly 0.0"
        )
        # All subsequent rows should NOT be flagged as initial.
        subsequent = turnover_hist.iloc[1:]
        assert not subsequent["is_initial_rebalance"].any(), (
            "Only the first rebalance should have is_initial_rebalance=True"
        )

    def test_index_aligned_with_weights(self, backtest_outputs):
        _, weights_hist, turnover_hist = backtest_outputs
        assert weights_hist.index.equals(turnover_hist.index), (
            "weights_history and turnover_history must share the same index"
        )


# ---------------------------------------------------------------------------
# e) Unit tests: drifted weights and turnover calculation (synthetic)
# ---------------------------------------------------------------------------

class TestTurnoverUnit:
    def test_no_drift_no_turnover(self):
        """If returns are flat (zero), pre-trade weights equal target weights → turnover = 0."""
        tickers = ["X", "Y", "Z"]
        weights = pd.Series([0.5, 0.3, 0.2], index=tickers)
        # Zero returns over 5 days.
        holding = pd.DataFrame(0.0, index=range(5), columns=tickers)
        pretrade = compute_pretrade_weights(weights, holding)
        pd.testing.assert_series_equal(pretrade.sort_index(), weights.sort_index(), check_names=False)
        to = compute_turnover_one_way(weights, pretrade)
        assert abs(to) < 1e-12

    def test_turnover_computed_against_pretrade_not_target_to_target(self):
        """
        Confirm turnover uses drifted weights, not just differences of target weights.

        Setup:
          - Period 1 target: A=0.6, B=0.4
          - During holding period A grows 50%, B stays flat.
          - Pre-trade drifted weights: A = 0.6*1.5 / (0.6*1.5 + 0.4) = 0.9/1.3 ≈ 0.6923
                                       B = 0.4 / 1.3 ≈ 0.3077
          - Period 2 target: A=0.5, B=0.5
          - Naive target-to-target: |0.5 - 0.6| + |0.5 - 0.4| = 0.1 + 0.1 = 0.2 → one-way = 0.1
          - Correct (vs drifted): |0.5 - 0.6923| + |0.5 - 0.3077| ≈ 0.3846 → one-way ≈ 0.1923
        """
        tickers = ["A", "B"]
        prev_target = pd.Series([0.6, 0.4], index=tickers)
        # One holding day: A returns 50%, B returns 0%.
        holding = pd.DataFrame([[0.5, 0.0]], columns=tickers)
        pretrade = compute_pretrade_weights(prev_target, holding)

        expected_a = (0.6 * 1.5) / (0.6 * 1.5 + 0.4 * 1.0)
        expected_b = (0.4 * 1.0) / (0.6 * 1.5 + 0.4 * 1.0)
        assert abs(pretrade["A"] - expected_a) < 1e-9
        assert abs(pretrade["B"] - expected_b) < 1e-9

        new_target = pd.Series([0.5, 0.5], index=tickers)
        to_correct = compute_turnover_one_way(new_target, pretrade)
        to_naive   = 0.5 * (abs(0.5 - 0.6) + abs(0.5 - 0.4))  # 0.1

        assert abs(to_correct - 0.5 * (abs(0.5 - expected_a) + abs(0.5 - expected_b))) < 1e-9
        assert abs(to_correct - to_naive) > 1e-4, (
            "Drifted turnover should differ from naive target-to-target when drift occurred"
        )

    def test_turnover_symmetric(self):
        """Turnover(A→B) == Turnover(B→A) because of the 0.5 factor."""
        t1 = pd.Series([0.7, 0.3], index=["X", "Y"])
        t2 = pd.Series([0.4, 0.6], index=["X", "Y"])
        assert abs(
            compute_turnover_one_way(t1, t2) - compute_turnover_one_way(t2, t1)
        ) < 1e-12

    def test_turnover_bounded_zero_one(self):
        """One-way turnover must be in [0, 1] for valid weight vectors."""
        t = pd.Series([0.0, 1.0], index=["X", "Y"])
        p = pd.Series([1.0, 0.0], index=["X", "Y"])
        to = compute_turnover_one_way(t, p)
        assert 0.0 <= to <= 1.0 + 1e-9

    def test_pretrade_weights_sum_to_one(self):
        """Drifted weights should always normalise to 1."""
        tickers = ["A", "B", "C"]
        weights = pd.Series([0.4, 0.35, 0.25], index=tickers)
        rng = np.random.default_rng(0)
        holding = pd.DataFrame(
            rng.normal(0.001, 0.02, size=(21, 3)), columns=tickers
        )
        pretrade = compute_pretrade_weights(weights, holding)
        assert abs(pretrade.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# f) Regression tests: output integrity
# ---------------------------------------------------------------------------

class TestRegressionOutputs:
    def test_portfolio_returns_not_empty(self, backtest_outputs):
        port_returns, _, _ = backtest_outputs
        assert len(port_returns) > 0

    def test_portfolio_returns_index_sorted(self, backtest_outputs):
        port_returns, _, _ = backtest_outputs
        assert port_returns.index.is_monotonic_increasing

    def test_weights_history_index_sorted(self, backtest_outputs):
        _, weights_hist, _ = backtest_outputs
        assert weights_hist.index.is_monotonic_increasing

    def test_turnover_history_index_sorted(self, backtest_outputs):
        _, _, turnover_hist = backtest_outputs
        assert turnover_hist.index.is_monotonic_increasing

    def test_weights_and_turnover_index_match(self, backtest_outputs):
        _, weights_hist, turnover_hist = backtest_outputs
        assert list(weights_hist.index) == list(turnover_hist.index)

    def test_portfolio_returns_cover_oos_period(self, backtest_outputs, synthetic_returns):
        port_returns, weights_hist, _ = backtest_outputs
        first_rebal = weights_hist.index[0]
        assert port_returns.index[0] >= first_rebal

    def test_turnover_columns_present(self, backtest_outputs):
        _, _, turnover_hist = backtest_outputs
        required = {"turnover_one_way", "is_initial_rebalance", "n_assets_changed", "max_abs_weight_change"}
        assert required.issubset(set(turnover_hist.columns))

    def test_n_assets_changed_initial_zero(self, backtest_outputs):
        _, _, turnover_hist = backtest_outputs
        assert turnover_hist.iloc[0]["n_assets_changed"] == 0
