"""Paired block bootstrap for time-series Sharpe ratio difference confidence intervals.

Design decisions (documented explicitly):
--------------------------------------------------------------------------
Block size:
    block_size=20 (approximately one month of trading days).
    This is a pragmatic choice for daily financial returns. Block sizes of
    15-25 are commonly used in the empirical finance literature (Politis &
    Romano 1994, Sullivan et al. 1999). A block of 20 captures most
    short-run autocorrelation and momentum in portfolio returns without
    extreme loss of effective sample size. Sensitivity to this choice is a
    known limitation; we do not tune it data-adaptively to avoid
    over-fitting the inference.

Bootstrap replications:
    bootstrap_n=5000 is the default. This gives stable 95% CI quantiles
    (Monte Carlo error on quantile estimate < 0.002 for typical setups)
    without excessive runtime.

Why these specific comparisons and not 20 more:
    Pre-registered set of four (and one optional) comparisons:
    1. baseline_ch1 vs minvar_no_crypto_control  — the anchor pair; tests
       whether crypto inclusion adds Sharpe on a common sample.
    2. covariance_sample vs ledoit_wolf (baseline_ch1 family) — tests
       whether shrinkage meaningfully changes the estimator.
    3. baseline_ch1 vs best lookback (lookback_504) — tests sensitivity
       to estimation window length within the same universe/freq.
    4. rebalance_monthly vs rebalance_quarterly — tests whether rebalance
       cadence matters within the same parameter set.
    5. baseline_ch1 gross vs baseline_ch1 net at 25 bps — checks whether
       costs materially compress Sharpe (both series exist from net CSV).
    Rationale for exclusion of other comparisons: crypto_cap_0_10/0_20/0_25
    are near-identical to baseline_ch1 by construction (same universe,
    trivially close Sharpe), and lookback_126/252 are already covered by
    comparison 3 with lookback_504. Adding those would increase the
    multiple-comparison problem without adding meaningful scientific content.

Sample alignment:
    We use the common sample (intersection of date indices) for all paired
    comparisons. This is necessary because strategies may have different
    start dates (e.g. lookback_504 starts in 2019-06, lookback_126 in
    2018-06). Aligning on the common sample ensures the bootstrap
    statistic is a valid paired difference.

Known limitations:
    - Multiple comparisons: even with 5 comparisons, the family-wise
      Type-I error is elevated. A Bonferroni correction would use alpha=0.01
      per test for a 0.95 family-wise level. We report raw 95% CIs and
      flag this explicitly in the notes column.
    - Sensitivity to block size: the CI width is sensitive to the chosen
      block size; results should be interpreted as directionally informative,
      not as exact frequentist coverage guarantees.
    - IC != truth: a CI that excludes zero is confirmatory evidence, not
      proof of superiority. The bootstrap distribution reflects historical
      sample variation, not a structural econometric model.
    - This is a light confirmatory layer. It is not a substitute for a
      full factor model, Bayesian updating, or regime-conditional analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import TRADING_DAYS_PER_YEAR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE: int = 20
DEFAULT_BOOTSTRAP_N: int = 5_000
DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComparisonSpec:
    """Declaration of one pre-registered paired comparison."""

    comparison_id: str
    comparison_family: str
    strategy_a: str
    strategy_b: str
    metric: str = "sharpe_diff"
    notes: str = ""


# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------


def _sharpe_from_returns(returns: np.ndarray) -> float:
    """Annualised Sharpe ratio (risk-free = 0) from a 1-D array of daily returns."""
    if len(returns) < 2:
        return float("nan")
    mu = np.mean(returns) * TRADING_DAYS_PER_YEAR
    sigma = np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    if sigma == 0.0:
        return float("nan")
    return mu / sigma


def _block_bootstrap_sharpe_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    *,
    block_size: int,
    n_replications: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return bootstrap distribution of (Sharpe_A - Sharpe_B).

    Args:
        returns_a: 1-D array of aligned daily returns for strategy A.
        returns_b: 1-D array of aligned daily returns for strategy B.
        block_size: Length of each non-overlapping block.
        n_replications: Number of bootstrap samples.
        rng: Seeded numpy Generator for reproducibility.

    Returns:
        1-D array of length n_replications with bootstrapped Sharpe differences.
    """
    n = len(returns_a)
    assert len(returns_b) == n, "Series must be pre-aligned."

    # Number of full blocks; any tail remainder is dropped.
    n_blocks = n // block_size
    n_used = n_blocks * block_size

    # Reshape into (n_blocks, block_size) for paired sampling.
    a_blocks = returns_a[:n_used].reshape(n_blocks, block_size)
    b_blocks = returns_b[:n_used].reshape(n_blocks, block_size)

    diffs = np.empty(n_replications, dtype=float)
    for i in range(n_replications):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        a_sample = a_blocks[idx].ravel()
        b_sample = b_blocks[idx].ravel()
        diffs[i] = _sharpe_from_returns(a_sample) - _sharpe_from_returns(b_sample)

    return diffs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Full result for one paired comparison."""

    comparison_id: str
    comparison_family: str
    strategy_a: str
    strategy_b: str
    sample_scope_used: str
    metric_compared: str
    point_estimate_difference: float
    ci_lower: float
    ci_upper: float
    ci_includes_zero: bool
    bootstrap_n: int
    block_size: int
    random_seed: int
    n_observations_aligned: int
    p_zero_crossing: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "comparison_family": self.comparison_family,
            "strategy_a": self.strategy_a,
            "strategy_b": self.strategy_b,
            "sample_scope_used": self.sample_scope_used,
            "metric_compared": self.metric_compared,
            "point_estimate_difference": self.point_estimate_difference,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_includes_zero": self.ci_includes_zero,
            "bootstrap_n": self.bootstrap_n,
            "block_size": self.block_size,
            "random_seed": self.random_seed,
            "n_observations_aligned": self.n_observations_aligned,
            "p_zero_crossing": self.p_zero_crossing,
            "notes": self.notes,
        }


def run_paired_bootstrap(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    comparison_id: str,
    comparison_family: str,
    strategy_a: str,
    strategy_b: str,
    metric: str = "sharpe_diff",
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_replications: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_SEED,
    ci_level: float = 0.95,
    notes: str = "",
) -> BootstrapResult:
    """Run a paired block bootstrap for Sharpe ratio difference.

    The two series are aligned to their common date intersection before
    computing the bootstrap. If the common sample has fewer than 2 *
    block_size observations the function raises ValueError.

    Args:
        series_a: Daily return series for strategy A (pd.Series, DatetimeIndex).
        series_b: Daily return series for strategy B (pd.Series, DatetimeIndex).
        comparison_id: Unique identifier for this comparison.
        comparison_family: Grouping label (e.g. 'covariance_method').
        strategy_a: Human-readable label for strategy A.
        strategy_b: Human-readable label for strategy B.
        metric: Currently only 'sharpe_diff' is supported.
        block_size: Non-overlapping block length in days.
        n_replications: Number of bootstrap replications.
        seed: Random seed for reproducibility.
        ci_level: Confidence level (default 0.95 → 95% CI).
        notes: Free-text note stored in the result.

    Returns:
        BootstrapResult with CI and metadata.
    """
    if metric != "sharpe_diff":
        raise NotImplementedError(f"Only 'sharpe_diff' is currently supported; got '{metric}'.")

    # Align on common sample.
    common_idx = series_a.index.intersection(series_b.index).sort_values()
    a_aligned = series_a.reindex(common_idx).to_numpy(dtype=float)
    b_aligned = series_b.reindex(common_idx).to_numpy(dtype=float)
    n_obs = len(common_idx)

    if n_obs < 2 * block_size:
        raise ValueError(
            f"[{comparison_id}] Common sample has only {n_obs} observations, "
            f"need at least 2×block_size={2 * block_size}."
        )

    rng = np.random.default_rng(seed)
    point_diff = _sharpe_from_returns(a_aligned) - _sharpe_from_returns(b_aligned)

    boot_diffs = _block_bootstrap_sharpe_diff(
        a_aligned,
        b_aligned,
        block_size=block_size,
        n_replications=n_replications,
        rng=rng,
    )

    alpha = 1.0 - ci_level
    ci_lower = float(np.quantile(boot_diffs, alpha / 2))
    ci_upper = float(np.quantile(boot_diffs, 1.0 - alpha / 2))
    ci_includes_zero = bool(ci_lower <= 0.0 <= ci_upper)

    # Fraction of bootstrap samples where difference crosses zero.
    p_zero = float(np.mean(boot_diffs <= 0.0)) if point_diff > 0 else float(np.mean(boot_diffs >= 0.0))

    return BootstrapResult(
        comparison_id=comparison_id,
        comparison_family=comparison_family,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        sample_scope_used="common_aligned",
        metric_compared=metric,
        point_estimate_difference=float(point_diff),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_includes_zero=ci_includes_zero,
        bootstrap_n=n_replications,
        block_size=block_size,
        random_seed=seed,
        n_observations_aligned=n_obs,
        p_zero_crossing=p_zero,
        notes=notes,
    )


def build_confidence_summary(results: list[BootstrapResult]) -> pd.DataFrame:
    """Convert a list of BootstrapResult objects to the canonical output DataFrame."""
    rows = [r.to_dict() for r in results]
    return pd.DataFrame(rows)
