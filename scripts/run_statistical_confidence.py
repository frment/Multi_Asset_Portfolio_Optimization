"""Runner for Chapter 2 statistical confidence layer.

Computes 95% paired block bootstrap confidence intervals for five
pre-registered Sharpe ratio difference comparisons.

Pre-registered comparisons (documented in src/bootstrap.py):
    C1  baseline_ch1 vs minvar_no_crypto_control        (anchor pair)
    C2  covariance_sample vs covariance_ledoit_wolf      (cov method family)
    C3  baseline_ch1 vs lookback_504                     (lookback family)
    C4  rebalance_monthly vs rebalance_quarterly          (rebalance family)
    C5  baseline_ch1 gross vs baseline_ch1 net 25 bps    (cost sensitivity)

Outputs:
    data/processed/robustness/confidence_summary.csv

Usage:
    python scripts/run_statistical_confidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bootstrap import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_SEED,
    BootstrapResult,
    build_confidence_summary,
    run_paired_bootstrap,
)
from src.utils import ensure_directory  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBUSTNESS_DIR = PROJECT_ROOT / "data" / "processed" / "robustness"
OUTPUT_PATH = ROBUSTNESS_DIR / "confidence_summary.csv"

# Cost scenario to use for C5.
COST_BPS_FOR_NET = 25.0

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_returns_panel() -> pd.DataFrame:
    """Load robustness_returns.csv as a panel DataFrame."""
    path = ROBUSTNESS_DIR / "robustness_returns.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Returns panel not found: {path}\n"
            "Run scripts/run_robustness.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


def _load_net_summary() -> pd.DataFrame:
    """Load robustness_summary_net.csv."""
    path = ROBUSTNESS_DIR / "robustness_summary_net.csv"
    if not path.exists():
        raise FileNotFoundError(f"Net summary not found: {path}")
    return pd.read_csv(path)


def _get_series(panel: pd.DataFrame, experiment_id: str) -> pd.Series:
    """Extract daily return series for a single experiment from the native panel."""
    subset = panel[panel["experiment_id"] == experiment_id]["portfolio_return"]
    if subset.empty:
        raise KeyError(f"experiment_id '{experiment_id}' not found in returns panel.")
    return subset.sort_index()


# ---------------------------------------------------------------------------
# Net-of-cost series reconstruction for C5
# ---------------------------------------------------------------------------


def _build_net_series_from_turnover(
    gross_series: pd.Series,
    experiment_id: str,
    cost_bps: float,
) -> Optional[pd.Series]:
    """Reconstruct net-of-cost daily returns from turnover panel.

    We look up per-rebalance turnover, apply cost = turnover * cost_rate
    on each rebalance date, and subtract from gross returns.

    Returns None if the required turnover data is not available.
    """
    turnover_path = ROBUSTNESS_DIR / "robustness_turnover_panel.csv"
    if not turnover_path.exists():
        return None

    turnover_df = pd.read_csv(turnover_path, parse_dates=["rebalance_date"])
    spec_turnover = turnover_df[
        (turnover_df["experiment_id"] == experiment_id)
        & (~turnover_df["is_initial_rebalance"])
    ].set_index("rebalance_date")["turnover_one_way"]

    if spec_turnover.empty:
        return None

    cost_rate = cost_bps / 10_000.0
    rebalance_costs = spec_turnover * cost_rate

    net = gross_series.copy().astype(float)
    common_idx = net.index.intersection(rebalance_costs.index)
    if len(common_idx) > 0:
        net.loc[common_idx] -= rebalance_costs.reindex(common_idx).values

    return net


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all pre-registered comparisons and write confidence_summary.csv."""
    print("Loading returns panel...")
    panel = _load_returns_panel()

    results: list[BootstrapResult] = []
    skipped: list[str] = []

    # ------------------------------------------------------------------ C1
    print("C1: baseline_ch1 vs minvar_no_crypto_control")
    try:
        r = run_paired_bootstrap(
            _get_series(panel, "baseline_ch1"),
            _get_series(panel, "minvar_no_crypto_control"),
            comparison_id="C1_anchor_pair",
            comparison_family="anchors",
            strategy_a="baseline_ch1",
            strategy_b="minvar_no_crypto_control",
            block_size=DEFAULT_BLOCK_SIZE,
            n_replications=DEFAULT_BOOTSTRAP_N,
            seed=DEFAULT_SEED,
            notes=(
                "Primary anchor comparison. Tests whether crypto inclusion adds Sharpe. "
                "Common sample used (identical start dates). "
                "Multiple comparisons caveat applies (5 comparisons total)."
            ),
        )
        results.append(r)
        print(f"  Δ Sharpe = {r.point_estimate_difference:+.4f}  "
              f"95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]  "
              f"CI includes zero: {r.ci_includes_zero}")
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        skipped.append(f"C1: {exc}")

    # ------------------------------------------------------------------ C2
    print("C2: covariance_sample (baseline_ch1) vs covariance_ledoit_wolf (baseline_ch1)")
    try:
        r = run_paired_bootstrap(
            _get_series(panel, "covariance_sample_baseline_ch1"),
            _get_series(panel, "covariance_ledoit_wolf_baseline_ch1"),
            comparison_id="C2_covariance_method",
            comparison_family="covariance_method",
            strategy_a="covariance_sample_baseline_ch1",
            strategy_b="covariance_ledoit_wolf_baseline_ch1",
            block_size=DEFAULT_BLOCK_SIZE,
            n_replications=DEFAULT_BOOTSTRAP_N,
            seed=DEFAULT_SEED,
            notes=(
                "Tests whether Ledoit-Wolf shrinkage changes Sharpe vs sample covariance "
                "in the baseline_ch1 universe. Both series share identical dates. "
                "covariance_sample_baseline_ch1 is an alias for baseline_ch1."
            ),
        )
        results.append(r)
        print(f"  Δ Sharpe = {r.point_estimate_difference:+.4f}  "
              f"95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]  "
              f"CI includes zero: {r.ci_includes_zero}")
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        skipped.append(f"C2: {exc}")

    # ------------------------------------------------------------------ C3
    print("C3: baseline_ch1 vs lookback_504")
    try:
        r = run_paired_bootstrap(
            _get_series(panel, "baseline_ch1"),
            _get_series(panel, "lookback_504"),
            comparison_id="C3_lookback_504_vs_baseline",
            comparison_family="lookback",
            strategy_a="baseline_ch1",
            strategy_b="lookback_504",
            block_size=DEFAULT_BLOCK_SIZE,
            n_replications=DEFAULT_BOOTSTRAP_N,
            seed=DEFAULT_SEED,
            notes=(
                "lookback_504 starts 2019-06 vs baseline_ch1 2018-10. "
                "Common sample alignment used (2019-06 onwards). "
                "Longer window reduces effective sample relative to native comparison. "
                "lookback_126 not included: differences in summary are negligible "
                "and lookback_504 provides a more informative stress case."
            ),
        )
        results.append(r)
        print(f"  Δ Sharpe = {r.point_estimate_difference:+.4f}  "
              f"95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]  "
              f"CI includes zero: {r.ci_includes_zero}")
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        skipped.append(f"C3: {exc}")

    # ------------------------------------------------------------------ C4
    print("C4: rebalance_monthly vs rebalance_quarterly")
    try:
        r = run_paired_bootstrap(
            _get_series(panel, "rebalance_monthly"),
            _get_series(panel, "rebalance_quarterly"),
            comparison_id="C4_rebalance_frequency",
            comparison_family="rebalance_frequency",
            strategy_a="rebalance_monthly",
            strategy_b="rebalance_quarterly",
            block_size=DEFAULT_BLOCK_SIZE,
            n_replications=DEFAULT_BOOTSTRAP_N,
            seed=DEFAULT_SEED,
            notes=(
                "Tests whether monthly vs quarterly rebalancing materially differs in Sharpe. "
                "Both series share identical dates. "
                "Weekly rebalance excluded from this layer (out of scope for first-pass)."
            ),
        )
        results.append(r)
        print(f"  Δ Sharpe = {r.point_estimate_difference:+.4f}  "
              f"95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]  "
              f"CI includes zero: {r.ci_includes_zero}")
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        skipped.append(f"C4: {exc}")

    # ------------------------------------------------------------------ C5
    print(f"C5: baseline_ch1 gross vs net at {COST_BPS_FOR_NET:.0f} bps")
    try:
        gross_series = _get_series(panel, "baseline_ch1")
        net_series = _build_net_series_from_turnover(gross_series, "baseline_ch1", COST_BPS_FOR_NET)
        if net_series is None:
            raise RuntimeError(
                "Could not reconstruct net series from turnover panel. "
                "Turnover data missing or empty for baseline_ch1."
            )
        r = run_paired_bootstrap(
            gross_series,
            net_series,
            comparison_id="C5_gross_vs_net_25bps",
            comparison_family="cost_sensitivity",
            strategy_a="baseline_ch1_gross",
            strategy_b=f"baseline_ch1_net_{int(COST_BPS_FOR_NET)}bps",
            block_size=DEFAULT_BLOCK_SIZE,
            n_replications=DEFAULT_BOOTSTRAP_N,
            seed=DEFAULT_SEED,
            notes=(
                f"baseline_ch1 gross vs net at {COST_BPS_FOR_NET:.0f} bps one-way. "
                "Net series reconstructed from turnover panel (initial rebalance excluded). "
                "Both series share identical dates by construction. "
                "The point estimate is always non-negative; CI tests "
                "whether the cost drag is statistically distinguishable from zero."
            ),
        )
        results.append(r)
        print(f"  Δ Sharpe = {r.point_estimate_difference:+.4f}  "
              f"95% CI [{r.ci_lower:.4f}, {r.ci_upper:.4f}]  "
              f"CI includes zero: {r.ci_includes_zero}")
    except Exception as exc:
        print(f"  SKIPPED — {exc}")
        skipped.append(f"C5: {exc}")

    # ------------------------------------------------------------------ Save
    if not results:
        print("\nNo results produced — nothing written.")
        return

    ensure_directory(ROBUSTNESS_DIR)
    summary = build_confidence_summary(results)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(results)} comparison(s) → {OUTPUT_PATH}")

    if skipped:
        print("\nSkipped comparisons:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    main()
