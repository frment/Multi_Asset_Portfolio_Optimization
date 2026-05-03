"""Script entry point for Chapter 2 first-pass robustness experiments.

Runs one-factor-at-a-time robustness experiments for minimum variance:
- lookback sensitivity: 126, 252, 504
- crypto cap sensitivity: 0.00, 0.10, 0.20, 0.25
- rebalance frequency sensitivity: monthly, quarterly
- transaction-cost scenarios (post-processing): 0, 10, 25, 50 bps

Notes:
- Two anchors are frozen: baseline_ch1 and minvar_no_crypto_control.
- Weekly rebalance support exists in the engine but is excluded from this
  first pass by default.
- Benchmarks are intentionally deferred. This runner focuses on min_variance
  robustness and stores enough metadata for future aligned benchmark analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_calendar_settings,
    load_dataset_metadata,
    load_robustness,
    load_settings,
    resolve_annualization_factor,
)
from src.optimizer import load_optimizer_config  # noqa: E402
from src.robustness import run_first_pass_robustness  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_returns(returns_csv: Path) -> pd.DataFrame:
    """Load processed simple returns from CSV."""
    if not returns_csv.exists():
        raise FileNotFoundError(
            f"Returns file not found: {returns_csv}\n"
            "Run scripts/run_download.py and scripts/run_build_dataset.py first."
        )
    return pd.read_csv(returns_csv, index_col=0, parse_dates=True)


def main() -> None:
    """Run first-pass robustness experiments and save outputs."""
    try:
        settings = load_settings()
        processed_dir = PROJECT_ROOT / settings["paths"]["data_processed"]
        robustness_dir = processed_dir / "robustness"
        ensure_directory(robustness_dir)

        returns_csv = processed_dir / "returns_simple.csv"
        returns = _load_returns(returns_csv)

        optimizer_config = load_optimizer_config()
        backtest_cfg = settings.get("backtest", {})
        robustness_cfg = load_robustness()
        dataset_metadata = load_dataset_metadata()
        calendar_cfg = get_calendar_settings(settings)
        annualization_factor = resolve_annualization_factor(settings=settings, dataset_metadata=dataset_metadata)
        holding_return_method = str(backtest_cfg.get("holding_return_method", "drifted_buy_and_hold"))
        allow_weekend_rebalances = bool(calendar_cfg.get("allow_weekend_rebalances", False))

        default_lookback = int(backtest_cfg.get("lookback_window_days", 252))
        default_rebalance_frequency = str(backtest_cfg.get("rebalance_frequency", "monthly"))
        risk_free_rate = float(backtest_cfg.get("risk_free_rate", 0.0))

        anchors_cfg = robustness_cfg.get("anchors", {})
        baseline_cfg = anchors_cfg.get("baseline_ch1", {})

        lookback = int(baseline_cfg.get("lookback_window_days", default_lookback))
        rebalance_frequency = str(
            baseline_cfg.get("rebalance_frequency", default_rebalance_frequency)
        )
        optimizer_config["max_crypto_weight"] = float(
            baseline_cfg.get(
                "max_total_crypto_weight",
                optimizer_config.get("max_crypto_weight", 0.20),
            )
        )

        first_pass_cfg = robustness_cfg.get("first_pass", {})
        include_weekly = bool(first_pass_cfg.get("include_weekly_in_first_pass", False))
        lookback_values = [int(x) for x in first_pass_cfg.get("lookback_values", [126, 252, 504])]
        crypto_cap_values = [
            float(x) for x in first_pass_cfg.get("crypto_cap_values", [0.00, 0.10, 0.20, 0.25])
        ]
        rebalance_values = [
            str(x) for x in first_pass_cfg.get("rebalance_values", ["monthly", "quarterly"])
        ]
        covariance_methods = [
            str(x) for x in first_pass_cfg.get("covariance_methods", ["sample", "ledoit_wolf"])
        ]
        include_no_crypto_anchor_in_covariance_family = bool(
            first_pass_cfg.get("include_no_crypto_anchor_in_covariance_family", True)
        )

        cost_scenarios_bps = [
            float(x) for x in first_pass_cfg.get("cost_scenarios_bps", [0.0, 10.0, 25.0, 50.0])
        ]

        print("Running Chapter 2 first-pass robustness (min_variance only)...")
        print(f"  calendar_policy       : {dataset_metadata.get('calendar_policy', calendar_cfg.get('policy'))}")
        print(f"  annualization_factor  : {annualization_factor}")
        print(f"  holding_return_method : {holding_return_method}")
        print(f"  rebalance_frequency   : {rebalance_frequency}")
        outputs = run_first_pass_robustness(
            returns=returns,
            base_optimizer_config=optimizer_config,
            risk_free_rate=risk_free_rate,
            base_lookback_window_days=lookback,
            base_rebalance_frequency=rebalance_frequency,
            include_weekly_in_first_pass=include_weekly,
            lookback_values=lookback_values,
            crypto_cap_values=crypto_cap_values,
            rebalance_values=rebalance_values,
            covariance_methods=covariance_methods,
            include_no_crypto_anchor_in_covariance_family=include_no_crypto_anchor_in_covariance_family,
            cost_scenarios_bps=cost_scenarios_bps,
            annualization_factor=annualization_factor,
            holding_return_method=holding_return_method,
            allow_weekend_rebalances=allow_weekend_rebalances,
        )

        summary_csv = robustness_dir / "robustness_summary.csv"
        summary_gross_csv = robustness_dir / "robustness_summary_gross.csv"
        summary_net_csv = robustness_dir / "robustness_summary_net.csv"
        returns_out_csv = robustness_dir / "robustness_returns.csv"
        metadata_csv = robustness_dir / "robustness_metadata.csv"
        weights_csv = robustness_dir / "robustness_weights_panel.csv"
        turnover_csv = robustness_dir / "robustness_turnover_panel.csv"
        common_family_csv = robustness_dir / "robustness_summary_common_family.csv"
        common_family_net_csv = robustness_dir / "robustness_summary_common_family_net.csv"

        outputs["robustness_summary"].to_csv(summary_csv, index=False)
        outputs["robustness_summary_gross"].to_csv(summary_gross_csv, index=False)
        outputs["robustness_summary_net"].to_csv(summary_net_csv, index=False)
        outputs["robustness_returns"].to_csv(returns_out_csv, index=False)
        outputs["robustness_metadata"].to_csv(metadata_csv, index=False)
        outputs["robustness_weights_panel"].to_csv(weights_csv, index=False)
        outputs["robustness_turnover_panel"].to_csv(turnover_csv, index=False)
        outputs["robustness_summary_common_family"].to_csv(common_family_csv, index=False)
        outputs["robustness_summary_common_family_net"].to_csv(common_family_net_csv, index=False)

        print("Robustness run complete.")
        print(f"  Experiments : {outputs['robustness_summary']['experiment_id'].nunique()}")
        print(f"  Rows summary: {len(outputs['robustness_summary'])}")
        print(f"  Saved summary               : {summary_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved summary gross         : {summary_gross_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved summary net           : {summary_net_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved returns panel         : {returns_out_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved metadata              : {metadata_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved weights panel         : {weights_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved turnover panel        : {turnover_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved common-family summary : {common_family_csv.relative_to(PROJECT_ROOT)}")
        print(f"  Saved common-family net     : {common_family_net_csv.relative_to(PROJECT_ROOT)}")

    except Exception as error:
        print(f"\nRobustness run failed. Reason: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
