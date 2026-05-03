"""
Chapter 4 — Regime Analysis entry-point script.

Usage
-----
Run feature construction only (Phase 1):
    python scripts/run_regime_analysis.py --features-only

Run regime detection only (Phase 2, requires features CSV to exist):
    python scripts/run_regime_analysis.py --detect-only

Run both phases sequentially:
    python scripts/run_regime_analysis.py --features-only --detect-only

Run regime-conditional evaluation only (Phase 3):
    python scripts/run_regime_analysis.py --evaluate-only

The script exits with code 0 on success and non-zero on any error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is importable when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_regime_analysis  # noqa: E402
from regime_features import load_and_build, save_features  # noqa: E402
from regime_detection import load_and_detect, save_results  # noqa: E402
from regime_evaluation import load_and_evaluate, save_evaluation_results  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_features(cfg: dict[str, Any], project_root: Path) -> tuple[Any, Path]:
    """Execute Phase 1: build and persist the regime-feature panel."""
    logger.info("=== Chapter 4 — Phase 1: Regime Features ===")
    features = load_and_build(cfg, project_root)
    output_path = save_features(features, cfg, project_root)

    logger.info("Output summary:")
    logger.info("  Rows  : %d", len(features))
    logger.info("  Cols  : %d — %s", features.shape[1], list(features.columns))
    logger.info("  Start : %s", features.index.min().date())
    logger.info("  End   : %s", features.index.max().date())
    logger.info("  File  : %s", output_path)
    logger.info("Phase 1 complete.")
    return features, output_path


def run_detection(cfg: dict[str, Any], project_root: Path) -> tuple[Any, dict[str, Path]]:
    """Execute Phase 2: detect regimes and persist labels + summaries."""
    logger.info("=== Chapter 4 — Phase 2: Regime Detection ===")
    result = load_and_detect(cfg, project_root)
    paths = save_results(result, cfg, project_root)

    logger.info("Detection output summary:")
    logger.info("  Model       : %s", result.model_id)
    logger.info("  States      : %d", result.n_states)
    logger.info("  Date range  : %s → %s",
                result.labels.index.min().date(),
                result.labels.index.max().date())
    for sid, name in result.state_names.items():
        n = int((result.labels == sid).sum())
        pct = 100.0 * n / len(result.labels)
        logger.info("  %s : %d days (%.1f%%)", name, n, pct)
    logger.info("  Files:")
    for key, p in paths.items():
        logger.info("    %s → %s", key, p)

    logger.info("Transition matrix:\n%s", result.transition_matrix.to_string())
    logger.info("Phase 2 complete.")
    return result, paths


def run_evaluation(cfg: dict[str, Any], project_root: Path) -> tuple[Any, dict[str, Path]]:
    """Execute Phase 3: conditional regime evaluation and persist outputs."""
    logger.info("=== Chapter 4 — Phase 3: Regime Evaluation ===")
    result = load_and_evaluate(cfg, project_root)
    paths = save_evaluation_results(result, cfg, project_root)

    logger.info("Evaluation output summary:")
    logger.info("  Conditional (gross) rows : %d", len(result.conditional_performance))
    logger.info("  Conditional (net) rows   : %d", len(result.conditional_performance_net))
    logger.info("  Crypto exposure rows     : %d", len(result.crypto_exposure))
    logger.info("  Stress-window map rows   : %d", len(result.drawdown_tail_summary))
    logger.info("  Files:")
    for key, p in paths.items():
        logger.info("    %s → %s", key, p)
    logger.info("Phase 3 complete.")
    return result, paths


def _print_final_validations(
    *,
    features: Any | None,
    detection_result: Any | None,
    output_paths: dict[str, Path],
) -> None:
    logger.info("=== Chapter 4 — Final Validations ===")

    if features is not None and len(features) > 0:
        logger.info(
            "Features date range: %s → %s",
            features.index.min().date(),
            features.index.max().date(),
        )
        logger.info("Features observations: %d", len(features))

    if detection_result is not None:
        logger.info("Primary model used: %s", detection_result.model_id)
        logger.info("Regime temporal distribution:")
        total = len(detection_result.labels)
        for sid, name in detection_result.state_names.items():
            n = int((detection_result.labels == sid).sum())
            pct = 100.0 * n / total if total > 0 else 0.0
            logger.info("  %s: %d days (%.1f%%)", name, n, pct)

    if output_paths:
        logger.info("Output paths generated:")
        for key, path in output_paths.items():
            logger.info("  %s → %s", key, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 4 Regime Analysis pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Build regime features from returns_simple.csv and save to CSV. (Phase 1)",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Detect regimes from existing regime_features.csv and save labels. (Phase 2)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate conditional performance by regime and save Chapter 4 diagnostics. (Phase 3)",
    )
    args = parser.parse_args()

    run_full = not any(vars(args).values())

    cfg = load_regime_analysis()
    output_paths: dict[str, Path] = {}
    features = None
    detection_result = None

    if run_full or args.features_only:
        features, features_path = run_features(cfg, PROJECT_ROOT)
        output_paths["regime_features"] = features_path

    if run_full or args.detect_only:
        detection_result, detection_paths = run_detection(cfg, PROJECT_ROOT)
        output_paths.update(detection_paths)

    if run_full or args.evaluate_only:
        _, eval_paths = run_evaluation(cfg, PROJECT_ROOT)
        output_paths.update(eval_paths)

    _print_final_validations(
        features=features,
        detection_result=detection_result,
        output_paths=output_paths,
    )


if __name__ == "__main__":
    main()

