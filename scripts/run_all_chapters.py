"""Run the full Chapter 1-4 pipeline in a reproducible order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS_IN_ORDER = [
    "scripts/run_build_dataset.py",
    "scripts/run_benchmarks.py",
    "scripts/run_optimizer.py",
    "scripts/run_backtest.py",
    "scripts/run_robustness.py",
    "scripts/run_statistical_confidence.py",
    "scripts/run_tail_risk.py",
    "scripts/run_regime_analysis.py",
    "scripts/run_audit_fix_comparison.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full research pipeline")
    parser.add_argument(
        "--include-chapter5",
        action="store_true",
        help="Include Chapter 5 supervised risk overlay pipeline.",
    )
    args = parser.parse_args()

    python_exe = sys.executable
    scripts = list(SCRIPTS_IN_ORDER)
    if args.include_chapter5:
        scripts.append("scripts/run_chapter5.py")

    chapter_label = "Chapter 1-5" if args.include_chapter5 else "Chapter 1-4"
    print(f"Running full pipeline ({chapter_label})")
    print(f"Python executable: {python_exe}")
    print(f"Project root: {PROJECT_ROOT}")

    for script in scripts:
        print(f"\n=== Running {script} ===")
        cmd = [python_exe, str(PROJECT_ROOT / script)]
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Pipeline failed at {script} with exit code {completed.returncode}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
