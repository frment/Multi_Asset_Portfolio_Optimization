"""Run the full Chapter 1-4 pipeline in a reproducible order."""

from __future__ import annotations

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
    python_exe = sys.executable

    print("Running full pipeline (Chapter 1-4)")
    print(f"Python executable: {python_exe}")
    print(f"Project root: {PROJECT_ROOT}")

    for script in SCRIPTS_IN_ORDER:
        print(f"\n=== Running {script} ===")
        cmd = [python_exe, str(PROJECT_ROOT / script)]
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Pipeline failed at {script} with exit code {completed.returncode}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
