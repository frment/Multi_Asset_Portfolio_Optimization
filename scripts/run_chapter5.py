from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chapter5_runtime import parse_debug_args  # noqa: E402

SCRIPTS = [
    "scripts/run_supervised_targets.py",
    "scripts/run_supervised_features.py",
    "scripts/run_supervised_models.py",
    "scripts/run_risk_overlay.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 5 end-to-end pipeline")
    parser.add_argument("--config", default="supervised_risk_overlay.yaml", help="Config filename under config/")
    parse_debug_args(parser)
    args = parser.parse_args()

    py = sys.executable
    print("Running Chapter 5 pipeline")
    print(f"Python executable: {py}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Debug mode: {args.debug}")

    for script in SCRIPTS:
        if args.skip_models and script.endswith("run_supervised_models.py"):
            print(f"[SKIP] {script} (--skip-models)")
            continue
        if args.skip_overlay and script.endswith("run_risk_overlay.py"):
            print(f"[SKIP] {script} (--skip-overlay)")
            continue

        print(f"\n=== Running {script} ===")
        cmd = [py, str(PROJECT_ROOT / script), "--config", args.config]

        # Propagate debug / limit flags to sub-scripts that accept them.
        if script.endswith(("run_supervised_models.py", "run_risk_overlay.py")):
            if args.debug:
                cmd.append("--debug")
            if args.force:
                cmd.append("--force")
        if script.endswith("run_supervised_models.py"):
            if args.max_targets is not None:
                cmd.extend(["--max-targets", str(args.max_targets)])
            if args.max_models is not None:
                cmd.extend(["--max-models", str(args.max_models)])
            if args.max_splits is not None:
                cmd.extend(["--max-splits", str(args.max_splits)])

        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Chapter 5 pipeline failed at {script} with exit code {completed.returncode}")

    print("\nChapter 5 pipeline completed successfully.")


if __name__ == "__main__":
    main()
