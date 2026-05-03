from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml  # noqa: E402
from src.supervised_targets import make_supervised_targets  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_baseline_returns(path: Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "min_variance" in df.columns:
        return df["min_variance"].astype(float)
    return df.iloc[:, 0].astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 5 supervised target generation")
    parser.add_argument("--config", default="supervised_risk_overlay.yaml", help="Config filename under config/")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})

    returns_path = PROJECT_ROOT / data_cfg.get("returns_path", "data/processed/returns_simple.csv")
    baseline_returns_path = PROJECT_ROOT / data_cfg.get(
        "baseline_returns_path",
        "data/processed/portfolio_returns.csv",
    )

    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    baseline_returns = _load_baseline_returns(baseline_returns_path).reindex(returns.index)

    targets = make_supervised_targets(returns=returns, baseline_portfolio_returns=baseline_returns, config=cfg)

    out_path = PROJECT_ROOT / "data/processed/supervised_targets.csv"
    ensure_directory(out_path.parent)
    targets.to_csv(out_path)

    print("Chapter 5 - supervised targets")
    print(f"  calendar_policy       : {cfg.get('calendar', {}).get('policy', 'business_day_aligned')}")
    print(f"  annualization_factor  : {cfg.get('calendar', {}).get('annualization_factor', 252)}")
    print(f"  targets_enabled       : {list(cfg.get('targets', {}).keys())}")
    print(f"  output_path           : {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
