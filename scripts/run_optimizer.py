"""Script entry point for the static minimum variance optimiser.

Runs the first MVP optimisation on the full processed simple-return sample,
prints weights and key checks, and saves outputs under data/processed/.

Usage:
    python scripts/run_optimizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path so "src" imports work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.optimizer import load_optimizer_config, minimise_variance, validate_weights  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_returns_simple(returns_csv: Path) -> pd.DataFrame:
    """Load processed daily simple returns from CSV."""
    if not returns_csv.exists():
        raise FileNotFoundError(
            f"Simple returns file not found: {returns_csv}\n"
            "Run scripts/run_download.py and scripts/run_build_dataset.py first."
        )
    return pd.read_csv(returns_csv, index_col=0, parse_dates=True)


def main() -> None:
    """Run static minimum variance optimisation and save results."""
    try:
        settings = load_settings()
        processed_dir = PROJECT_ROOT / settings["paths"]["data_processed"]
        ensure_directory(processed_dir)

        returns_csv = processed_dir / "returns_simple.csv"
        output_weights_csv = processed_dir / "min_variance_weights_static.csv"
        output_returns_csv = processed_dir / "min_variance_returns_static.csv"

        returns = _load_returns_simple(returns_csv)
        config = load_optimizer_config()

        weights = minimise_variance(returns=returns, config=config)
        violations = validate_weights(weights, config=config)

        print("Minimum Variance optimisation complete (static full-sample).")
        print(f"Date range: {returns.index[0].date()} to {returns.index[-1].date()}")
        print(f"Assets used: {list(weights.index)}")

        print("\nOptimal weights:")
        for ticker, weight in weights.items():
            print(f"  - {ticker}: {weight:.4f}")

        total_crypto = weights[[t for t in config["crypto_assets"] if t in weights.index]].sum()
        print("\nConstraint checks:")
        print(f"  - Sum of weights: {weights.sum():.6f}")
        print(f"  - Max asset weight: {weights.max():.4f} (limit {config['max_weight']:.2f})")
        print(
            "  - Total crypto weight: "
            f"{total_crypto:.4f} (limit {config['max_crypto_weight']:.2f})"
        )

        if violations:
            print("\nValidation warnings:")
            for msg in violations:
                print(f"  - {msg}")

        # Save static weights as a one-row CSV.
        weights_df = weights.to_frame().T
        weights_df.index = ["static_full_sample"]
        weights_df.index.name = "rebalance_id"
        weights_df.to_csv(output_weights_csv)

        # Save the implied daily portfolio return series for reference.
        minvar_returns = (returns[weights.index] * weights.values).sum(axis=1)
        minvar_returns.name = "min_variance_static"
        minvar_returns.to_frame().to_csv(output_returns_csv)

        print(f"\nSaved weights to: {output_weights_csv.relative_to(PROJECT_ROOT)}")
        print(f"Saved returns to: {output_returns_csv.relative_to(PROJECT_ROOT)}")

    except Exception as error:
        print(f"\nOptimizer run failed. Reason: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
