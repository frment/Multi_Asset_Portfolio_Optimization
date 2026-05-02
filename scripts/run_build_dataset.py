"""Script entry point for building processed datasets from raw prices.

Usage:
    python scripts/run_build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path so "src" imports work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    clean_prices_mvp,
    compute_log_returns,
    compute_simple_returns,
    load_preprocessing_config,
    load_raw_prices,
    save_dataframe_to_csv,
)


def _format_missing_summary(data: pd.DataFrame) -> str:
    """Format missing values by column for readable terminal output."""
    missing_counts = data.isna().sum().sort_index()
    lines = [f"  - {column}: {int(count)}" for column, count in missing_counts.items()]
    return "\n".join(lines)


def main() -> None:
    """Build cleaned prices and return series datasets from raw prices."""
    try:
        config = load_preprocessing_config()

        raw_prices = load_raw_prices(config["raw_prices_csv"])
        prices_clean = clean_prices_mvp(raw_prices=raw_prices, ticker_order=config["tickers"])

        returns_simple = compute_simple_returns(prices_clean)
        returns_log = compute_log_returns(prices_clean)

        prices_clean_path = save_dataframe_to_csv(
            data=prices_clean,
            output_csv=config["prices_clean_csv"],
        )
        returns_simple_path = save_dataframe_to_csv(
            data=returns_simple,
            output_csv=config["returns_simple_csv"],
        )
        returns_log_path = save_dataframe_to_csv(
            data=returns_log,
            output_csv=config["returns_log_csv"],
        )

        print("Dataset build complete.")
        print(f"Saved cleaned prices: {prices_clean_path}")
        print(f"Saved simple returns: {returns_simple_path}")
        print(f"Saved log returns: {returns_log_path}")
        print(f"Cleaned prices shape: {prices_clean.shape}")
        print(f"Simple returns shape: {returns_simple.shape}")
        print(f"Log returns shape: {returns_log.shape}")
        print("Missing values in cleaned prices:")
        print(_format_missing_summary(prices_clean))
        print("Missing values in simple returns:")
        print(_format_missing_summary(returns_simple))
        print("Missing values in log returns:")
        print(_format_missing_summary(returns_log))
    except Exception as error:
        print("Dataset build failed.")
        print(f"Reason: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
