"""Script entry point for downloading raw market prices.

Usage:
    python scripts/run_download.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path so "src" imports work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    download_price_data,
    load_download_config,
    save_prices_csv,
    validate_prices,
)


def _format_missing_summary(prices: pd.DataFrame) -> str:
    """Format missing values summary for terminal output."""
    missing_counts = prices.isna().sum().sort_index()
    lines = [f"  - {ticker}: {int(count)}" for ticker, count in missing_counts.items()]
    return "\n".join(lines)


def main() -> None:
    """Run data download workflow for MVP asset universe."""
    try:
        config = load_download_config()

        prices = download_price_data(
            tickers=config["tickers"],
            start_date=config["start_date"],
            frequency=config["frequency"],
            preferred_price_field=config["preferred_price_field"],
            fallback_price_field=config["fallback_price_field"],
        )

        prices = validate_prices(prices=prices, required_tickers=config["tickers"])
        output_path = save_prices_csv(prices=prices, output_csv=config["output_csv"])

        print("Download complete.")
        print(f"Saved file: {output_path}")
        print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
        print(f"Shape: {prices.shape}")
        print("Missing values by asset (reported only, no imputation yet):")
        print(_format_missing_summary(prices))
    except Exception as error:
        print("Download failed.")
        print(f"Reason: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
