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
    align_prices_to_calendar,
    build_dataset_metadata,
    clean_prices_mvp,
    compute_log_returns,
    compute_simple_returns,
    load_preprocessing_config,
    load_raw_prices,
    save_dataframe_to_csv,
    save_metadata_json,
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

        calendar_cfg = config["calendar"]
        calendar_policy = str(calendar_cfg["policy"]).lower().strip()
        annualization_factor = (
            float(calendar_cfg["annualization_factor"])
            if calendar_policy == "business_day_aligned"
            else float(calendar_cfg["calendar_day_annualization_factor"])
        )

        prices_aligned = align_prices_to_calendar(
            prices=prices_clean,
            policy=calendar_policy,
            tradfi_assets=[str(x) for x in calendar_cfg["tradfi_assets"]],
            crypto_assets=[str(x) for x in calendar_cfg["crypto_assets"]],
            require_tradfi_observation=bool(calendar_cfg["require_tradfi_observation"]),
        )

        returns_simple = compute_simple_returns(prices_aligned)
        returns_log = compute_log_returns(prices_aligned)

        metadata = build_dataset_metadata(
            prices=prices_aligned,
            calendar_policy=calendar_policy,
            annualization_factor=annualization_factor,
            tradfi_assets=[str(x) for x in calendar_cfg["tradfi_assets"]],
            crypto_assets=[str(x) for x in calendar_cfg["crypto_assets"]],
        )

        prices_clean_path = save_dataframe_to_csv(
            data=prices_aligned,
            output_csv=config["prices_clean_csv"],
        )
        prices_aligned_path = save_dataframe_to_csv(
            data=prices_aligned,
            output_csv=config["prices_aligned_csv"],
        )
        returns_simple_path = save_dataframe_to_csv(
            data=returns_simple,
            output_csv=config["returns_simple_csv"],
        )
        returns_log_path = save_dataframe_to_csv(
            data=returns_log,
            output_csv=config["returns_log_csv"],
        )
        metadata_path = save_metadata_json(
            metadata=metadata,
            output_json=config["dataset_metadata_json"],
        )

        print("Dataset build complete.")
        print(f"Saved cleaned prices: {prices_clean_path}")
        print(f"Saved aligned prices: {prices_aligned_path}")
        print(f"Saved simple returns: {returns_simple_path}")
        print(f"Saved log returns: {returns_log_path}")
        print(f"Saved dataset metadata: {metadata_path}")
        print(f"Calendar policy: {metadata['calendar_policy']}")
        print(f"Annualization factor: {metadata['annualization_factor']}")
        print(f"Weekend rows in aligned prices: {metadata['n_weekend_rows']}")
        print(f"Cleaned prices shape: {prices_aligned.shape}")
        print(f"Simple returns shape: {returns_simple.shape}")
        print(f"Log returns shape: {returns_log.shape}")
        print("Missing values in cleaned prices:")
        print(_format_missing_summary(prices_aligned))
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
