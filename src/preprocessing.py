"""Preprocessing utilities for building clean price and return datasets.

This module implements a transparent MVP cleaning policy:
1. sort dates,
2. drop duplicate dates,
3. drop rows entirely NaN,
4. forward-fill missing prices,
5. drop remaining rows with NaNs.

The policy is intentionally conservative and may be refined later.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_assets, load_settings
from src.utils import ensure_directory


def load_preprocessing_config() -> dict[str, Any]:
    """Load preprocessing-related paths and ticker order from config files.

    Returns:
        Dictionary containing:
        - tickers: ordered ticker list from assets config
        - raw_prices_csv: input CSV path
        - prices_clean_csv: output path for cleaned prices
        - returns_simple_csv: output path for simple returns
        - returns_log_csv: output path for log returns
    """
    assets_cfg = load_assets()
    settings_cfg = load_settings()

    tickers = assets_cfg.get("all_tickers", [])
    paths_cfg = settings_cfg.get("paths", {})

    data_raw_dir = Path(paths_cfg.get("data_raw", "data/raw"))
    data_processed_dir = Path(paths_cfg.get("data_processed", "data/processed"))

    if not tickers:
        raise ValueError("assets.yaml must define a non-empty 'all_tickers' list.")

    return {
        "tickers": [str(ticker) for ticker in tickers],
        "raw_prices_csv": data_raw_dir / "prices_raw.csv",
        "prices_clean_csv": data_processed_dir / "prices_clean.csv",
        "returns_simple_csv": data_processed_dir / "returns_simple.csv",
        "returns_log_csv": data_processed_dir / "returns_log.csv",
    }


def load_raw_prices(raw_prices_csv: str | Path) -> pd.DataFrame:
    """Load raw prices from CSV.

    Args:
        raw_prices_csv: Path to raw prices CSV.

    Returns:
        DataFrame with date index and ticker columns.
    """
    csv_path = Path(raw_prices_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw prices file not found: {csv_path}")

    return pd.read_csv(csv_path, index_col=0, parse_dates=True)


def clean_prices_mvp(raw_prices: pd.DataFrame, ticker_order: list[str]) -> pd.DataFrame:
    """Clean raw price data using the MVP policy.

    Policy steps:
    1) ensure datetime index,
    2) sort dates,
    3) drop duplicate dates,
    4) drop rows that are fully NaN,
    5) forward-fill missing values,
    6) drop any remaining NaNs.

    Args:
        raw_prices: Raw price DataFrame loaded from CSV.
        ticker_order: Desired ticker column order from config.

    Returns:
        Cleaned price DataFrame.
    """
    if raw_prices.empty:
        raise ValueError("Raw prices DataFrame is empty.")

    prices = raw_prices.copy()

    # Ensure datetime index so time operations behave correctly.
    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Raw price index must be a DatetimeIndex or convertible to datetimes."
            ) from error

    missing_tickers = [ticker for ticker in ticker_order if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError(
            "Raw prices are missing required ticker columns: " + ", ".join(missing_tickers)
        )

    # Keep only required tickers and preserve config-defined order.
    prices = prices.loc[:, ticker_order]

    # Standardise index name so output CSVs have a consistent 'Date' column header.
    prices.index.name = "Date"

    # MVP cleaning policy (conservative and easy to audit).
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    prices = prices.dropna(how="any")

    if prices.empty:
        raise ValueError("Cleaned prices are empty after applying MVP cleaning policy.")

    return prices


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily simple returns using percentage change.

    Simple returns are used for benchmark construction, backtesting, and
    performance metrics.  Log returns are saved separately for analysis
    and future extensions (e.g. normality checks, GARCH modelling).

    Args:
        prices: Cleaned price DataFrame.

    Returns:
        DataFrame of daily simple returns.
    """
    returns = prices.pct_change()
    returns = returns.dropna(how="any")
    return returns


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns using log(price / lagged price).

    Args:
        prices: Cleaned price DataFrame.

    Returns:
        DataFrame of daily log returns.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna(how="any")
    return returns


def save_dataframe_to_csv(data: pd.DataFrame, output_csv: str | Path) -> Path:
    """Save a DataFrame to CSV, creating parent directories if needed.

    Args:
        data: DataFrame to save.
        output_csv: Destination CSV path.

    Returns:
        Absolute path to saved CSV.
    """
    output_path = Path(output_csv)
    ensure_directory(output_path.parent)
    data.to_csv(output_path)
    return output_path.resolve()
