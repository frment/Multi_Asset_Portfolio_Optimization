"""Data download utilities for market prices.

This module provides simple, reusable functions to:
- read download-related settings from project config files,
- download price data from yfinance,
- validate the resulting price matrix,
- save the output to disk.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.config import load_assets, load_settings
from src.utils import ensure_directory


def load_download_config() -> dict[str, Any]:
    """Load and combine download settings from project config files.

    Returns:
        Dictionary with tickers, start_date, frequency, preferred_price_field,
        fallback_price_field, and output_csv path.

    Raises:
        ValueError: If required config values are missing.
    """
    assets_cfg = load_assets()
    settings_cfg = load_settings()

    tickers = assets_cfg.get("all_tickers", [])
    start_date = assets_cfg.get("start_date")

    data_cfg = settings_cfg.get("data", {})
    # Use settings.yaml as the single source of truth for download frequency.
    frequency = data_cfg.get("frequency", "daily")
    preferred_price_field = data_cfg.get("price_field", "Adj Close")

    paths_cfg = settings_cfg.get("paths", {})
    data_raw_dir = paths_cfg.get("data_raw", "data/raw")

    if not tickers:
        raise ValueError("assets.yaml must define a non-empty 'all_tickers' list.")
    if not start_date:
        raise ValueError("assets.yaml must define 'start_date'.")

    return {
        "tickers": [str(ticker) for ticker in tickers],
        "start_date": str(start_date),
        "frequency": str(frequency),
        "preferred_price_field": str(preferred_price_field),
        "fallback_price_field": "Close",
        "output_csv": Path(data_raw_dir) / "prices_raw.csv",
    }


def _to_yfinance_interval(frequency: str) -> str:
    """Map config frequency to yfinance interval.

    Args:
        frequency: Frequency string from config (e.g., "daily").

    Returns:
        yfinance interval string.

    Raises:
        ValueError: If the frequency is unsupported.
    """
    mapping = {"daily": "1d"}
    interval = mapping.get(frequency.lower())
    if interval is None:
        raise ValueError(
            f"Unsupported frequency '{frequency}'. "
            "MVP downloader currently supports only 'daily'."
        )
    return interval


def _extract_price_matrix(
    raw_data: pd.DataFrame,
    tickers: list[str],
    preferred_field: str,
    fallback_field: str,
) -> pd.DataFrame:
    """Extract one price field from yfinance output in wide format.

    Args:
        raw_data: Raw DataFrame returned by yfinance download.
        tickers: Required tickers in expected output.
        preferred_field: Preferred price field (usually "Adj Close").
        fallback_field: Fallback field when preferred is unavailable.

    Returns:
        Wide DataFrame with date index and one column per ticker.

    Raises:
        ValueError: If raw data is empty or no usable price field exists.
    """
    if raw_data.empty:
        raise ValueError("Downloaded data is empty. Check ticker list or start date.")

    if isinstance(raw_data.columns, pd.MultiIndex):
        available_fields = set(raw_data.columns.get_level_values(0))

        if preferred_field in available_fields:
            prices = raw_data[preferred_field].copy()
        elif fallback_field in available_fields:
            prices = raw_data[fallback_field].copy()
        else:
            raise ValueError(
                "No usable price column found in download output. "
                f"Expected '{preferred_field}' or '{fallback_field}'."
            )
    else:
        # Single-level columns happen for single-ticker downloads. We still support it.
        if preferred_field in raw_data.columns:
            prices = raw_data[[preferred_field]].copy()
        elif fallback_field in raw_data.columns:
            prices = raw_data[[fallback_field]].copy()
        else:
            raise ValueError(
                "No usable price column found in download output. "
                f"Expected '{preferred_field}' or '{fallback_field}'."
            )

        # If only one ticker is requested, rename the single column to that ticker.
        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    if not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame()

    # Keep ticker order as defined in config for deterministic output.
    existing_columns = [str(column) for column in prices.columns]
    prices.columns = existing_columns

    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError(
            "Missing ticker columns after price extraction: "
            + ", ".join(missing_tickers)
        )

    prices = prices.loc[:, tickers]
    return prices


def download_price_data(
    tickers: list[str],
    start_date: str,
    frequency: str,
    preferred_price_field: str = "Adj Close",
    fallback_price_field: str = "Close",
) -> pd.DataFrame:
    """Download historical prices from yfinance and return a wide DataFrame.

    Args:
        tickers: List of ticker symbols.
        start_date: Download start date (YYYY-MM-DD).
        frequency: Frequency label from config. MVP supports "daily".
        preferred_price_field: Preferred price field.
        fallback_price_field: Fallback price field.

    Returns:
        Wide DataFrame indexed by date with one column per ticker.

    Raises:
        ValueError: If data is empty, frequency unsupported, or price field missing.
    """
    interval = _to_yfinance_interval(frequency)

    raw_data = yf.download(
        tickers=tickers,
        start=start_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    return _extract_price_matrix(
        raw_data=raw_data,
        tickers=tickers,
        preferred_field=preferred_price_field,
        fallback_field=fallback_price_field,
    )


def validate_prices(prices: pd.DataFrame, required_tickers: list[str]) -> pd.DataFrame:
    """Validate and standardize downloaded prices.

    Validation includes:
    - non-empty frame,
    - required tickers present,
    - DatetimeIndex validation,
    - sorted date index,
    - duplicate dates removed.

    Missing values are intentionally not imputed at this stage.
    They are only reported by the download script.

    Args:
        prices: Price DataFrame to validate.
        required_tickers: Required ticker columns.

    Returns:
        Validated and standardized DataFrame.

    Raises:
        ValueError: If frame is empty, required tickers are missing,
            or index cannot be interpreted as datetimes.
    """
    if prices.empty:
        raise ValueError("Price DataFrame is empty after extraction.")

    missing_tickers = [ticker for ticker in required_tickers if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError("Missing required tickers: " + ", ".join(missing_tickers))

    prices = prices.copy()

    if not isinstance(prices.index, pd.DatetimeIndex):
        try:
            prices.index = pd.to_datetime(prices.index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Price index must be a DatetimeIndex or convertible to datetimes."
            ) from error

    prices = prices.loc[:, required_tickers]
    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.sort_index()

    return prices


def save_prices_csv(prices: pd.DataFrame, output_csv: str | Path) -> Path:
    """Save validated price data to CSV.

    Args:
        prices: Validated price DataFrame.
        output_csv: Output CSV path.

    Returns:
        Absolute path to the saved CSV file.
    """
    output_path = Path(output_csv)
    ensure_directory(output_path.parent)
    prices.to_csv(output_path)
    return output_path.resolve()
