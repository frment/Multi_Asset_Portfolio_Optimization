"""Preprocessing utilities for building clean price and return datasets.

Baseline policy (after methodological audit):
1. Use business-day aligned calendar based on observed TradFi assets.
2. Keep only dates with real observations for required TradFi assets.
3. Sample crypto prices on those same dates.
4. Compute returns on the aligned index.

Optional sensitivity mode keeps a full calendar-day index.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_calendar_settings, load_assets, load_settings
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

    calendar_cfg = get_calendar_settings(settings_cfg)

    return {
        "tickers": [str(ticker) for ticker in tickers],
        "raw_prices_csv": data_raw_dir / "prices_raw.csv",
        "prices_clean_csv": data_processed_dir / "prices_clean.csv",
        "prices_aligned_csv": data_processed_dir / "prices_aligned.csv",
        "returns_simple_csv": data_processed_dir / "returns_simple.csv",
        "returns_log_csv": data_processed_dir / "returns_log.csv",
        "dataset_metadata_json": data_processed_dir / "dataset_metadata.json",
        "calendar": calendar_cfg,
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


def align_prices_to_calendar(
    prices: pd.DataFrame,
    policy: str,
    tradfi_assets: list[str],
    crypto_assets: list[str],
    require_tradfi_observation: bool = True,
) -> pd.DataFrame:
    """Align a multi-asset price panel to an explicit calendar policy.

    Args:
        prices: Price panel with DatetimeIndex and asset columns.
        policy: Calendar policy. One of "business_day_aligned" or "calendar_day".
        tradfi_assets: TradFi assets that define the baseline business-day index.
        crypto_assets: Crypto assets sampled on the chosen index.
        require_tradfi_observation: If True, require all tradfi assets present.

    Returns:
        Calendar-aligned and NaN-free price panel.
    """
    if prices.empty:
        raise ValueError("Cannot align empty price DataFrame.")

    out = prices.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]

    required_assets = list(dict.fromkeys([*tradfi_assets, *crypto_assets]))
    missing_assets = [asset for asset in required_assets if asset not in out.columns]
    if missing_assets:
        raise ValueError("Missing required assets in price panel: " + ", ".join(missing_assets))

    out = out.loc[:, required_assets]
    policy_key = str(policy).lower().strip()

    if policy_key == "business_day_aligned":
        tradfi_prices = out[tradfi_assets]
        if require_tradfi_observation:
            valid_mask = tradfi_prices.notna().all(axis=1)
        else:
            valid_mask = tradfi_prices.notna().any(axis=1)

        out = out.loc[valid_mask].copy()
        out = out[out.index.dayofweek < 5]
        # Limited fill addresses occasional single-day crypto gaps without creating
        # synthetic weekend rows for TradFi assets.
        out[crypto_assets] = out[crypto_assets].ffill(limit=1)
        out = out.dropna(how="any")

    elif policy_key == "calendar_day":
        full_index = pd.date_range(out.index.min(), out.index.max(), freq="D")
        out = out.reindex(full_index)
        out.index.name = "Date"
        # Explicitly allow TradFi forward-fill in calendar-day sensitivity mode.
        out[tradfi_assets] = out[tradfi_assets].ffill()
        out[crypto_assets] = out[crypto_assets].ffill(limit=1)
        out = out.dropna(how="any")
    else:
        raise ValueError(
            f"Unsupported calendar policy: {policy}. Use 'business_day_aligned' or 'calendar_day'."
        )

    if out.empty:
        raise ValueError("Calendar alignment produced an empty price panel.")

    out.index.name = "Date"
    return out


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


def build_dataset_metadata(
    prices: pd.DataFrame,
    *,
    calendar_policy: str,
    annualization_factor: float,
    tradfi_assets: list[str],
    crypto_assets: list[str],
) -> dict[str, Any]:
    """Build metadata describing the processed dataset and methodology."""
    weekend_rows = int((prices.index.dayofweek >= 5).sum())
    return {
        "calendar_policy": str(calendar_policy),
        "annualization_factor": float(annualization_factor),
        "start_date": prices.index.min().date().isoformat(),
        "end_date": prices.index.max().date().isoformat(),
        "n_observations": int(len(prices)),
        "n_weekend_rows": weekend_rows,
        "tradfi_assets": [str(x) for x in tradfi_assets],
        "crypto_assets": [str(x) for x in crypto_assets],
    }


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


def save_metadata_json(metadata: dict[str, Any], output_json: str | Path) -> Path:
    """Save dataset metadata as JSON."""
    output_path = Path(output_json)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return output_path.resolve()
