from __future__ import annotations

import pandas as pd

from src.preprocessing import align_prices_to_calendar, build_dataset_metadata, compute_simple_returns


def _make_price_panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-05", periods=6, freq="D")  # Fri..Wed
    return pd.DataFrame(
        {
            "SPY": [100.0, None, None, 101.0, 102.0, 103.0],
            "QQQ": [200.0, None, None, 201.0, 202.0, 203.0],
            "GLD": [150.0, None, None, 151.0, 152.0, 153.0],
            "TLT": [120.0, None, None, 121.0, 122.0, 123.0],
            "BTC-USD": [40000.0, 42000.0, 43000.0, 41000.0, 41500.0, 41800.0],
            "ETH-USD": [2000.0, 2100.0, 2200.0, 2050.0, 2060.0, 2080.0],
        },
        index=idx,
    )


def test_business_day_aligned_has_no_weekend_rows() -> None:
    prices = _make_price_panel()
    aligned = align_prices_to_calendar(
        prices,
        policy="business_day_aligned",
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
        require_tradfi_observation=True,
    )
    assert (aligned.index.dayofweek < 5).all()


def test_business_day_aligned_avoids_artificial_weekend_zero_returns() -> None:
    prices = _make_price_panel()
    aligned = align_prices_to_calendar(
        prices,
        policy="business_day_aligned",
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
        require_tradfi_observation=True,
    )
    simple = compute_simple_returns(aligned)
    assert (simple.index.dayofweek < 5).all()


def test_dataset_metadata_business_day_defaults() -> None:
    prices = _make_price_panel()
    aligned = align_prices_to_calendar(
        prices,
        policy="business_day_aligned",
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
        require_tradfi_observation=True,
    )
    meta = build_dataset_metadata(
        aligned,
        calendar_policy="business_day_aligned",
        annualization_factor=252,
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
    )
    assert meta["n_weekend_rows"] == 0
    assert meta["annualization_factor"] == 252.0


def test_calendar_day_policy_uses_calendar_annualization() -> None:
    prices = _make_price_panel()
    aligned = align_prices_to_calendar(
        prices,
        policy="calendar_day",
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
        require_tradfi_observation=True,
    )
    meta = build_dataset_metadata(
        aligned,
        calendar_policy="calendar_day",
        annualization_factor=365.25,
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
    )
    assert meta["annualization_factor"] == 365.25
