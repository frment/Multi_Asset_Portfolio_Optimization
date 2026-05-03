from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import run_min_variance_backtest
from src.preprocessing import align_prices_to_calendar, build_dataset_metadata, compute_simple_returns


def test_smoke_pipeline_outputs_metadata_and_backtest() -> None:
    idx = pd.date_range("2023-01-01", periods=420, freq="D")
    rng = np.random.default_rng(12)

    prices = pd.DataFrame(
        {
            "SPY": 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(idx))),
            "QQQ": 100 * np.cumprod(1 + rng.normal(0.0003, 0.012, len(idx))),
            "GLD": 100 * np.cumprod(1 + rng.normal(0.0001, 0.008, len(idx))),
            "TLT": 100 * np.cumprod(1 + rng.normal(0.0001, 0.007, len(idx))),
            "BTC-USD": 100 * np.cumprod(1 + rng.normal(0.0005, 0.025, len(idx))),
            "ETH-USD": 100 * np.cumprod(1 + rng.normal(0.0007, 0.03, len(idx))),
        },
        index=idx,
    )

    # emulate TradFi no-weekend observations
    weekend_mask = prices.index.dayofweek >= 5
    prices.loc[weekend_mask, ["SPY", "QQQ", "GLD", "TLT"]] = np.nan

    aligned = align_prices_to_calendar(
        prices,
        policy="business_day_aligned",
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
        require_tradfi_observation=True,
    )
    returns = compute_simple_returns(aligned)

    metadata = build_dataset_metadata(
        aligned,
        calendar_policy="business_day_aligned",
        annualization_factor=252,
        tradfi_assets=["SPY", "QQQ", "GLD", "TLT"],
        crypto_assets=["BTC-USD", "ETH-USD"],
    )

    config = {
        "tickers": ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD", "TLT"],
        "crypto_assets": ["BTC-USD", "ETH-USD"],
        "long_only": True,
        "max_weight": 0.6,
        "max_crypto_weight": 0.3,
    }

    portfolio_returns, weights_history, turnover_history = run_min_variance_backtest(
        returns=returns,
        optimizer_config=config,
        lookback_window=126,
        rebalance_frequency="monthly",
        holding_return_method="drifted_buy_and_hold",
        allow_weekend_rebalances=False,
    )

    assert metadata["n_weekend_rows"] == 0
    assert metadata["annualization_factor"] == 252.0
    assert len(portfolio_returns) > 0
    assert len(weights_history) > 0
    assert len(turnover_history) == len(weights_history)
