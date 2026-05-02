"""Tests for covariance estimators used by the optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.optimizer import estimate_covariance, minimise_variance


def _make_synthetic_returns(
    n_days: int = 700,
    tickers: list[str] | None = None,
    seed: int = 123,
) -> pd.DataFrame:
    if tickers is None:
        tickers = ["A", "B", "C", "D"]

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2019-01-02", periods=n_days)
    data = rng.normal(loc=0.0002, scale=0.01, size=(n_days, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _config() -> dict:
    return {
        "tickers": ["A", "B", "C", "D"],
        "crypto_assets": [],
        "long_only": True,
        "max_weight": 1.0,
        "max_crypto_weight": 1.0,
    }


def test_ledoit_wolf_covariance_shape_and_numeric_usability():
    returns = _make_synthetic_returns()
    cov = estimate_covariance(returns, covariance_method="ledoit_wolf")

    assert cov.shape == (returns.shape[1], returns.shape[1])
    assert list(cov.index) == list(returns.columns)
    assert list(cov.columns) == list(returns.columns)
    assert np.isfinite(cov.values).all()

    weights = minimise_variance(
        returns,
        config=_config(),
        covariance_method="ledoit_wolf",
    )
    assert np.isfinite(weights.values).all()
    assert abs(float(weights.sum()) - 1.0) < 1e-6
    assert (weights.values >= -1e-8).all()


def test_sample_default_matches_explicit_sample_baseline():
    returns = _make_synthetic_returns()

    weights_default = minimise_variance(returns, config=_config())
    weights_explicit = minimise_variance(
        returns,
        config=_config(),
        covariance_method="sample",
    )

    pd.testing.assert_series_equal(weights_default, weights_explicit, check_exact=False, atol=1e-10)
