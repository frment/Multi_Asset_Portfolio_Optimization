"""Tests for Chapter 3 tail-risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import annualised_return, expected_shortfall_historical, return_over_es


def test_expected_shortfall_historical_simple_losses():
    returns = pd.Series([-0.01, -0.02, -0.03, -0.04])
    es = expected_shortfall_historical(returns, beta=0.75)
    assert abs(es - 0.04) < 1e-12


def test_expected_shortfall_historical_nan_on_empty():
    es = expected_shortfall_historical(pd.Series(dtype=float), beta=0.95)
    assert np.isnan(es)


def test_return_over_es_matches_definition():
    returns = pd.Series([0.01, -0.02, 0.015, -0.03, 0.01])
    beta = 0.8

    ratio = return_over_es(returns, beta=beta)
    expected = annualised_return(returns) / abs(expected_shortfall_historical(returns, beta=beta))

    assert np.isfinite(ratio)
    assert abs(ratio - expected) < 1e-12


def test_return_over_es_nan_when_es_zero_like():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    ratio = return_over_es(returns, beta=0.95)
    assert np.isnan(ratio)
