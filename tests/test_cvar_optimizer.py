"""Tests for minimum historical CVaR optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cvar_optimizer import historical_cvar_for_weights, minimise_historical_cvar


def _synthetic_returns(
    n_days: int = 700,
    tickers: list[str] | None = None,
    seed: int = 1234,
) -> pd.DataFrame:
    if tickers is None:
        tickers = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD", "TLT"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    data = rng.normal(loc=0.0002, scale=0.015, size=(n_days, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _config(max_crypto: float = 0.20) -> dict:
    return {
        "tickers": ["BTC-USD", "ETH-USD", "SPY", "QQQ", "GLD", "TLT"],
        "crypto_assets": ["BTC-USD", "ETH-USD"],
        "long_only": True,
        "max_weight": 0.35,
        "max_crypto_weight": max_crypto,
    }


def test_minimise_historical_cvar_constraints_hold():
    returns = _synthetic_returns()
    cfg = _config(max_crypto=0.20)

    w = minimise_historical_cvar(returns, config=cfg, beta=0.95)

    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w.values >= -1e-10).all()
    assert float(w.max()) <= 0.35 + 1e-8

    crypto_total = float(w[["BTC-USD", "ETH-USD"]].sum())
    assert crypto_total <= 0.20 + 1e-8


def test_no_crypto_control_forces_zero_crypto_weight():
    returns = _synthetic_returns()
    cfg = _config(max_crypto=0.0)

    w = minimise_historical_cvar(returns, config=cfg, beta=0.95)
    assert float(w[["BTC-USD", "ETH-USD"]].sum()) <= 1e-8


def test_historical_cvar_sanity_simple_example():
    # Deterministic portfolio return series: losses are [0.01, 0.02, 0.03, 0.04].
    returns = pd.DataFrame(
        {
            "A": [-0.01, -0.02, -0.03, -0.04],
        },
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    weights = pd.Series([1.0], index=["A"])

    es_75 = historical_cvar_for_weights(returns, weights, beta=0.75)
    # VaR_75 is 0.0325 (linear interpolation), ES is mean of losses >= 0.0325 => 0.04.
    assert abs(es_75 - 0.04) < 1e-12


def test_beta_sensitivity_changes_solution_or_risk_level():
    returns = _synthetic_returns()
    cfg = _config(max_crypto=0.20)

    w_95 = minimise_historical_cvar(returns, config=cfg, beta=0.95)
    w_975 = minimise_historical_cvar(returns, config=cfg, beta=0.975)

    es_95 = historical_cvar_for_weights(returns, w_95, beta=0.95)
    es_975 = historical_cvar_for_weights(returns, w_975, beta=0.975)

    assert np.isfinite(es_95)
    assert np.isfinite(es_975)
    # Either weights differ materially, or resulting tail objective differs.
    assert bool((w_95 - w_975).abs().max() > 1e-5 or abs(es_95 - es_975) > 1e-6)
