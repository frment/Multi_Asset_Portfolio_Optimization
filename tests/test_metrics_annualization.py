from __future__ import annotations

import pandas as pd

from src.metrics import annualized_return, annualized_volatility, compute_all_metrics, sharpe_ratio


def test_metrics_change_with_annualization_factor() -> None:
    series = pd.Series([0.001, -0.0005, 0.0012, 0.0008, -0.0004] * 40)

    ann_ret_252 = annualized_return(series, annualization_factor=252)
    ann_ret_365 = annualized_return(series, annualization_factor=365.25)
    assert ann_ret_252 != ann_ret_365

    vol_252 = annualized_volatility(series, annualization_factor=252)
    vol_365 = annualized_volatility(series, annualization_factor=365.25)
    assert vol_252 != vol_365

    sharpe_252 = sharpe_ratio(series, annualization_factor=252)
    sharpe_365 = sharpe_ratio(series, annualization_factor=365.25)
    assert sharpe_252 != sharpe_365


def test_compute_all_metrics_accepts_annualization_factor() -> None:
    series = pd.Series([0.001, 0.0, -0.001, 0.0005] * 30)
    metrics = compute_all_metrics(series, annualization_factor=365.25)
    assert "ann_return" in metrics
    assert "ann_volatility" in metrics
    assert "sharpe" in metrics
