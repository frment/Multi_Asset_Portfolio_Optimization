from __future__ import annotations

import numpy as np
import pandas as pd

from src.overlay_backtest import run_overlay_backtest


def _sample_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = pd.bdate_range("2023-01-02", periods=140)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(idx), 4)),
        index=idx,
        columns=["SPY", "TLT", "BTC-USD", "ETH-USD"],
    )

    rebal = idx[::21]
    w = pd.DataFrame(
        {
            "SPY": 0.45,
            "TLT": 0.35,
            "BTC-USD": 0.10,
            "ETH-USD": 0.10,
        },
        index=rebal,
    )

    forecasts = pd.DataFrame({"forecast_vol": 0.15, "crash_probability": 0.2}, index=idx)
    return returns, w, forecasts


def test_overlay_backtest_outputs_expected_metrics() -> None:
    returns, base_weights, forecasts = _sample_data()
    cfg = {
        "calendar": {"annualization_factor": 252},
        "overlay": {
            "dynamic_crypto_cap": {
                "enabled": True,
                "normal_cap": 0.2,
                "medium_risk_cap": 0.05,
                "high_risk_cap": 0.0,
                "crash_probability_thresholds": {"medium": 0.25, "high": 0.4},
            },
            "volatility_targeting": {"enabled": False},
            "de_risking": {"enabled": True, "defensive_assets": ["TLT"]},
        },
        "costs": {"apply_costs": True, "cost_bps": 10.0},
    }

    result = run_overlay_backtest(returns, base_weights, forecasts, regime_data=None, config=cfg)

    assert len(result.net_daily_returns) > 0
    assert "turnover_one_way" in result.turnover.columns
    for col in ["ann_return", "sharpe", "max_drawdown", "es95", "avg_turnover"]:
        assert col in result.summary.columns
