from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model_evaluation import evaluate_all_models
from src.overlay_backtest import run_overlay_backtest
from src.supervised_features import make_supervised_features
from src.supervised_models import fit_predict_walk_forward
from src.supervised_targets import make_supervised_targets
from src.supervised_validation import align_features_targets, generate_walk_forward_splits


def test_chapter5_pipeline_smoke(tmp_path: Path) -> None:
    idx = pd.bdate_range("2021-01-01", periods=300)
    rng = np.random.default_rng(99)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(idx), 6)),
        index=idx,
        columns=["SPY", "TLT", "GLD", "QQQ", "BTC-USD", "ETH-USD"],
    )
    baseline = returns.mean(axis=1).rename("min_variance")

    cfg = {
        "calendar": {"annualization_factor": 252, "policy": "business_day_aligned"},
        "targets": {
            "volatility": {"horizons": [21, 63]},
            "drawdown_event": {"thresholds": {"portfolio_21d": -0.05, "portfolio_63d": -0.1, "btc_21d": -0.15}},
            "stress_event": {"horizon": 21, "quantile": 0.1},
        },
        "features": {"lookbacks": {"short": 21, "medium": 63, "long": 126}},
        "overlay": {
            "dynamic_crypto_cap": {
                "enabled": True,
                "normal_cap": 0.2,
                "medium_risk_cap": 0.05,
                "high_risk_cap": 0.0,
                "crash_probability_thresholds": {"medium": 0.25, "high": 0.4},
            },
            "volatility_targeting": {"enabled": False},
            "de_risking": {"enabled": False},
        },
        "costs": {"apply_costs": True, "cost_bps": 10.0},
    }

    targets = make_supervised_targets(returns, baseline, cfg)
    features = make_supervised_features(returns, baseline, regime_data=None, config=cfg)

    X, y = align_features_targets(features, targets, "target_portfolio_vol_21d")
    splits = generate_walk_forward_splits(X.index, 180, 21, 21, 120, 5)
    pred = fit_predict_walk_forward(X, y, "naive_rolling_vol", "regression", splits)

    out_dir = tmp_path / "chapter5"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "predictions_target_portfolio_vol_21d_naive_rolling_vol.csv", index=False)

    scores, calibration, buckets = evaluate_all_models(out_dir)
    assert len(scores) >= 1
    _ = calibration
    _ = buckets

    rebal_dates = idx[::21]
    base_weights = pd.DataFrame(
        {
            "SPY": 0.4,
            "TLT": 0.3,
            "GLD": 0.2,
            "BTC-USD": 0.05,
            "ETH-USD": 0.05,
        },
        index=rebal_dates,
    )

    forecasts = pd.DataFrame(index=idx)
    forecasts["forecast_vol"] = features["portfolio_vol_21d"].reindex(idx).ffill()
    forecasts["crash_probability"] = 0.2

    result = run_overlay_backtest(
        returns=returns[["SPY", "TLT", "GLD", "BTC-USD", "ETH-USD"]],
        base_weights_history=base_weights,
        forecasts=forecasts,
        regime_data=None,
        config=cfg,
    )

    assert len(result.net_daily_returns) > 0
    assert "strategy" in result.summary.columns
