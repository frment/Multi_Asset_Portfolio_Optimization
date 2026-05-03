from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk_overlay import apply_crypto_cap, make_overlay_decision


def test_apply_crypto_cap_reduces_crypto_and_keeps_sum_one() -> None:
    w = pd.Series({"SPY": 0.4, "TLT": 0.2, "BTC-USD": 0.3, "ETH-USD": 0.1})
    out = apply_crypto_cap(w, ["BTC-USD", "ETH-USD"], crypto_cap=0.2)
    assert out[["BTC-USD", "ETH-USD"]].sum() <= 0.200001
    assert np.isclose(out.sum(), 1.0)
    assert (out >= 0.0).all()


def test_overlay_reason_codes_and_no_unnecessary_change_under_cap() -> None:
    base = pd.Series({"SPY": 0.5, "TLT": 0.4, "BTC-USD": 0.05, "ETH-USD": 0.05})
    cfg = {
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
        }
    }

    decision = make_overlay_decision(
        date=pd.Timestamp("2024-01-31"),
        base_weights=base,
        forecasts={"forecast_vol": 0.1, "crash_probability": 0.1},
        regime_state={"regime_high_stress_dummy": 0},
        config=cfg,
    )

    adj = decision["adjusted_weights"]
    assert np.isclose(adj.sum(), 1.0)
    assert np.isclose(adj[["BTC-USD", "ETH-USD"]].sum(), 0.10)
    assert isinstance(decision["reason"], str)
