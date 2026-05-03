from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.regime_detection import detect_regimes


def _features(n_rows: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n_rows)
    rng = np.random.default_rng(1)
    data = rng.normal(size=(n_rows, 10))
    cols = [
        "realized_vol_spy_63d",
        "realized_vol_btc_usd_63d",
        "drawdown_spy_126d",
        "drawdown_btc_usd_126d",
        "corr_spy_tlt_126d",
        "corr_spy_btc_usd_126d",
        "corr_btc_usd_eth_usd_126d",
        "momentum_spy_126d",
        "momentum_btc_usd_126d",
        "momentum_tlt_126d",
    ]
    return pd.DataFrame(data, index=idx, columns=cols)


def _cfg(allow_fallback: bool) -> dict:
    return {
        "detection": {
            "random_state": 42,
            "primary_model": "hmm",
            "allow_fallback": allow_fallback,
            "n_states": 2,
            "hmm": {"covariance_type": "full", "n_iter": 50, "tol": 1e-4},
            "candidate_models": [{"model": "kmeans", "n_states": 2}],
            "stress_score_features": {
                "realized_vol_spy_63d": 1.0,
                "realized_vol_btc_usd_63d": 1.0,
                "drawdown_spy_126d": -1.0,
                "drawdown_btc_usd_126d": -1.0,
                "corr_spy_btc_usd_126d": 1.0,
                "corr_spy_tlt_126d": 1.0,
            },
            "state_names": {2: {0: "Low", 1: "High"}},
            "outputs": {},
        },
        "dataset_metadata": {
            "calendar_policy": "business_day_aligned",
            "annualization_factor": 252,
        },
    }


def test_hmm_required_raises_when_missing_and_no_fallback() -> None:
    with patch("src.regime_detection._HMM_AVAILABLE", False):
        with pytest.raises(ImportError, match="hmmlearn"):
            detect_regimes(_features(), _cfg(allow_fallback=False))


def test_model_used_matches_real_model() -> None:
    with patch("src.regime_detection._HMM_AVAILABLE", False):
        result = detect_regimes(_features(), _cfg(allow_fallback=True))

    assert result.model_id.startswith("kmeans")
    assert result.extra["model_used"] == result.model_id
    assert result.extra["primary_model_requested"] == "hmm"
