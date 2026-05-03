from __future__ import annotations

import numpy as np
import pandas as pd

from src.supervised_features import make_supervised_features


def _sample_returns(n: int = 180) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(123)
    data = rng.normal(0.0002, 0.01, size=(n, 6))
    return pd.DataFrame(
        data,
        index=idx,
        columns=["SPY", "TLT", "GLD", "QQQ", "BTC-USD", "ETH-USD"],
    )


def test_features_have_expected_columns_and_initial_nans() -> None:
    returns = _sample_returns()
    baseline = returns.mean(axis=1)

    cfg = {"calendar": {"annualization_factor": 252}, "features": {"lookbacks": {"short": 21, "medium": 63, "long": 126}}}
    feats = make_supervised_features(returns, baseline, regime_data=None, config=cfg)

    expected = {
        "portfolio_ret_21d",
        "portfolio_vol_63d",
        "btc_eth_corr_63d",
        "corr_spy_btc_126d",
        "month",
        "quarter",
    }
    assert expected.issubset(set(feats.columns))
    assert feats.iloc[:20]["portfolio_ret_21d"].isna().all()
    assert feats.index.is_monotonic_increasing
    assert feats.index.is_unique
    assert (feats.index.dayofweek < 5).all()


def test_features_do_not_use_future_values() -> None:
    returns = _sample_returns()
    baseline = returns.mean(axis=1)
    cfg = {"calendar": {"annualization_factor": 252}, "features": {"lookbacks": {"short": 21, "medium": 63, "long": 126}}}

    f1 = make_supervised_features(returns, baseline, regime_data=None, config=cfg)

    returns2 = returns.copy()
    returns2.iloc[-1, returns2.columns.get_loc("SPY")] = 10.0
    f2 = make_supervised_features(returns2, baseline, regime_data=None, config=cfg)

    # A trailing rolling feature at t-2 should be invariant to the last observation.
    assert np.isclose(f1["spy_vol_21d"].iloc[-2], f2["spy_vol_21d"].iloc[-2], equal_nan=True)
