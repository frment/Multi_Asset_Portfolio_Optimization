from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def rolling_realized_vol(
    returns: pd.Series,
    lookback: int,
    annualization_factor: float,
) -> pd.Series:
    return returns.rolling(int(lookback), min_periods=int(lookback)).std() * np.sqrt(
        float(annualization_factor)
    )


def rolling_return(returns: pd.Series, lookback: int) -> pd.Series:
    return (1.0 + returns).rolling(int(lookback), min_periods=int(lookback)).apply(np.prod, raw=True) - 1.0


def rolling_drawdown(returns: pd.Series, lookback: int) -> pd.Series:
    def _window_dd(x: np.ndarray) -> float:
        wealth = np.cumprod(1.0 + x.astype(float))
        wealth = np.concatenate([[1.0], wealth])
        peak = np.maximum.accumulate(wealth)
        dd = wealth / peak - 1.0
        return float(dd.min())

    return returns.rolling(int(lookback), min_periods=int(lookback)).apply(_window_dd, raw=True)


def rolling_correlation(returns: pd.DataFrame, asset_a: str, asset_b: str, lookback: int) -> pd.Series:
    return returns[asset_a].rolling(int(lookback), min_periods=int(lookback)).corr(returns[asset_b])


def rolling_momentum(returns: pd.Series, lookback: int, skip: int = 0) -> pd.Series:
    shifted = returns.shift(int(skip)) if int(skip) > 0 else returns
    return rolling_return(shifted, lookback)


def rolling_var(returns: pd.Series, lookback: int, alpha: float = 0.95) -> pd.Series:
    q = 1.0 - float(alpha)
    return returns.rolling(int(lookback), min_periods=int(lookback)).quantile(q)


def rolling_es(returns: pd.Series, lookback: int, alpha: float = 0.95) -> pd.Series:
    q = 1.0 - float(alpha)

    def _window_es(x: np.ndarray) -> float:
        if len(x) == 0:
            return np.nan
        thr = np.quantile(x, q)
        tail = x[x <= thr]
        if len(tail) == 0:
            return float(thr)
        return float(np.mean(tail))

    return returns.rolling(int(lookback), min_periods=int(lookback)).apply(_window_es, raw=True)


def _regime_high_stress_dummy(regime_series: pd.Series) -> pd.Series:
    if np.issubdtype(regime_series.dtype, np.number):
        max_state = regime_series.dropna().max() if regime_series.notna().any() else np.nan
        return (regime_series == max_state).astype(float)

    s = regime_series.astype(str).str.lower()
    return s.str.contains("high|stress|risk_off").astype(float)


def make_supervised_features(
    returns: pd.DataFrame,
    baseline_portfolio_returns: pd.Series,
    regime_data: pd.DataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create leakage-safe Chapter 5 features using information available at t."""
    ann = float(config.get("calendar", {}).get("annualization_factor", 252.0))
    lb = config.get("features", {}).get("lookbacks", {})
    short = int(lb.get("short", 21))
    medium = int(lb.get("medium", 63))
    long = int(lb.get("long", 126))

    portfolio = baseline_portfolio_returns.reindex(returns.index).astype(float)

    feats = pd.DataFrame(index=returns.index)

    feats["portfolio_ret_21d"] = rolling_return(portfolio, short)
    feats["portfolio_ret_63d"] = rolling_return(portfolio, medium)
    feats["portfolio_vol_21d"] = rolling_realized_vol(portfolio, short, ann)
    feats["portfolio_vol_63d"] = rolling_realized_vol(portfolio, medium, ann)
    feats["portfolio_dd_63d"] = rolling_drawdown(portfolio, medium)
    feats["portfolio_es95_126d"] = rolling_es(portfolio, long, alpha=0.95)

    feats["spy_vol_21d"] = rolling_realized_vol(returns["SPY"], short, ann)
    feats["spy_vol_63d"] = rolling_realized_vol(returns["SPY"], medium, ann)
    feats["tlt_vol_21d"] = rolling_realized_vol(returns["TLT"], short, ann)
    feats["gld_vol_21d"] = rolling_realized_vol(returns["GLD"], short, ann)
    feats["spy_mom_63d"] = rolling_momentum(returns["SPY"], medium)
    feats["tlt_mom_63d"] = rolling_momentum(returns["TLT"], medium)
    feats["gld_mom_63d"] = rolling_momentum(returns["GLD"], medium)
    feats["spy_dd_63d"] = rolling_drawdown(returns["SPY"], medium)

    feats["btc_vol_21d"] = rolling_realized_vol(returns["BTC-USD"], short, ann)
    feats["btc_vol_63d"] = rolling_realized_vol(returns["BTC-USD"], medium, ann)
    feats["eth_vol_21d"] = rolling_realized_vol(returns["ETH-USD"], short, ann)
    feats["btc_mom_63d"] = rolling_momentum(returns["BTC-USD"], medium)
    feats["eth_mom_63d"] = rolling_momentum(returns["ETH-USD"], medium)
    feats["btc_dd_63d"] = rolling_drawdown(returns["BTC-USD"], medium)
    feats["eth_dd_63d"] = rolling_drawdown(returns["ETH-USD"], medium)
    feats["btc_eth_corr_63d"] = rolling_correlation(returns, "BTC-USD", "ETH-USD", medium)

    feats["corr_spy_tlt_63d"] = rolling_correlation(returns, "SPY", "TLT", medium)
    feats["corr_spy_gld_63d"] = rolling_correlation(returns, "SPY", "GLD", medium)
    feats["corr_spy_btc_63d"] = rolling_correlation(returns, "SPY", "BTC-USD", medium)
    feats["corr_spy_btc_126d"] = rolling_correlation(returns, "SPY", "BTC-USD", long)
    feats["corr_btc_eth_63d"] = rolling_correlation(returns, "BTC-USD", "ETH-USD", medium)

    if regime_data is not None and not regime_data.empty:
        regime = regime_data.reindex(feats.index)
        label_col = None
        for c in ["regime_label", "regime", "label", "state", "state_name"]:
            if c in regime.columns:
                label_col = c
                break
        if label_col is not None:
            feats["regime_label"] = regime[label_col]
            feats["regime_high_stress_dummy"] = _regime_high_stress_dummy(regime[label_col])
        else:
            feats["regime_label"] = np.nan
            feats["regime_high_stress_dummy"] = np.nan

        prob_col = None
        for c in ["regime_probability_high_stress", "high_stress_probability", "stress_probability"]:
            if c in regime.columns:
                prob_col = c
                break
        feats["regime_probability_high_stress"] = regime[prob_col] if prob_col else np.nan

        stress_col = "stress_score" if "stress_score" in regime.columns else None
        feats["stress_score"] = regime[stress_col] if stress_col else np.nan
    else:
        feats["regime_label"] = np.nan
        feats["regime_high_stress_dummy"] = np.nan
        feats["regime_probability_high_stress"] = np.nan
        feats["stress_score"] = np.nan

    feats["month"] = feats.index.month.astype(float)
    feats["quarter"] = feats.index.quarter.astype(float)

    feats = feats.sort_index()
    feats = feats[~feats.index.duplicated(keep="first")]
    feats.index.name = returns.index.name or "Date"

    return feats
