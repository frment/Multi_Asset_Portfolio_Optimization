"""
Chapter 4 — Regime Features
============================
Builds a daily panel of market-regime features from simple-return data.

All features are computed in a strictly causal fashion: the value at date t
uses only information available up to and including t (no look-ahead).

Feature catalogue
-----------------
realized_vol_spy_63d    : Annualised realised vol of SPY over a 63-day window.
realized_vol_btc_63d    : Annualised realised vol of BTC over a 63-day window.
drawdown_spy_126d       : Rolling 126-day drawdown of SPY (always ≤ 0).
drawdown_btc_126d       : Rolling 126-day drawdown of BTC (always ≤ 0).
corr_spy_tlt_126d       : Rolling 126-day Pearson correlation SPY ↔ TLT.
corr_spy_btc_126d       : Rolling 126-day Pearson correlation SPY ↔ BTC.
corr_btc_eth_126d       : Rolling 126-day Pearson correlation BTC ↔ ETH.
momentum_spy_126d       : Cumulative simple return of SPY over 126 days.
momentum_btc_126d       : Cumulative simple return of BTC over 126 days.
momentum_tlt_126d       : Cumulative simple return of TLT over 126 days.

All features are returned raw (not standardised). Standardisation is deferred
to the regime-detection module (regime_detection.py, Phase 2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_returns(prices_or_simple_returns: pd.DataFrame) -> pd.DataFrame:
    """Convert simple returns to log-returns.

    r_log = log(1 + r_simple)
    """
    return np.log1p(prices_or_simple_returns)


def _realized_volatility(
    simple_returns: pd.DataFrame,
    ticker: str,
    window: int,
    min_periods: int,
    annualization_factor: int,
) -> pd.Series:
    """Annualised realised volatility from a rolling window of log-returns.

    The value at date t uses only the [t-window+1 … t] window.
    Annualisation: sigma_annual = sigma_daily * sqrt(annualization_factor).

    Parameters
    ----------
    simple_returns : DataFrame with daily simple returns indexed by date.
    ticker         : Column name in simple_returns.
    window         : Rolling window length in trading days.
    min_periods    : Minimum number of observations required for a valid result.
    annualization_factor : Typically 252 trading days per year.

    Returns
    -------
    pd.Series aligned to the input index.
    """
    log_ret = _log_returns(simple_returns[[ticker]])
    vol = (
        log_ret[ticker]
        .rolling(window=window, min_periods=min_periods)
        .std()
        .mul(np.sqrt(annualization_factor))
    )
    vol.name = f"realized_vol_{ticker.lower().replace('-', '_')}_{window}d"
    return vol


def _rolling_drawdown(
    simple_returns: pd.DataFrame,
    ticker: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling maximum drawdown from a rolling peak.

    drawdown(t) = (cumulative_wealth(t) / peak_over_window(t)) - 1

    The peak is the maximum cumulative-wealth level observed within the
    [t-window+1 … t] window, so no future information is used.

    Parameters
    ----------
    simple_returns : DataFrame with daily simple returns indexed by date.
    ticker         : Column name.
    window         : Rolling window in trading days.
    min_periods    : Minimum observations required.

    Returns
    -------
    pd.Series with values ≤ 0.
    """
    r = simple_returns[ticker]
    # Build cumulative wealth starting from an arbitrary base (1.0).
    # We work with a rolling window, so we rebase inside each window.
    # Implementation: at each date t, take the sub-series, build wealth, find drawdown.
    # Efficient vectorised approach using expanding cummax within a rolling window.

    # Step 1: cumulative wealth over the full series (anchored at first date).
    cum_wealth = (1 + r).cumprod()

    # Step 2: rolling peak of cumulative wealth within the window.
    rolling_peak = cum_wealth.rolling(window=window, min_periods=min_periods).max()

    # Step 3: drawdown relative to the rolling peak.
    dd = (cum_wealth / rolling_peak) - 1.0

    dd.name = f"drawdown_{ticker.lower().replace('-', '_')}_{window}d"
    return dd


def _rolling_correlation(
    simple_returns: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Rolling Pearson correlation between two assets.

    Parameters
    ----------
    simple_returns : DataFrame with daily simple returns.
    ticker_a, ticker_b : Column names.
    window, min_periods : Rolling parameters.

    Returns
    -------
    pd.Series with values in [-1, 1].
    """
    corr = (
        simple_returns[ticker_a]
        .rolling(window=window, min_periods=min_periods)
        .corr(simple_returns[ticker_b])
    )
    label_a = ticker_a.lower().replace("-", "_")
    label_b = ticker_b.lower().replace("-", "_")
    corr.name = f"corr_{label_a}_{label_b}_{window}d"
    return corr


def _rolling_momentum(
    simple_returns: pd.DataFrame,
    ticker: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Cumulative simple return (momentum) over a rolling window.

    momentum(t) = product(1 + r[t-window+1 .. t]) - 1

    Computed as exp(sum of log-returns) - 1 for numerical stability.

    Parameters
    ----------
    simple_returns : DataFrame with daily simple returns.
    ticker         : Column name.
    window, min_periods : Rolling parameters.

    Returns
    -------
    pd.Series.
    """
    log_ret = _log_returns(simple_returns[[ticker]])[ticker]
    cum_log = log_ret.rolling(window=window, min_periods=min_periods).sum()
    mom = np.expm1(cum_log)
    mom.name = f"momentum_{ticker.lower().replace('-', '_')}_{window}d"
    return mom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_regime_features(
    simple_returns: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Construct the full regime-feature panel from daily simple returns.

    All features are built strictly causally (no look-ahead).
    The returned DataFrame has no NaN rows (rows with any NaN are dropped).

    Parameters
    ----------
    simple_returns : Daily simple-return DataFrame indexed by date (DatetimeIndex
                     or string dates sortable as dates). Columns are asset tickers.
    cfg            : Dictionary loaded from ``config/regime_analysis.yaml``.

    Returns
    -------
    pd.DataFrame — one row per trading day, one column per feature, no NaNs.
    """
    # Ensure chronological order.
    returns = simple_returns.sort_index().copy()

    feat_cfg = cfg["features"]
    vol_cfg = feat_cfg["realized_volatility"]
    annualization_factor = float(
        cfg.get("dataset_metadata", {}).get(
            "annualization_factor",
            vol_cfg["annualization_factor"],
        )
    )
    dd_cfg = feat_cfg["drawdown"]
    corr_cfg = feat_cfg["correlation"]
    mom_cfg = feat_cfg["momentum"]

    series_list: list[pd.Series] = []

    # 1–2  Realized volatility
    for ticker in vol_cfg["tickers"]:
        s = _realized_volatility(
            returns,
            ticker=ticker,
            window=vol_cfg["window"],
            min_periods=vol_cfg["min_periods"],
            annualization_factor=annualization_factor,
        )
        series_list.append(s)
        logger.debug("Built feature: %s", s.name)

    # 3–4  Rolling drawdown
    for ticker in dd_cfg["tickers"]:
        s = _rolling_drawdown(
            returns,
            ticker=ticker,
            window=dd_cfg["window"],
            min_periods=dd_cfg["min_periods"],
        )
        series_list.append(s)
        logger.debug("Built feature: %s", s.name)

    # 5–7  Rolling pairwise correlations
    for pair in corr_cfg["pairs"]:
        ticker_a, ticker_b = pair
        s = _rolling_correlation(
            returns,
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            window=corr_cfg["window"],
            min_periods=corr_cfg["min_periods"],
        )
        series_list.append(s)
        logger.debug("Built feature: %s", s.name)

    # 8–10  Momentum
    for ticker in mom_cfg["tickers"]:
        s = _rolling_momentum(
            returns,
            ticker=ticker,
            window=mom_cfg["window"],
            min_periods=mom_cfg["min_periods"],
        )
        series_list.append(s)
        logger.debug("Built feature: %s", s.name)

    # Assemble panel
    panel = pd.concat(series_list, axis=1)
    panel.index.name = "date"

    # Drop rows that still contain NaNs (burn-in period at the start).
    n_before = len(panel)
    panel = panel.dropna()
    n_after = len(panel)
    dropped = n_before - n_after
    logger.info(
        "Regime features built: %d rows (dropped %d NaN burn-in rows). "
        "Date range: %s → %s. Columns: %s",
        n_after,
        dropped,
        panel.index.min(),
        panel.index.max(),
        list(panel.columns),
    )

    return panel


def load_and_build(cfg: dict[str, Any], project_root: Path) -> pd.DataFrame:
    """Load simple returns from disk and build the feature panel.

    Parameters
    ----------
    cfg          : Dictionary loaded from ``config/regime_analysis.yaml``.
    project_root : Absolute path to the project root directory.

    Returns
    -------
    pd.DataFrame — regime feature panel, no NaNs.
    """
    returns_path = project_root / cfg["paths"]["returns_simple"]
    logger.info("Loading simple returns from %s", returns_path)
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    returns.index = pd.to_datetime(returns.index)
    returns.sort_index(inplace=True)
    return build_regime_features(returns, cfg)


def save_features(features: pd.DataFrame, cfg: dict[str, Any], project_root: Path) -> Path:
    """Persist the feature panel to CSV.

    Parameters
    ----------
    features     : DataFrame returned by ``build_regime_features``.
    cfg          : Dictionary loaded from ``config/regime_analysis.yaml``.
    project_root : Absolute path to the project root directory.

    Returns
    -------
    Path to the saved file.
    """
    output_path = project_root / cfg["paths"]["regime_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path)
    logger.info("Regime features saved to %s", output_path)
    return output_path
