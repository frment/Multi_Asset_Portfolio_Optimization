"""Benchmark portfolio construction.

Each function receives the full daily returns DataFrame and returns a single
pandas Series representing the benchmark's daily portfolio return.

Benchmark weights are read from config/settings.yaml wherever possible,
keeping the implementation config-driven and easy to update without
touching code.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_settings


def _load_benchmark_weights(settings_cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Extract benchmark weight dictionaries from settings config.

    Returns:
        Dictionary keyed by benchmark name, each value being a ticker->weight dict.
    """
    benchmarks_cfg = settings_cfg.get("benchmarks", {})

    weights: dict[str, dict[str, float]] = {}

    sixty_forty = benchmarks_cfg.get("sixty_forty", {})
    if sixty_forty.get("weights"):
        weights["sixty_forty"] = {str(k): float(v) for k, v in sixty_forty["weights"].items()}

    fixed_crypto = benchmarks_cfg.get("fixed_small_crypto", {})
    if fixed_crypto.get("weights"):
        weights["fixed_small_crypto"] = {
            str(k): float(v) for k, v in fixed_crypto["weights"].items()
        }

    return weights


def _apply_weights(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Apply a fixed weight vector to a returns DataFrame.

    Weights do not need to match all columns in returns; only the columns
    present in weights are used, which keeps this function flexible.

    Args:
        returns: Daily returns DataFrame with ticker columns.
        weights: Dict mapping ticker -> portfolio weight.

    Returns:
        Series of daily portfolio returns.
    """
    missing = [ticker for ticker in weights if ticker not in returns.columns]
    if missing:
        raise ValueError(
            "Benchmark weight dict references tickers not found in returns: "
            + ", ".join(missing)
        )

    tickers = list(weights.keys())
    weight_array = np.array([weights[ticker] for ticker in tickers])

    # Verify weights sum to 1 (warn, but do not crash, to be beginner-friendly).
    total = weight_array.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        import warnings
        warnings.warn(
            f"Benchmark weights sum to {total:.6f}, not 1.0. "
            "Results may not represent a fully-invested portfolio.",
            UserWarning,
            stacklevel=2,
        )

    portfolio_returns = returns[tickers].values @ weight_array
    return pd.Series(portfolio_returns, index=returns.index)


def equal_weight_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute daily returns for the equal-weight (1/N) benchmark.

    Each asset receives the same weight: 1 / number_of_assets.

    Args:
        returns: Daily returns DataFrame.

    Returns:
        Series of daily benchmark returns.
    """
    n = len(returns.columns)
    weights = {ticker: 1.0 / n for ticker in returns.columns}
    return _apply_weights(returns, weights)


def sixty_forty_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute daily returns for the 60/40 benchmark.

    Weights are read from config/settings.yaml (benchmarks.sixty_forty.weights).
    Expected: 60 % SPY, 40 % TLT.

    Args:
        returns: Daily returns DataFrame.

    Returns:
        Series of daily benchmark returns.
    """
    settings_cfg = load_settings()
    all_weights = _load_benchmark_weights(settings_cfg)

    weights = all_weights.get("sixty_forty")
    if not weights:
        # Fallback in case config entry is missing.
        weights = {"SPY": 0.60, "TLT": 0.40}

    return _apply_weights(returns, weights)


def fixed_small_crypto_returns(returns: pd.DataFrame) -> pd.Series:
    """Compute daily returns for the fixed small-crypto benchmark.

    Weights are read from config/settings.yaml
    (benchmarks.fixed_small_crypto.weights).
    Expected: 5 % BTC-USD, 5 % ETH-USD, 22.5 % each for SPY / QQQ / GLD / TLT.

    Args:
        returns: Daily returns DataFrame.

    Returns:
        Series of daily benchmark returns.
    """
    settings_cfg = load_settings()
    all_weights = _load_benchmark_weights(settings_cfg)

    weights = all_weights.get("fixed_small_crypto")
    if not weights:
        # Fallback if config entry is missing.
        weights = {
            "BTC-USD": 0.05,
            "ETH-USD": 0.05,
            "SPY": 0.225,
            "QQQ": 0.225,
            "GLD": 0.225,
            "TLT": 0.225,
        }

    return _apply_weights(returns, weights)


def build_all_benchmarks(returns: pd.DataFrame) -> pd.DataFrame:
    """Build all benchmark return series and return them as a wide DataFrame.

    Args:
        returns: Daily returns DataFrame with ticker columns.

    Returns:
        DataFrame with one column per benchmark, same date index as returns.
    """
    return pd.DataFrame(
        {
            "equal_weight": equal_weight_returns(returns),
            "sixty_forty": sixty_forty_returns(returns),
            "fixed_small_crypto": fixed_small_crypto_returns(returns),
        }
    )
