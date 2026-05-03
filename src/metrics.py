"""Portfolio performance metrics.

All annualized metrics accept an explicit `annualization_factor`.
Baseline business-day aligned convention is 252 observations per year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_ANNUALIZATION_FACTOR: float = 252.0
# Backward-compatible alias used by existing modules/tests.
TRADING_DAYS_PER_YEAR: float = DEFAULT_ANNUALIZATION_FACTOR


def annualised_return(
    daily_returns: pd.Series,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Compute compound annualized return from simple returns."""
    n_days = len(daily_returns)
    if n_days == 0:
        return float("nan")

    cumulative = float((1.0 + daily_returns).prod())
    return float(cumulative ** (float(annualization_factor) / n_days) - 1.0)


def annualised_volatility(
    daily_returns: pd.Series,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Compute annualized volatility from daily simple returns."""
    if len(daily_returns) < 2:
        return float("nan")
    return float(daily_returns.std() * np.sqrt(float(annualization_factor)))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Compute annualized Sharpe ratio."""
    vol = annualised_volatility(daily_returns, annualization_factor=annualization_factor)
    if vol == 0.0 or np.isnan(vol):
        return float("nan")

    excess_return = annualised_return(
        daily_returns,
        annualization_factor=annualization_factor,
    ) - risk_free_rate
    return float(excess_return / vol)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Compute maximum drawdown from daily simple returns."""
    if len(daily_returns) == 0:
        return float("nan")

    wealth = (1.0 + daily_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def calmar_ratio(
    daily_returns: pd.Series,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Compute Calmar ratio (annualized return / |max drawdown|)."""
    ann_ret = annualised_return(daily_returns, annualization_factor=annualization_factor)
    mdd = max_drawdown(daily_returns)
    if mdd == 0.0 or np.isnan(mdd):
        return float("nan")
    return float(ann_ret / abs(mdd))


def expected_shortfall_historical(
    daily_returns: pd.Series,
    beta: float = 0.95,
) -> float:
    """Compute historical Expected Shortfall (CVaR) of daily losses."""
    if len(daily_returns) == 0:
        return float("nan")
    if not 0.0 < float(beta) < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")

    losses = -daily_returns.astype(float)
    var_beta = float(np.quantile(losses, beta))
    tail = losses[losses >= var_beta]
    if len(tail) == 0:
        return var_beta
    return float(tail.mean())


def return_over_es(
    daily_returns: pd.Series,
    beta: float = 0.95,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> float:
    """Compute annualized return divided by absolute historical ES."""
    es = expected_shortfall_historical(daily_returns, beta=beta)
    if np.isnan(es) or np.isclose(es, 0.0):
        return float("nan")
    return float(
        annualised_return(daily_returns, annualization_factor=annualization_factor) / abs(es)
    )


def compute_all_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR,
) -> dict[str, float]:
    """Compute the full set of performance metrics for one return series."""
    return {
        "ann_return": annualised_return(
            daily_returns,
            annualization_factor=annualization_factor,
        ),
        "ann_volatility": annualised_volatility(
            daily_returns,
            annualization_factor=annualization_factor,
        ),
        "sharpe": sharpe_ratio(
            daily_returns,
            risk_free_rate=risk_free_rate,
            annualization_factor=annualization_factor,
        ),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(
            daily_returns,
            annualization_factor=annualization_factor,
        ),
    }


# American spelling aliases for public API readability.
def annualized_return(daily_returns: pd.Series, annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR) -> float:
    return annualised_return(daily_returns, annualization_factor=annualization_factor)


def annualized_volatility(daily_returns: pd.Series, annualization_factor: float = DEFAULT_ANNUALIZATION_FACTOR) -> float:
    return annualised_volatility(daily_returns, annualization_factor=annualization_factor)
