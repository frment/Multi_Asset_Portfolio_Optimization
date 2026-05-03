"""Portfolio performance metrics.

Each function takes a pandas Series of daily simple returns and returns
a single scalar.  All functions are stateless and easy to test individually.

Annualisation convention used throughout:
- 252 trading days per year
- risk-free rate = 0.0 (can be overridden on every call)
"""

import numpy as np
import pandas as pd

# Default annualisation factor (trading days per year).
TRADING_DAYS_PER_YEAR: int = 252


def annualised_return(daily_returns: pd.Series) -> float:
    """Compute compound annualised return from daily simple returns.

    Args:
        daily_returns: Series of daily simple returns.

    Returns:
        Annualised return as a decimal (e.g. 0.12 = 12 %).
    """
    n_days = len(daily_returns)
    if n_days == 0:
        return float("nan")

    # Compound all daily returns, then scale to one year.
    cumulative = (1.0 + daily_returns).prod()
    return float(cumulative ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)


def annualised_volatility(daily_returns: pd.Series) -> float:
    """Compute annualised volatility (standard deviation) from daily returns.

    Args:
        daily_returns: Series of daily simple returns.

    Returns:
        Annualised volatility as a decimal.
    """
    if len(daily_returns) < 2:
        return float("nan")

    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualised Sharpe ratio.

    Args:
        daily_returns: Series of daily simple returns.
        risk_free_rate: Annualised risk-free rate (default 0.0).

    Returns:
        Sharpe ratio.  Returns NaN if volatility is zero.
    """
    vol = annualised_volatility(daily_returns)
    if vol == 0.0 or np.isnan(vol):
        return float("nan")

    excess_return = annualised_return(daily_returns) - risk_free_rate
    return float(excess_return / vol)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Compute maximum drawdown from daily simple returns.

    Maximum drawdown is the largest peak-to-trough decline in the
    cumulative wealth index over the full history.

    Args:
        daily_returns: Series of daily simple returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.35 = -35 %).
    """
    if len(daily_returns) == 0:
        return float("nan")

    # Build cumulative wealth index starting at 1.
    wealth = (1.0 + daily_returns).cumprod()
    # Running peak at each point in time.
    running_peak = wealth.cummax()
    # Drawdown at each point.
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    """Compute Calmar ratio (annualised return / |max drawdown|).

    Args:
        daily_returns: Series of daily simple returns.

    Returns:
        Calmar ratio.  Returns NaN if max drawdown is zero.
    """
    ann_ret = annualised_return(daily_returns)
    mdd = max_drawdown(daily_returns)
    if mdd == 0.0 or np.isnan(mdd):
        return float("nan")

    return float(ann_ret / abs(mdd))


def expected_shortfall_historical(
    daily_returns: pd.Series,
    beta: float = 0.95,
) -> float:
    """Compute historical Expected Shortfall (CVaR) of daily losses.

    Args:
        daily_returns: Series of daily simple returns.
        beta: Confidence level in (0, 1), for example 0.95.

    Returns:
        Tail mean of losses beyond historical VaR at ``beta``.
    """
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
) -> float:
    """Compute annualized return divided by absolute historical ES.

    Args:
        daily_returns: Series of daily simple returns.
        beta: Confidence level used for Expected Shortfall.

    Returns:
        Ratio ``annualised_return / abs(expected_shortfall)``.
    """
    es = expected_shortfall_historical(daily_returns, beta=beta)
    if np.isnan(es) or np.isclose(es, 0.0):
        return float("nan")
    return float(annualised_return(daily_returns) / abs(es))


def compute_all_metrics(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute the full set of performance metrics for one return series.

    Args:
        daily_returns: Series of daily simple returns.
        risk_free_rate: Annualised risk-free rate used in Sharpe (default 0.0).

    Returns:
        Dictionary with keys: ann_return, ann_volatility, sharpe, max_drawdown, calmar.
    """
    return {
        "ann_return": annualised_return(daily_returns),
        "ann_volatility": annualised_volatility(daily_returns),
        "sharpe": sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate),
        "max_drawdown": max_drawdown(daily_returns),
        "calmar": calmar_ratio(daily_returns),
    }
