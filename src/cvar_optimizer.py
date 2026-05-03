"""Historical CVaR (Expected Shortfall) portfolio optimizer.

Implements minimum historical CVaR using the Rockafellar-Uryasev linear
program formulation:

    min_w,zeta,u  zeta + (1 / ((1 - beta) * T)) * sum_t u_t

subject to:
    u_t >= loss_t(w) - zeta
    u_t >= 0
    sum_i w_i = 1
    long-only and per-asset bounds
    total crypto exposure cap

where loss_t(w) = -r_t' w and beta is the tail confidence level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.optimizer import load_optimizer_config, validate_weights


def historical_cvar_for_weights(
    returns: pd.DataFrame,
    weights: pd.Series,
    beta: float = 0.95,
) -> float:
    """Compute historical CVaR (Expected Shortfall) of portfolio losses.

    Args:
        returns: Daily simple returns indexed by date.
        weights: Portfolio weights indexed by ticker.
        beta: Tail confidence level in (0, 1).

    Returns:
        Historical CVaR of losses (non-negative in typical cases).
    """
    if not 0.0 < float(beta) < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")

    if returns.empty:
        raise ValueError("returns must be non-empty.")

    tickers = [t for t in weights.index if t in returns.columns]
    if not tickers:
        raise ValueError("No overlapping tickers between returns and weights.")

    port_returns = (returns[tickers] * weights.reindex(tickers).values).sum(axis=1)
    losses = -port_returns.values

    var_beta = float(np.quantile(losses, beta))
    tail = losses[losses >= var_beta]
    if tail.size == 0:
        return var_beta
    return float(np.mean(tail))


def minimise_historical_cvar(
    returns: pd.DataFrame,
    config: dict[str, Any] | None = None,
    beta: float = 0.95,
) -> pd.Series:
    """Find weights that minimize historical CVaR of portfolio losses.

    Args:
        returns: DataFrame of daily simple returns.
        config: Constraint config. If None, loaded from project settings.
        beta: Tail confidence level in (0, 1).

    Returns:
        Optimal weights as Series indexed by ticker.

    Raises:
        ValueError: If optimization fails or constraints are violated.
    """
    if config is None:
        config = load_optimizer_config()

    beta = float(beta)
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}.")

    tickers = [t for t in config["tickers"] if t in returns.columns]
    if len(tickers) < 2:
        raise ValueError(
            f"Need at least 2 assets to optimise, found {len(tickers)} matching tickers."
        )

    x = returns[tickers].to_numpy(dtype=float)
    n_obs, n_assets = x.shape
    if n_obs < 2:
        raise ValueError("Need at least 2 observations to estimate historical CVaR.")

    # Decision vector is [w_1..w_N, zeta, u_1..u_T].
    n_vars = n_assets + 1 + n_obs

    c = np.zeros(n_vars, dtype=float)
    c[n_assets] = 1.0
    c[n_assets + 1 :] = 1.0 / ((1.0 - beta) * n_obs)

    # Equality: sum(w) = 1.
    a_eq = np.zeros((1, n_vars), dtype=float)
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0], dtype=float)

    # Inequalities:
    # 1) u_t >= loss_t - zeta  -> loss_t - zeta - u_t <= 0
    #    with loss_t = -r_t'w.
    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []

    for t in range(n_obs):
        row = np.zeros(n_vars, dtype=float)
        row[:n_assets] = -x[t, :]   # loss coefficients for w
        row[n_assets] = -1.0        # zeta coefficient
        row[n_assets + 1 + t] = -1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

    # 2) Crypto cap: sum(w_crypto) <= max_crypto_weight.
    crypto_indices = [
        i for i, ticker in enumerate(tickers) if ticker in config.get("crypto_assets", [])
    ]
    if crypto_indices:
        row = np.zeros(n_vars, dtype=float)
        row[crypto_indices] = 1.0
        a_ub_rows.append(row)
        b_ub.append(float(config.get("max_crypto_weight", 1.0)))

    a_ub = np.vstack(a_ub_rows) if a_ub_rows else None
    b_ub_arr = np.array(b_ub, dtype=float) if b_ub else None

    max_w = float(config.get("max_weight", 1.0))
    lower = 0.0 if bool(config.get("long_only", True)) else -max_w
    bounds = [(lower, max_w)] * n_assets
    bounds.append((None, None))      # zeta
    bounds.extend([(0.0, None)] * n_obs)

    result = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub_arr,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise ValueError(f"CVaR optimisation did not converge: {result.message}")

    weights = pd.Series(result.x[:n_assets], index=tickers, name="weight")

    # Remove tiny numerical negatives and renormalize cautiously.
    weights = weights.where(weights > 1e-12, 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("CVaR optimisation produced non-positive total weight.")
    weights = weights / total

    violations = validate_weights(weights, config)
    if violations:
        raise ValueError("CVaR solution violates constraints: " + " | ".join(violations))

    return weights
