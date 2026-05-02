"""Minimum variance portfolio optimiser (MVP — first implementation).

This module finds the portfolio weights that minimise portfolio variance
subject to a set of practical constraints.  It uses scipy.optimize, which
is well-documented and beginner-friendly.

NOTE: The covariance estimator used here is the *sample* covariance matrix.
Sample covariance is the simplest possible choice and works well when the
number of observations is large relative to the number of assets (here ~3000
observations, 6 assets).  More robust alternatives (Ledoit-Wolf shrinkage,
exponentially-weighted covariance) can be swapped in later without changing
the rest of the code — see `estimate_covariance()` below.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from src.config import load_assets, load_settings


# ---------------------------------------------------------------------------
# Constraint / config helpers
# ---------------------------------------------------------------------------

def load_optimizer_config() -> dict[str, Any]:
    """Load the constraints and asset list needed by the optimiser.

    Reads from config/assets.yaml and config/settings.yaml.

    Returns:
        Dictionary with keys:
        - tickers          : list[str]  — ordered asset universe
        - crypto_assets    : list[str]  — tickers treated as crypto
        - long_only        : bool
        - max_weight       : float      — upper bound per asset
        - max_crypto_weight: float      — upper bound for total crypto exposure
    """
    assets_cfg = load_assets()
    settings_cfg = load_settings()
    constraints_cfg = settings_cfg.get("portfolio_constraints", {})

    return {
        "tickers": assets_cfg["all_tickers"],
        "crypto_assets": assets_cfg.get("crypto_assets", []),
        "long_only": bool(constraints_cfg.get("long_only", True)),
        "max_weight": float(constraints_cfg.get("max_weight_per_asset", 1.0)),
        "max_crypto_weight": float(constraints_cfg.get("max_total_crypto_weight", 1.0)),
    }


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def estimate_covariance(
    returns: pd.DataFrame,
    covariance_method: str = "sample",
) -> pd.DataFrame:
    """Estimate the annualised covariance matrix from daily returns.

    This is the first and simplest covariance estimator.  It divides by
    (n - 1) (the pandas / numpy default) to get an unbiased estimate, then
    scales by 252 to annualise.

    Future improvements to consider (do NOT change the call signature):
    - Ledoit-Wolf shrinkage  : sklearn.covariance.LedoitWolf
    - EWMA covariance        : weight recent observations more heavily
    - Robust covariance      : reduce sensitivity to outliers

    Supported methods:
    - "sample": unbiased sample covariance (ddof=1)
    - "ledoit_wolf": shrinkage covariance via sklearn Ledoit-Wolf

    Args:
        returns: DataFrame of daily simple returns, shape (T, N).
        covariance_method: Covariance estimator to use.

    Returns:
        Annualised covariance matrix as a DataFrame, shape (N, N).
    """
    method = str(covariance_method).lower().strip()

    if method == "sample":
        cov_daily = returns.cov()      # sample covariance, unbiased (ddof=1)
    elif method == "ledoit_wolf":
        try:
            from sklearn.covariance import LedoitWolf
        except ImportError as exc:
            raise ImportError(
                "covariance_method='ledoit_wolf' requires scikit-learn. "
                "Install it with 'pip install scikit-learn'."
            ) from exc
        lw = LedoitWolf()
        lw.fit(returns.values)
        cov_daily = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns,
        )
    else:
        raise ValueError(
            "Unsupported covariance_method: "
            f"{covariance_method}. Use 'sample' or 'ledoit_wolf'."
        )

    return cov_daily * 252             # annualise: multiply by trading days/year


# ---------------------------------------------------------------------------
# Portfolio variance (objective function)
# ---------------------------------------------------------------------------

def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Compute portfolio variance for a given weight vector.

    This is the objective function that scipy.optimize will minimise.

    Args:
        weights   : 1-D NumPy array of portfolio weights, shape (N,).
        cov_matrix: 2-D NumPy array (covariance matrix), shape (N, N).

    Returns:
        Portfolio variance as a scalar float.
    """
    # w' * Sigma * w  — standard quadratic form for portfolio variance.
    return float(weights @ cov_matrix @ weights)


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------

def validate_weights(
    weights: pd.Series,
    config: dict[str, Any],
    tolerance: float = 1e-4,
) -> list[str]:
    """Check that a weight vector satisfies the portfolio constraints.

    This function is intentionally non-raising — it returns a list of
    human-readable messages describing any violations.  The caller decides
    whether to raise an error or just warn.

    Args:
        weights   : Series of asset weights indexed by ticker.
        config    : Config dict as returned by `load_optimizer_config()`.
        tolerance : Numerical tolerance for floating-point comparisons.

    Returns:
        List of violation strings.  Empty list means all constraints pass.
    """
    violations: list[str] = []

    # 1. Weights must sum to 1.
    total = weights.sum()
    if not np.isclose(total, 1.0, atol=tolerance):
        violations.append(f"Weights sum to {total:.6f}, expected 1.0.")

    # 2. Long-only (no negative weights).
    if config.get("long_only", True):
        negative = weights[weights < -tolerance]
        if not negative.empty:
            violations.append(
                "Negative weights (long-only violated): "
                + ", ".join(f"{t}={w:.4f}" for t, w in negative.items())
            )

    # 3. Per-asset maximum weight.
    max_w = config.get("max_weight", 1.0)
    over_max = weights[weights > max_w + tolerance]
    if not over_max.empty:
        violations.append(
            f"Assets exceed max_weight ({max_w:.2f}): "
            + ", ".join(f"{t}={w:.4f}" for t, w in over_max.items())
        )

    # 4. Maximum total crypto weight.
    max_crypto = config.get("max_crypto_weight", 1.0)
    crypto_tickers = [t for t in config.get("crypto_assets", []) if t in weights.index]
    if crypto_tickers:
        total_crypto = weights[crypto_tickers].sum()
        if total_crypto > max_crypto + tolerance:
            violations.append(
                f"Total crypto weight {total_crypto:.4f} exceeds limit {max_crypto:.2f}."
            )

    return violations


# ---------------------------------------------------------------------------
# Core optimiser
# ---------------------------------------------------------------------------

def minimise_variance(
    returns: pd.DataFrame,
    config: dict[str, Any] | None = None,
    covariance_method: str = "sample",
) -> pd.Series:
    """Find the minimum variance portfolio weights.

    Minimises portfolio variance (w' Σ w) subject to:
    - weights sum to 1
    - long-only (weights >= 0)
    - max weight per asset  (from config/settings.yaml)
    - max total crypto weight (from config/settings.yaml)

    Args:
        returns: DataFrame of daily simple returns.  Columns are tickers.
        config : Optional pre-loaded config dict.  If None, loaded from YAML.
        covariance_method: Covariance estimator to use ("sample" or
            "ledoit_wolf").

    Returns:
        Series of optimal weights indexed by ticker.

    Raises:
        ValueError: If the optimiser fails to converge or weights are invalid.
    """
    if config is None:
        config = load_optimizer_config()

    # --- 1. Align returns to the expected asset order -------------------------
    # Only keep tickers that appear in both the config and the returns columns.
    tickers = [t for t in config["tickers"] if t in returns.columns]
    if len(tickers) < 2:
        raise ValueError(
            f"Need at least 2 assets to optimise, found {len(tickers)} "
            f"matching between config and returns columns."
        )
    returns_aligned = returns[tickers]

    # --- 2. Covariance matrix -------------------------------------------------
    cov_df = estimate_covariance(returns_aligned, covariance_method=covariance_method)
    cov_matrix = cov_df.values  # plain NumPy array for scipy

    n = len(tickers)

    # --- 3. Build constraints for scipy.optimize.minimize --------------------
    #
    # scipy uses a list of dicts.  Each dict has:
    #   "type": "eq" (equality) or "ineq" (inequality, must be >= 0)
    #   "fun" : callable returning the constraint value

    constraints: list[dict] = [
        # Weights must sum to 1.
        {
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
        },
    ]

    # Max total crypto weight constraint.
    # Expressed as inequality: max_crypto - sum(crypto_weights) >= 0
    crypto_indices = [
        i for i, t in enumerate(tickers) if t in config.get("crypto_assets", [])
    ]
    if crypto_indices:
        max_crypto = config["max_crypto_weight"]
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idx=crypto_indices, limit=max_crypto: (
                    limit - sum(w[i] for i in idx)
                ),
            }
        )

    # --- 4. Bounds (per-asset) -----------------------------------------------
    max_w = config["max_weight"]
    lower = 0.0 if config.get("long_only", True) else -max_w
    bounds = [(lower, max_w)] * n

    # --- 5. Initial guess: equal weights -------------------------------------
    w0 = np.full(n, 1.0 / n)

    # --- 6. Run optimisation --------------------------------------------------
    result: OptimizeResult = minimize(
        fun=portfolio_variance,
        x0=w0,
        args=(cov_matrix,),
        method="SLSQP",          # Sequential Least Squares Programming
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise ValueError(
            f"Optimisation did not converge: {result.message}\n"
            "Consider checking your constraints or input data."
        )

    # --- 7. Wrap result as a labelled Series ----------------------------------
    optimal_weights = pd.Series(result.x, index=tickers, name="weight")

    # Small floating-point residuals can produce tiny negatives like -1e-16.
    # Clip to zero and renormalise so the output is clean.
    optimal_weights = optimal_weights.clip(lower=0.0)
    optimal_weights = optimal_weights / optimal_weights.sum()

    # --- 8. Validate the result -----------------------------------------------
    violations = validate_weights(optimal_weights, config)
    if violations:
        # This should not happen with a converged solution, but warn loudly.
        warnings.warn(
            "Optimal weights failed constraint check:\n" + "\n".join(violations),
            UserWarning,
            stacklevel=2,
        )

    return optimal_weights
