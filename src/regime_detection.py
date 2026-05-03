"""
Chapter 4 — Regime Detection
==============================
Fits market-regime models on the standardised feature panel produced by
``regime_features.py`` and outputs:

* ``regime_labels.csv``            — one row per trading day with the assigned
                                      regime label (integer and human name).
* ``regime_model_summary.csv``     — per-regime statistics (share, mean duration,
                                      median duration, mean feature values).
* ``regime_transition_matrix.csv`` — empirical day-over-day transition matrix.

Model hierarchy
---------------
Primary  : Gaussian HMM (hmmlearn), 2 states.
Fallback : KMeans, 2 states  — used automatically when hmmlearn is not installed.
Reported : All four candidate models (KMeans-2, KMeans-3, HMM-2, HMM-3) are fit
           and their quality metrics are logged; only the primary model's labels
           are written to disk.

State ordering
--------------
States are reordered after fitting using a scalar "stress score":
    stress_score(state) = sum_f(w_f * mean_f(state))
where the weights w_f come from ``detection.stress_score_features`` in the config.
State 0 is always the *least* stressed; state 1 (or 2 for k=3) is the most stressed.
This makes labels deterministic regardless of the random seed's arbitrary cluster IDs.

Standardisation
---------------
All features are z-scored (zero mean, unit variance) using the global mean/std of
the *full* feature panel before fitting.  The scaler is fit on the full series —
appropriate for a retrospective/diagnostic analysis.  No look-ahead issue arises
because the regime labels are used only for post-hoc attribution, not as trading
signals.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional HMM import — graceful fallback
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM  # type: ignore[import]
    _HMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GaussianHMM = None  # type: ignore[assignment,misc]
    _HMM_AVAILABLE = False
    logger.warning(
        "hmmlearn is not installed. HMM models will be unavailable; "
        "KMeans will be used as the primary model."
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Holds all outputs from a single model fit."""

    model_id: str                         # e.g. "hmm_2", "kmeans_3"
    n_states: int
    labels: pd.Series                     # int labels, DatetimeIndex
    state_names: dict[int, str]           # {0: "Low-stress / Risk-on", ...}
    transition_matrix: pd.DataFrame       # n_states × n_states
    model_summary: pd.DataFrame           # per-regime statistics
    log_likelihood: float | None = None   # HMM log-likelihood (None for KMeans)
    inertia: float | None = None          # KMeans inertia (None for HMM)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _standardise(features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Z-score all columns; return (scaled_array, fitted_scaler)."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    return X, scaler


def _compute_stress_scores(
    means_std: np.ndarray,
    feature_names: list[str],
    stress_weights: dict[str, float],
) -> np.ndarray:
    """Compute a scalar stress score for each state using weighted mean features.

    Parameters
    ----------
    means_std     : (n_states, n_features) array of per-state mean standardised values.
    feature_names : list of feature column names matching the second axis of means_std.
    stress_weights: dict mapping feature name → weight (+/-).

    Returns
    -------
    1-D array of length n_states with stress scores.
    """
    weights = np.array([stress_weights.get(f, 0.0) for f in feature_names])
    return means_std @ weights  # (n_states,)


def _reorder_states_by_stress(
    raw_labels: np.ndarray,
    stress_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return relabelled array where state 0 = least stressed, last = most stressed.

    Parameters
    ----------
    raw_labels    : 1-D integer array of state assignments (0-indexed).
    stress_scores : 1-D array with one stress score per state.

    Returns
    -------
    new_labels      : relabelled integer array.
    sorted_old_ids  : original state IDs sorted from low to high stress.
    """
    sorted_old_ids = np.argsort(stress_scores)       # ascending stress
    mapping = {old: new for new, old in enumerate(sorted_old_ids)}
    new_labels = np.vectorize(mapping.__getitem__)(raw_labels)
    return new_labels, sorted_old_ids


def _empirical_transition_matrix(labels: np.ndarray, n_states: int) -> np.ndarray:
    """Compute the empirical day-over-day transition matrix.

    T[i, j] = P(state_{t+1} = j | state_t = i)

    Returns
    -------
    (n_states, n_states) row-stochastic matrix.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(labels) - 1):
        counts[labels[t], labels[t + 1]] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for states that never appear as origin.
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return counts / row_sums


def _run_durations(labels: np.ndarray) -> dict[int, list[int]]:
    """Compute the length of every contiguous run for each state.

    Returns
    -------
    dict mapping state_id → list of run lengths.
    """
    if len(labels) == 0:
        return {}
    durations: dict[int, list[int]] = {}
    current = labels[0]
    run = 1
    for lbl in labels[1:]:
        if lbl == current:
            run += 1
        else:
            durations.setdefault(current, []).append(run)
            current = lbl
            run = 1
    durations.setdefault(current, []).append(run)
    return durations


def _build_model_summary(
    labels: np.ndarray,
    X_std: np.ndarray,
    feature_names: list[str],
    state_names: dict[int, str],
    n_states: int,
) -> pd.DataFrame:
    """Construct per-regime summary statistics.

    Returns
    -------
    DataFrame indexed by state_id with columns for share, durations and
    mean standardised feature values.
    """
    durations = _run_durations(labels)
    rows = []
    for s in range(n_states):
        mask = labels == s
        share = mask.sum() / len(labels)
        runs = durations.get(s, [])
        mean_dur = float(np.mean(runs)) if runs else 0.0
        median_dur = float(np.median(runs)) if runs else 0.0
        feature_means = X_std[mask].mean(axis=0) if mask.sum() > 0 else np.zeros(len(feature_names))
        row = {
            "state_id": s,
            "state_name": state_names.get(s, str(s)),
            "n_days": int(mask.sum()),
            "share_pct": round(share * 100, 2),
            "mean_duration_days": round(mean_dur, 1),
            "median_duration_days": round(median_dur, 1),
            "n_runs": len(runs),
        }
        for fname, fval in zip(feature_names, feature_means):
            row[f"mean_std_{fname}"] = round(float(fval), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("state_id")


# ---------------------------------------------------------------------------
# Model fitting functions
# ---------------------------------------------------------------------------

def _fit_kmeans(
    X: np.ndarray,
    n_states: int,
    random_state: int,
) -> tuple[np.ndarray, float]:
    """Fit KMeans and return (raw_labels, inertia)."""
    km = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
    raw_labels = km.fit_predict(X)
    return raw_labels.astype(int), float(km.inertia_)


def _fit_hmm(
    X: np.ndarray,
    n_states: int,
    random_state: int,
    covariance_type: str,
    n_iter: int,
    tol: float,
) -> tuple[np.ndarray, float]:
    """Fit a Gaussian HMM and return (raw_labels, log_likelihood).

    Raises RuntimeError if hmmlearn is not available.
    """
    if not _HMM_AVAILABLE:
        raise RuntimeError("hmmlearn is not installed; cannot fit HMM.")
    model = _GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress convergence warnings from hmmlearn
        model.fit(X)
    raw_labels = model.predict(X).astype(int)
    log_ll = float(model.score(X) * len(X))
    return raw_labels, log_ll


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_regimes(
    features: pd.DataFrame,
    cfg: dict[str, Any],
) -> RegimeResult:
    """Fit the primary regime model and return a ``RegimeResult``.

    Steps
    -----
    1. Z-score the full feature panel.
    2. Fit all four candidate models; log their quality metrics.
    3. Select the primary model (HMM-2 if available, else KMeans-2).
    4. Reorder states by stress score.
    5. Assign human-readable state names.
    6. Build outputs: labels, transition matrix, model summary.

    Parameters
    ----------
    features : DataFrame from ``regime_features.build_regime_features``.
    cfg      : Dictionary from ``config/regime_analysis.yaml``.

    Returns
    -------
    RegimeResult for the primary model.
    """
    det = cfg["detection"]
    random_state: int = det["random_state"]
    primary_model: str = det["primary_model"]
    n_states_default: int = det["n_states_default"]
    hmm_cfg = det["hmm"]
    stress_weights: dict[str, float] = det["stress_score_features"]

    feature_names = list(features.columns)
    X_std, _ = _standardise(features)
    dates = features.index

    # ── Fit all candidate models and log metrics ──────────────────────────
    logger.info("Fitting all candidate models …")
    for cand in det["candidate_models"]:
        m, k = cand["model"], cand["n_states"]
        try:
            if m == "kmeans":
                _, inertia = _fit_kmeans(X_std, k, random_state)
                logger.info("  KMeans k=%d — inertia=%.2f", k, inertia)
            elif m == "hmm":
                _, log_ll = _fit_hmm(
                    X_std, k, random_state,
                    hmm_cfg["covariance_type"], hmm_cfg["n_iter"], hmm_cfg["tol"],
                )
                logger.info("  HMM    k=%d — log-likelihood=%.2f", k, log_ll)
        except RuntimeError as exc:
            logger.warning("  %s k=%d — skipped: %s", m.upper(), k, exc)

    # ── Select and fit the primary model ─────────────────────────────────
    use_hmm = (primary_model == "hmm") and _HMM_AVAILABLE
    if (primary_model == "hmm") and not _HMM_AVAILABLE:
        logger.warning(
            "Primary model is 'hmm' but hmmlearn is not available. "
            "Falling back to KMeans with n_states=%d.",
            n_states_default,
        )

    if use_hmm:
        logger.info(
            "Primary model: HMM (Gaussian, full covariance), n_states=%d",
            n_states_default,
        )
        raw_labels, log_ll = _fit_hmm(
            X_std, n_states_default, random_state,
            hmm_cfg["covariance_type"], hmm_cfg["n_iter"], hmm_cfg["tol"],
        )
        model_id = f"hmm_{n_states_default}"
        inertia = None
    else:
        logger.info("Primary model: KMeans, n_states=%d", n_states_default)
        raw_labels, inertia = _fit_kmeans(X_std, n_states_default, random_state)
        model_id = f"kmeans_{n_states_default}"
        log_ll = None

    # ── Stress-score based state reordering ───────────────────────────────
    per_state_means = np.array([
        X_std[raw_labels == s].mean(axis=0) if (raw_labels == s).any()
        else np.zeros(len(feature_names))
        for s in range(n_states_default)
    ])
    stress_scores = _compute_stress_scores(per_state_means, feature_names, stress_weights)
    ordered_labels, sorted_old_ids = _reorder_states_by_stress(raw_labels, stress_scores)

    logger.info(
        "State reordering — original ids (by ascending stress): %s",
        sorted_old_ids.tolist(),
    )

    # ── Human-readable names ──────────────────────────────────────────────
    names_cfg: dict[int, str] = det["state_names"][n_states_default]
    state_names: dict[int, str] = {int(k): v for k, v in names_cfg.items()}

    # ── Build outputs ─────────────────────────────────────────────────────
    label_series = pd.Series(ordered_labels, index=dates, name="regime")

    trans_mat = _empirical_transition_matrix(ordered_labels, n_states_default)
    state_labels_str = [state_names[s] for s in range(n_states_default)]
    trans_df = pd.DataFrame(trans_mat, index=state_labels_str, columns=state_labels_str)
    trans_df.index.name = "from_state"

    summary_df = _build_model_summary(
        ordered_labels, X_std, feature_names, state_names, n_states_default,
    )

    # Log quality summary
    for s in range(n_states_default):
        name = state_names[s]
        n_days = int((ordered_labels == s).sum())
        pct = 100.0 * n_days / len(ordered_labels)
        logger.info("  State %d (%s): %d days (%.1f%%)", s, name, n_days, pct)

    logger.info("Regime detection complete — model: %s", model_id)

    return RegimeResult(
        model_id=model_id,
        n_states=n_states_default,
        labels=label_series,
        state_names=state_names,
        transition_matrix=trans_df,
        model_summary=summary_df,
        log_likelihood=log_ll,
        inertia=inertia,
    )


def build_labels_csv(result: RegimeResult) -> pd.DataFrame:
    """Construct the regime_labels DataFrame (date, regime_id, regime_name).

    Returns
    -------
    DataFrame with columns [regime_id, regime_name], indexed by date.
    """
    df = pd.DataFrame({
        "regime_id": result.labels.values,
        "regime_name": result.labels.map(result.state_names).values,
    }, index=result.labels.index)
    df.index.name = "date"
    return df


def save_results(result: RegimeResult, cfg: dict[str, Any], project_root: Path) -> dict[str, Path]:
    """Persist all three output CSVs.

    Returns
    -------
    Dict mapping output key → Path written.
    """
    out_cfg = cfg["detection"]["outputs"]
    output_dir = project_root / cfg["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # regime_labels.csv
    labels_path = project_root / out_cfg["regime_labels"]
    build_labels_csv(result).to_csv(labels_path)
    paths["regime_labels"] = labels_path
    logger.info("Saved regime_labels      → %s", labels_path)

    # regime_model_summary.csv
    summary_path = project_root / out_cfg["regime_model_summary"]
    result.model_summary.to_csv(summary_path)
    paths["regime_model_summary"] = summary_path
    logger.info("Saved regime_model_summary → %s", summary_path)

    # regime_transition_matrix.csv
    trans_path = project_root / out_cfg["regime_transition_matrix"]
    result.transition_matrix.to_csv(trans_path)
    paths["regime_transition_matrix"] = trans_path
    logger.info("Saved regime_transition_matrix → %s", trans_path)

    return paths


def load_and_detect(cfg: dict[str, Any], project_root: Path) -> RegimeResult:
    """Load the feature panel from disk and run regime detection.

    Parameters
    ----------
    cfg          : Loaded from ``config/regime_analysis.yaml``.
    project_root : Absolute project root path.

    Returns
    -------
    RegimeResult for the primary model.
    """
    features_path = project_root / cfg["paths"]["regime_features"]
    logger.info("Loading regime features from %s", features_path)
    features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    features.index = pd.to_datetime(features.index)
    features.sort_index(inplace=True)
    return detect_regimes(features, cfg)
