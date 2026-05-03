from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.chapter5_runtime import log_event


def get_model(model_name: str, task: str, random_state: int) -> Any:
    """Return configured model object for regression or classification."""
    name = str(model_name).lower().strip()
    tsk = str(task).lower().strip()

    if tsk == "regression":
        if name in {"naive_rolling_vol", "ewma"}:
            return name
        if name == "ridge":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            )
        if name == "elastic_net":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ElasticNet(
                            alpha=0.01,
                            l1_ratio=0.2,
                            max_iter=5000,
                            tol=1e-3,
                            random_state=int(random_state),
                        ),
                    ),
                ]
            )
        if name == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=20,
                random_state=int(random_state),
                n_jobs=1,
            )

    if tsk == "classification":
        if name == "logistic":
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=int(random_state),
                        ),
                    ),
                ]
            )
        if name == "calibrated_logistic":
            base = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="lbfgs",
                            max_iter=2000,
                            random_state=int(random_state),
                        ),
                    ),
                ]
            )
            return CalibratedClassifierCV(base, cv=3, method="sigmoid")
        if name == "random_forest_classifier":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=20,
                random_state=int(random_state),
                n_jobs=1,
            )
        if name == "gradient_boosting_classifier":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=int(random_state),
            )

    raise ValueError(f"Unsupported model/task combination: model={model_name}, task={task}")


def _pick_naive_feature(X: pd.DataFrame, target_col: str) -> str:
    target = str(target_col).lower()
    candidates = [
        "portfolio_vol_21d",
        "portfolio_vol_63d",
        "spy_vol_21d",
        "btc_vol_21d",
    ]
    if "63" in target and "portfolio_vol_63d" in X.columns:
        return "portfolio_vol_63d"
    for c in candidates:
        if c in X.columns:
            return c
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature available for naive_rolling_vol model")
    return numeric_cols[0]


def _prepare_X(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if out[col].dtype == "object" or str(out[col].dtype).startswith("category"):
            out[col] = pd.factorize(out[col], sort=True)[0].astype(float)
    return out.astype(float)


def fit_predict_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    task: str,
    splits: list[dict[str, Any]],
    random_state: int = 42,
    max_splits: int | None = None,
) -> pd.DataFrame:
    """Fit model over walk-forward splits and return OOS predictions."""
    from time import perf_counter

    X_num = _prepare_X(X)
    y_series = y.astype(float)

    records: list[dict[str, Any]] = []
    target_col = str(y.name or "target")
    name = str(model_name).lower().strip()
    tsk = str(task).lower().strip()

    active_splits = splits[: int(max_splits)] if max_splits is not None and max_splits > 0 else splits

    for split in active_splits:
        split_id = int(split.get("split_id", 0))
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]

        X_train = X_num.iloc[train_idx]
        y_train = y_series.iloc[train_idx]
        X_test = X_num.iloc[test_idx]
        y_test = y_series.iloc[test_idx]

        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians).fillna(0.0)
        X_test = X_test.fillna(train_medians).fillna(0.0)

        if len(X_train) == 0 or len(X_test) == 0:
            log_event(
                "MODEL_SKIP",
                target=target_col,
                model=name,
                split=split_id,
                reason="empty_train_or_test",
            )
            continue

        # class counts for classification
        if tsk == "classification":
            counts_str = ",".join(
                f"{int(k)}:{int(v)}"
                for k, v in y_train.value_counts(dropna=False).sort_index().items()
            )
        else:
            counts_str = "na"

        log_event(
            "MODEL_START",
            target=target_col,
            model=name,
            split=split_id,
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=X_train.shape[1],
            y_train_class_counts=counts_str,
        )
        t0 = perf_counter()

        if tsk == "regression" and name == "naive_rolling_vol":
            proxy = _pick_naive_feature(X_train, target_col)
            y_pred = X_test[proxy].astype(float).values
            y_prob = np.full_like(y_pred, np.nan, dtype=float)
        elif tsk == "regression" and name == "ewma":
            proxy = _pick_naive_feature(X_train, target_col)
            y_pred = X_test[proxy].ewm(alpha=0.2, adjust=False).mean().values
            y_prob = np.full_like(y_pred, np.nan, dtype=float)
        else:
            if tsk == "classification" and y_train.nunique() < 2:
                log_event(
                    "MODEL_SKIP",
                    target=target_col,
                    model=name,
                    split=split_id,
                    reason="single_class_train",
                )
                continue
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", category=ConvergenceWarning)
                model = get_model(name, tsk, random_state=random_state)
                model.fit(X_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                log_event(
                    "MODEL_WARNING",
                    target=target_col,
                    model=name,
                    split=split_id,
                    warning="convergence_not_reached",
                )
            y_pred = model.predict(X_test)
            if tsk == "classification" and hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = np.full(len(y_pred), np.nan, dtype=float)

        elapsed = perf_counter() - t0
        log_event(
            "MODEL_END",
            target=target_col,
            model=name,
            split=split_id,
            duration_seconds=f"{elapsed:.3f}",
            n_predictions=len(y_pred),
        )
        if elapsed > 60.0:
            log_event(
                "MODEL_SLOW",
                target=target_col,
                model=name,
                split=split_id,
                duration_seconds=f"{elapsed:.3f}",
            )

        for dt, yt, yp, yp_prob in zip(X_test.index, y_test.values, y_pred, y_prob):
            records.append(
                {
                    "date": dt,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                    "y_prob": float(yp_prob) if not np.isnan(yp_prob) else np.nan,
                    "split_id": split_id,
                    "model_name": name,
                    "task": tsk,
                    "target_col": target_col,
                }
            )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    out = out.sort_values("date").reset_index(drop=True)
    return out
