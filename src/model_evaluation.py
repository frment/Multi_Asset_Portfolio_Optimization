from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    y_true = pred_df["y_true"].astype(float)
    y_pred = pred_df["y_pred"].astype(float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(pred_df) > 1 else np.nan
    dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2_oos": float(r2_score(y_true, y_pred)),
        "corr_pred_actual": corr,
        "directional_accuracy": dir_acc,
    }


def evaluate_classification_predictions(
    pred_df: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = pred_df["y_true"].astype(int)
    score_col = "y_prob" if pred_df["y_prob"].notna().any() else "y_pred"
    y_score = pred_df[score_col].astype(float)
    y_hat = (y_score >= float(threshold)).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.nunique() > 1 else np.nan,
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score.clip(0.0, 1.0))),
    }

    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    metrics["tn"] = float(cm[0, 0])
    metrics["fp"] = float(cm[0, 1])
    metrics["fn"] = float(cm[1, 0])
    metrics["tp"] = float(cm[1, 1])

    calib = pred_df[["y_true"]].copy()
    calib["score"] = y_score
    calib = calib.dropna()
    if len(calib) >= 10:
        calib["decile"] = pd.qcut(calib["score"], q=10, labels=False, duplicates="drop")
        calib_tbl = (
            calib.groupby("decile", observed=True)
            .agg(mean_score=("score", "mean"), event_rate=("y_true", "mean"), n=("y_true", "size"))
            .reset_index()
        )
    else:
        calib_tbl = pd.DataFrame(columns=["decile", "mean_score", "event_rate", "n"])

    return metrics, calib_tbl


def _bucket_analysis(pred_df: pd.DataFrame) -> pd.DataFrame:
    frame = pred_df[["y_true", "y_pred"]].dropna().copy()
    if len(frame) < 10:
        return pd.DataFrame(columns=["bucket", "mean_pred", "mean_realized", "n"])

    frame["bucket"] = pd.qcut(frame["y_pred"], q=5, labels=False, duplicates="drop")
    return (
        frame.groupby("bucket", observed=True)
        .agg(mean_pred=("y_pred", "mean"), mean_realized=("y_true", "mean"), n=("y_true", "size"))
        .reset_index()
    )


def evaluate_all_models(predictions_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_dir = Path(predictions_dir)
    files = sorted(pred_dir.glob("predictions_*.csv"))

    scores_rows: list[dict] = []
    calib_rows: list[pd.DataFrame] = []
    bucket_rows: list[pd.DataFrame] = []

    for file in files:
        pred = pd.read_csv(file, parse_dates=["date"])
        if pred.empty:
            continue

        model_name = str(pred["model_name"].iloc[0])
        task = str(pred["task"].iloc[0])
        target_col = str(pred["target_col"].iloc[0])

        base = {"target_col": target_col, "model_name": model_name, "task": task}

        if task == "regression":
            metrics = evaluate_regression_predictions(pred)
            scores_rows.append({**base, **metrics})

            buckets = _bucket_analysis(pred)
            if not buckets.empty:
                buckets.insert(0, "target_col", target_col)
                buckets.insert(1, "model_name", model_name)
                bucket_rows.append(buckets)
        else:
            metrics, calib = evaluate_classification_predictions(pred)
            scores_rows.append({**base, **metrics})

            if not calib.empty:
                calib.insert(0, "target_col", target_col)
                calib.insert(1, "model_name", model_name)
                calib_rows.append(calib)

    scores = pd.DataFrame(scores_rows)
    calibration = pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame()
    buckets = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()

    return scores, calibration, buckets
