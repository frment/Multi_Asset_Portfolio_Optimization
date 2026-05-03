from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chapter5_runtime import limit_sequence, log_event, parse_debug_args, timed_phase  # noqa: E402
from src.config import load_yaml  # noqa: E402
from src.model_evaluation import evaluate_all_models  # noqa: E402
from src.supervised_models import fit_predict_walk_forward  # noqa: E402
from src.supervised_validation import align_features_targets, generate_walk_forward_splits  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _task_for_target(target_col: str) -> str:
    return "classification" if "event" in target_col else "regression"


def _models_for_target(cfg: dict, target_col: str) -> list[str]:
    models_cfg = cfg.get("models", {})
    if "event" not in target_col:
        return [str(m) for m in models_cfg.get("volatility", [])]
    if "stress" in target_col:
        return [str(m) for m in models_cfg.get("stress_event", [])]
    return [str(m) for m in models_cfg.get("drawdown_event", [])]


def _best_model(scores: pd.DataFrame, target_col: str, task: str) -> tuple[str, bool]:
    required_cols = {"target_col", "task", "model_name"}
    if scores.empty or not required_cols.issubset(set(scores.columns)):
        return "", False

    subset = scores[(scores["target_col"] == target_col) & (scores["task"] == task)].copy()
    if "status" in subset.columns:
        subset = subset[subset["status"] == "ok"]
    if subset.empty:
        return "", False

    naive_name = "naive_rolling_vol" if task == "regression" else "logistic"
    naive = subset[subset["model_name"] == naive_name]

    if task == "regression":
        if "rmse" not in subset.columns:
            return "", False
        best_row = subset.sort_values("rmse", ascending=True).iloc[0]
        naive_rmse = float(naive["rmse"].iloc[0]) if not naive.empty else float("inf")
        improved = float(best_row["rmse"]) < naive_rmse
    else:
        if "roc_auc" not in subset.columns:
            return "", False
        best_row = subset.sort_values("roc_auc", ascending=False).iloc[0]
        naive_auc = float(naive["roc_auc"].iloc[0]) if not naive.empty else 0.0
        improved = float(best_row["roc_auc"]) > naive_auc

    return str(best_row["model_name"]), bool(improved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 5 supervised model training and evaluation")
    parser.add_argument("--config", default="supervised_risk_overlay.yaml", help="Config filename under config/")
    parse_debug_args(parser)
    args = parser.parse_args()

    debug: bool = bool(args.debug)
    max_targets: int | None = args.max_targets
    max_models: int | None = args.max_models
    max_splits: int | None = args.max_splits

    # Debug mode uses a separate output dir so it never overwrites official outputs.
    output_suffix = "_debug" if debug else ""

    cfg = load_yaml(args.config)
    features_path = PROJECT_ROOT / "data/processed/supervised_features.csv"
    targets_path = PROJECT_ROOT / "data/processed/supervised_targets.csv"
    base_output_dir = PROJECT_ROOT / cfg.get("reporting", {}).get("output_dir", "outputs/chapter5")
    output_dir = base_output_dir.parent / (base_output_dir.name + output_suffix)
    ensure_directory(output_dir)

    with timed_phase("load_inputs"):
        features = pd.read_csv(features_path, index_col=0, parse_dates=True)
        targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
        log_event("SHAPE", name="features_input", rows=features.shape[0], cols=features.shape[1])
        log_event("SHAPE", name="targets_input", rows=targets.shape[0], cols=targets.shape[1])

    vcfg = cfg.get("validation", {})
    random_state = int(vcfg.get("random_state", 42))
    diagnostics_rows: list[dict] = []

    # In debug mode pick 1 vol target + 1 classification target; then apply max_targets cap.
    all_target_cols = list(targets.columns)
    if debug:
        vol_targets = [c for c in all_target_cols if "event" not in c]
        cls_targets = [c for c in all_target_cols if "event" in c]
        preferred = []
        for candidates, fallback_name in [
            (vol_targets, "target_portfolio_vol_21d"),
            (cls_targets, "target_stress_event_21d"),
        ]:
            found = next((c for c in candidates if c == fallback_name), None) or (candidates[0] if candidates else None)
            if found:
                preferred.append(found)
        all_target_cols = preferred
        log_event("DEBUG", message="debug_target_selection", selected=all_target_cols)

    selected_target_cols = limit_sequence(all_target_cols, max_targets)
    log_event("TARGETS", selected=selected_target_cols, n=len(selected_target_cols))

    with timed_phase("model_training"):
      for target_col in selected_target_cols:
        task = _task_for_target(target_col)
        all_model_names = _models_for_target(cfg, target_col)

        # In debug mode restrict to cheap models only.
        if debug:
            cheap = {"naive_rolling_vol", "ridge"} if task == "regression" else {"logistic"}
            all_model_names = [m for m in all_model_names if m in cheap]
            log_event("DEBUG", message="debug_model_selection", target=target_col, active_models=all_model_names)

        model_names = limit_sequence(all_model_names, max_models)

        X, y = align_features_targets(features, targets, target_col)
        log_event("SHAPE", name=f"aligned_X:{target_col}", rows=X.shape[0], cols=X.shape[1])
        if X.empty:
            for model_name in model_names:
                diagnostics_rows.append(
                    {
                        "target_col": target_col,
                        "task": task,
                        "model_name": model_name,
                        "n_samples": 0,
                        "n_splits": 0,
                        "n_predictions": 0,
                        "status": "skipped_no_aligned_samples",
                    }
                )
            continue

        splits = generate_walk_forward_splits(
            index=X.index,
            train_window=int(vcfg.get("train_window", 756)),
            test_window=int(vcfg.get("test_window", 63)),
            step_size=int(vcfg.get("step_size", 21)),
            min_train_size=int(vcfg.get("min_train_size", 504)),
            embargo_days=int(vcfg.get("embargo_days", 21)),
        )
        log_event("SPLITS", target=target_col, n_splits_total=len(splits), max_splits=max_splits)

        if not splits:
            for model_name in model_names:
                diagnostics_rows.append(
                    {
                        "target_col": target_col,
                        "task": task,
                        "model_name": model_name,
                        "n_samples": int(len(X)),
                        "n_splits": 0,
                        "n_predictions": 0,
                        "status": "skipped_no_walkforward_splits",
                    }
                )
            continue

        for model_name in model_names:
            pred = fit_predict_walk_forward(
                X=X,
                y=y,
                model_name=model_name,
                task=task,
                splits=splits,
                random_state=random_state,
                max_splits=max_splits,
            )
            log_event("SHAPE", name=f"predictions:{target_col}:{model_name}", rows=len(pred))
            if pred.empty:
                diagnostics_rows.append(
                    {
                        "target_col": target_col,
                        "task": task,
                        "model_name": model_name,
                        "n_samples": int(len(X)),
                        "n_splits": int(len(splits)),
                        "n_predictions": 0,
                        "status": "skipped_no_predictions",
                    }
                )
                continue

            safe_target = target_col.replace("/", "_")
            out_file = output_dir / f"predictions_{safe_target}_{model_name}.csv"
            pred.to_csv(out_file, index=False)
            diagnostics_rows.append(
                {
                    "target_col": target_col,
                    "task": task,
                    "model_name": model_name,
                    "n_samples": int(len(X)),
                    "n_splits": int(len(splits)),
                    "n_predictions": int(len(pred)),
                    "status": "ok",
                }
            )

    with timed_phase("evaluate_models"):
        scores, calibration, buckets = evaluate_all_models(output_dir)
    diagnostics = pd.DataFrame(diagnostics_rows)

    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=["target_col", "task", "model_name", "n_samples", "n_splits", "n_predictions", "status"])

    merged_scores = diagnostics.merge(
        scores,
        on=["target_col", "task", "model_name"],
        how="left",
    )

    # Ensure metric columns exist even when models are skipped.
    for col in ["mae", "rmse", "r2_oos", "corr_pred_actual", "directional_accuracy", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier", "tn", "fp", "fn", "tp"]:
        if col not in merged_scores.columns:
            merged_scores[col] = np.nan

    scores_path = output_dir / "model_scores.csv"
    merged_scores.to_csv(scores_path, index=False)

    diagnostics_path = output_dir / "model_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    calib_path = output_dir / "calibration_tables.csv"
    calibration.to_csv(calib_path, index=False)

    bucket_path = output_dir / "forecast_bucket_analysis.csv"
    buckets.to_csv(bucket_path, index=False)

    selection_rows: list[dict] = []
    for target_col in targets.columns:
        task = _task_for_target(target_col)
        best_name, improved = _best_model(merged_scores, target_col, task)
        fallback = "naive_rolling_vol" if task == "regression" else "logistic"
        selected = best_name if improved and best_name else fallback
        selection_rows.append(
            {
                "target_col": target_col,
                "task": task,
                "best_model": best_name,
                "selected_model": selected,
                "improved_vs_naive": improved,
            }
        )

    selection = pd.DataFrame(selection_rows)
    selection_path = output_dir / "model_selection.csv"
    selection.to_csv(selection_path, index=False)

    print("Chapter 5 - supervised models")
    print(f"  debug_mode            : {debug}")
    print(f"  output_dir            : {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"  validation_method     : {vcfg.get('method', 'walk_forward')}")
    print(f"  validation_settings   : {vcfg}")
    print(f"  model_list            : {cfg.get('models', {})}")
    print(f"  model_scores_path     : {scores_path.relative_to(PROJECT_ROOT)}")
    print(f"  model_diagnostics     : {diagnostics_path.relative_to(PROJECT_ROOT)}")
    print(f"  calibration_path      : {calib_path.relative_to(PROJECT_ROOT)}")
    print(f"  bucket_analysis_path  : {bucket_path.relative_to(PROJECT_ROOT)}")
    log_event("SHAPE", name="model_scores_output", rows=merged_scores.shape[0], cols=merged_scores.shape[1])


if __name__ == "__main__":
    main()
