from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datetime import datetime, timezone

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_yaml  # noqa: E402
from src.overlay_backtest import run_overlay_backtest  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_baseline_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if "strategy" in df.columns:
        subset = df[df["strategy"] == "min_variance"].copy()
        if not subset.empty:
            subset = subset.drop(columns=[c for c in ["strategy"] if c in subset.columns])
            return subset
    return df


def _find_prediction_file(output_dir: Path, target: str, model: str) -> Path | None:
    f = output_dir / f"predictions_{target}_{model}.csv"
    if f.exists():
        return f
    matches = list(output_dir.glob(f"predictions_{target}_*.csv"))
    return matches[0] if matches else None


def _infer_model_name_from_prediction_path(path: Path, target: str) -> str:
    prefix = f"predictions_{target}_"
    stem = path.stem
    return stem.replace(prefix, "", 1) if stem.startswith(prefix) else stem


def _build_forecasts(output_dir: Path, selection: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    forecasts = pd.DataFrame()
    meta: dict[str, Any] = {}

    vol_target = "target_portfolio_vol_21d"
    stress_target = "target_stress_event_21d"

    if not selection.empty:
        vol_row = selection[selection["target_col"] == vol_target]
        stress_row = selection[selection["target_col"] == stress_target]

        vol_model = str(vol_row["selected_model"].iloc[0]) if not vol_row.empty else "naive_rolling_vol"
        stress_model = str(stress_row["selected_model"].iloc[0]) if not stress_row.empty else "logistic"
    else:
        vol_model = "naive_rolling_vol"
        stress_model = "logistic"

    vol_file = _find_prediction_file(output_dir, vol_target, vol_model)
    if vol_file is not None:
        vol = pd.read_csv(vol_file, parse_dates=["date"]).set_index("date")
        forecasts["forecast_vol"] = vol["y_pred"].astype(float).groupby(level=0).last()
        meta["vol_forecast_file"] = str(vol_file)
        meta["vol_model_requested"] = vol_model
        meta["vol_model_used"] = _infer_model_name_from_prediction_path(vol_file, vol_target)
        meta["vol_fallback_used"] = bool(meta["vol_model_used"] != vol_model)
    else:
        meta["vol_forecast_file"] = None
        meta["vol_model_requested"] = vol_model
        meta["vol_model_used"] = None
        meta["vol_fallback_used"] = True

    stress_file = _find_prediction_file(output_dir, stress_target, stress_model)
    if stress_file is not None:
        stress = pd.read_csv(stress_file, parse_dates=["date"]).set_index("date")
        if stress["y_prob"].notna().any():
            forecasts["crash_probability"] = stress["y_prob"].astype(float).groupby(level=0).last()
        else:
            forecasts["crash_probability"] = stress["y_pred"].astype(float).groupby(level=0).last()
        meta["stress_forecast_file"] = str(stress_file)
        meta["stress_model_requested"] = stress_model
        meta["stress_model_used"] = _infer_model_name_from_prediction_path(stress_file, stress_target)
        meta["stress_fallback_used"] = bool(meta["stress_model_used"] != stress_model)
    else:
        meta["stress_forecast_file"] = None
        meta["stress_model_requested"] = stress_model
        meta["stress_model_used"] = None
        meta["stress_fallback_used"] = True

    forecasts = forecasts.sort_index()
    if not forecasts.empty:
        forecasts = forecasts.groupby(level=0).last()
    return forecasts, meta


def _load_regime(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    reg = pd.read_csv(path, index_col=0, parse_dates=True)
    if "regime_high_stress_dummy" not in reg.columns:
        if "regime_label" in reg.columns:
            lbl = reg["regime_label"]
            if pd.api.types.is_numeric_dtype(lbl):
                reg["regime_high_stress_dummy"] = (lbl == lbl.max()).astype(float)
            else:
                reg["regime_high_stress_dummy"] = lbl.astype(str).str.contains("high|stress", case=False).astype(float)
    return reg


def _strategy_config(base_cfg: dict[str, Any], strategy: str) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))
    ov = cfg.setdefault("overlay", {})

    if strategy == "minvar_baseline_corrected":
        ov["dynamic_crypto_cap"]["enabled"] = False
        ov["de_risking"]["enabled"] = False
    elif strategy == "minvar_no_crypto_control":
        ov["dynamic_crypto_cap"]["enabled"] = True
        ov["dynamic_crypto_cap"]["normal_cap"] = 0.0
        ov["dynamic_crypto_cap"]["medium_risk_cap"] = 0.0
        ov["dynamic_crypto_cap"]["high_risk_cap"] = 0.0
        ov["de_risking"]["enabled"] = False
    elif strategy == "minvar_dynamic_crypto_cap_naive_vol":
        ov["dynamic_crypto_cap"]["enabled"] = True
        ov["de_risking"]["enabled"] = False
    elif strategy == "minvar_dynamic_crypto_cap_ml_vol":
        ov["dynamic_crypto_cap"]["enabled"] = True
        ov["de_risking"]["enabled"] = False
    elif strategy == "minvar_dynamic_crypto_cap_stress_probability":
        ov["dynamic_crypto_cap"]["enabled"] = True
        ov["de_risking"]["enabled"] = True
    elif strategy == "minvar_combined_overlay":
        ov["dynamic_crypto_cap"]["enabled"] = True
        ov["de_risking"]["enabled"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 5 risk overlay backtests")
    parser.add_argument("--config", default="supervised_risk_overlay.yaml", help="Config filename under config/")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})

    returns = pd.read_csv(
        PROJECT_ROOT / data_cfg.get("returns_path", "data/processed/returns_simple.csv"),
        index_col=0,
        parse_dates=True,
    )
    base_weights = _load_baseline_weights(
        PROJECT_ROOT / data_cfg.get("baseline_weights_path", "data/processed/weights_history.csv")
    )

    output_dir = PROJECT_ROOT / cfg.get("reporting", {}).get("output_dir", "outputs/chapter5")
    ensure_directory(output_dir)

    selection_path = output_dir / "model_selection.csv"
    selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()

    forecasts, forecast_meta = _build_forecasts(output_dir, selection)
    if not forecasts.empty:
        forecasts = forecasts.reindex(returns.index).ffill()

    regime_data = _load_regime(
        PROJECT_ROOT / data_cfg.get("regime_labels_path", "data/processed/regime_analysis/regime_labels.csv")
    )
    if regime_data is not None:
        regime_data = regime_data.reindex(returns.index).ffill()

    strategies = [
        "minvar_baseline_corrected",
        "minvar_no_crypto_control",
        "minvar_dynamic_crypto_cap_naive_vol",
        "minvar_dynamic_crypto_cap_ml_vol",
        "minvar_dynamic_crypto_cap_stress_probability",
        "minvar_combined_overlay",
    ]

    summary_rows: list[pd.DataFrame] = []
    daily_rows: list[pd.DataFrame] = []
    weights_rows: list[pd.DataFrame] = []
    decisions_rows: list[pd.DataFrame] = []
    turnover_rows: list[pd.DataFrame] = []

    for strategy in strategies:
        result = run_overlay_backtest(
            returns=returns,
            base_weights_history=base_weights,
            forecasts=forecasts,
            regime_data=regime_data,
            config=_strategy_config(cfg, strategy),
            strategy_name=strategy,
        )

        summary = result.summary.copy()
        summary["strategy"] = strategy
        summary_rows.append(summary)

        if not result.net_daily_returns.empty:
            dr = result.net_daily_returns.rename("daily_return").to_frame()
            dr["strategy"] = strategy
            daily_rows.append(dr.reset_index().rename(columns={dr.index.name or "index": "date"}))

        if not result.weights_history.empty:
            wh = result.weights_history.copy()
            wh["strategy"] = strategy
            weights_rows.append(wh.reset_index())

        if not result.decisions.empty:
            dec = result.decisions.copy()
            dec["strategy"] = strategy
            decisions_rows.append(dec.reset_index().rename(columns={dec.index.name or "index": "date"}))

        if not result.turnover.empty:
            to = result.turnover.copy()
            to["strategy"] = strategy
            turnover_rows.append(to.reset_index())

    summary_df = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    daily_df = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    weights_df = pd.concat(weights_rows, ignore_index=True) if weights_rows else pd.DataFrame()
    decisions_df = pd.concat(decisions_rows, ignore_index=True) if decisions_rows else pd.DataFrame()
    turnover_df = pd.concat(turnover_rows, ignore_index=True) if turnover_rows else pd.DataFrame()

    summary_path = output_dir / "overlay_backtest_summary.csv"
    daily_path = output_dir / "overlay_daily_returns.csv"
    weights_path = output_dir / "overlay_weights.csv"
    decisions_path = output_dir / "overlay_decisions.csv"
    turnover_path = output_dir / "overlay_turnover.csv"
    metadata_path = output_dir / "chapter5_metadata.json"

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    weights_df.to_csv(weights_path, index=False)
    decisions_df.to_csv(decisions_path, index=False)
    turnover_df.to_csv(turnover_path, index=False)

    metadata = {
        "chapter": 5,
        "name": "supervised_risk_overlay",
        "calendar_policy": cfg.get("calendar", {}).get("policy", "business_day_aligned"),
        "annualization_factor": cfg.get("calendar", {}).get("annualization_factor", 252),
        "holding_return_method": "drifted_buy_and_hold",
        "validation": cfg.get("validation", {}),
        "targets_config": cfg.get("targets", {}),
        "features_config": cfg.get("features", {}),
        "models_attempted": sorted(pd.read_csv(output_dir / "model_scores.csv")["model_name"].dropna().unique().tolist()) if (output_dir / "model_scores.csv").exists() else [],
        "models_selected": selection.to_dict(orient="records") if not selection.empty else [],
        "overlay_forecast_source": forecast_meta,
        "forecast_coverage": {
            "n_rebalance_dates": int(len(base_weights.index)),
            "n_forecast_rows": int(len(forecasts.index)) if not forecasts.empty else 0,
            "n_rebalance_with_forecast": int(base_weights.index.intersection(forecasts.dropna(how="all").index).size) if not forecasts.empty else 0,
        },
        "input_paths": {
            "returns_path": data_cfg.get("returns_path", "data/processed/returns_simple.csv"),
            "baseline_weights_path": data_cfg.get("baseline_weights_path", "data/processed/weights_history.csv"),
            "regime_labels_path": data_cfg.get("regime_labels_path", "data/processed/regime_analysis/regime_labels.csv"),
        },
        "output_paths": {
            "summary": str(summary_path),
            "daily_returns": str(daily_path),
            "weights": str(weights_path),
            "decisions": str(decisions_path),
            "turnover": str(turnover_path),
        },
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "notes": [
            "Chapter 5 focuses on risk forecasting (volatility/stress), not return prediction.",
            "If ML does not improve walk-forward validation vs naive, fallback uses naive/logistic baselines.",
            "No synthetic cash is introduced in base overlay; de-risking redistributes to defensive assets.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Chapter 5 - risk overlay")
    print(f"  calendar_policy       : {cfg.get('calendar', {}).get('policy', 'business_day_aligned')}")
    print(f"  annualization_factor  : {cfg.get('calendar', {}).get('annualization_factor', 252)}")
    print(f"  output_summary_path   : {summary_path.relative_to(PROJECT_ROOT)}")
    print(f"  output_decisions_path : {decisions_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
