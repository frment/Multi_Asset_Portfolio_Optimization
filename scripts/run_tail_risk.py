"""Script entry point for Chapter 3: Tail Risk and Alternative Objectives.

Implements a fair comparison between minimum variance and minimum historical
CVaR under the same universe and constraints, plus historical stress windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_min_variance_backtest, run_rolling_backtest  # noqa: E402
from src.config import (  # noqa: E402
    get_calendar_settings,
    load_dataset_metadata,
    load_settings,
    load_tail_risk,
    resolve_annualization_factor,
)
from src.costs import (  # noqa: E402
    apply_rebalance_costs_to_daily_returns,
    bps_to_rate,
    build_rebalance_cost_series,
)
from src.cvar_optimizer import minimise_historical_cvar  # noqa: E402
from src.metrics import compute_all_metrics, expected_shortfall_historical, return_over_es  # noqa: E402
from src.optimizer import load_optimizer_config  # noqa: E402
from src.stress import run_historical_stress_windows  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_returns(returns_csv: Path) -> pd.DataFrame:
    if not returns_csv.exists():
        raise FileNotFoundError(
            f"Returns file not found: {returns_csv}\n"
            "Run scripts/run_download.py and scripts/run_build_dataset.py first."
        )
    return pd.read_csv(returns_csv, index_col=0, parse_dates=True)


def _make_constraints_config(base_config: dict, max_crypto_weight: float) -> dict:
    cfg = dict(base_config)
    cfg["max_crypto_weight"] = float(max_crypto_weight)
    return cfg


def _summary_row(
    strategy: str,
    returns: pd.Series,
    turnover_history: pd.DataFrame,
    beta: float,
    risk_free_rate: float,
    objective: str,
    lookback_window_days: int,
    rebalance_frequency: str,
    max_total_crypto_weight: float,
    annualization_factor: float,
) -> dict:
    perf = compute_all_metrics(
        returns,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )
    es = expected_shortfall_historical(returns, beta=beta)
    ratio = return_over_es(returns, beta=beta, annualization_factor=annualization_factor)

    op_turnover = turnover_history.loc[
        ~turnover_history["is_initial_rebalance"],
        "turnover_one_way",
    ]

    return {
        "strategy": strategy,
        "objective": objective,
        "lookback_window_days": int(lookback_window_days),
        "rebalance_frequency": rebalance_frequency,
        "beta": float(beta),
        "max_total_crypto_weight": float(max_total_crypto_weight),
        "ann_return": perf["ann_return"],
        "ann_volatility": perf["ann_volatility"],
        "sharpe": perf["sharpe"],
        "max_drawdown": perf["max_drawdown"],
        "calmar": perf["calmar"],
        "expected_shortfall": es,
        "return_over_es": ratio,
        "mean_turnover": float(op_turnover.mean()) if len(op_turnover) > 0 else float("nan"),
        "median_turnover": float(op_turnover.median()) if len(op_turnover) > 0 else float("nan"),
        "max_turnover": float(op_turnover.max()) if len(op_turnover) > 0 else float("nan"),
        "oos_start": returns.index.min().date().isoformat(),
        "oos_end": returns.index.max().date().isoformat(),
        "n_oos_days": int(len(returns)),
        "n_rebalances": int(len(turnover_history)),
    }


def main() -> None:
    try:
        settings = load_settings()
        tail_cfg = load_tail_risk()

        processed_dir = PROJECT_ROOT / settings["paths"]["data_processed"]
        out_dir = processed_dir / "tail_risk"
        ensure_directory(out_dir)

        returns = _load_returns(processed_dir / "returns_simple.csv")

        backtest_cfg = settings.get("backtest", {})
        dataset_metadata = load_dataset_metadata()
        calendar_cfg = get_calendar_settings(settings)
        annualization_factor = resolve_annualization_factor(settings=settings, dataset_metadata=dataset_metadata)
        holding_return_method = str(backtest_cfg.get("holding_return_method", "drifted_buy_and_hold"))
        allow_weekend_rebalances = bool(calendar_cfg.get("allow_weekend_rebalances", False))
        risk_free_rate = float(backtest_cfg.get("risk_free_rate", 0.0))

        tr = tail_cfg.get("tail_risk", {})
        constraints = tail_cfg.get("constraints", {})
        stress_windows = tail_cfg.get("stress_windows", [])
        costs_cfg = tail_cfg.get("costs", {})

        lookback = int(tr.get("lookback_window_days", 252))
        rebalance_frequency = str(tr.get("rebalance_frequency", "monthly"))
        beta = float(tr.get("beta", 0.95))

        print("Methodology settings")
        print(f"  calendar_policy       : {dataset_metadata.get('calendar_policy', calendar_cfg.get('policy'))}")
        print(f"  annualization_factor  : {annualization_factor}")
        print(f"  holding_return_method : {holding_return_method}")
        print(f"  rebalance_frequency   : {rebalance_frequency}")

        central_crypto_cap = float(constraints.get("max_total_crypto_weight", 0.20))

        base_optimizer_cfg = load_optimizer_config()
        base_optimizer_cfg["max_weight"] = float(constraints.get("max_weight_per_asset", base_optimizer_cfg["max_weight"]))
        base_optimizer_cfg["long_only"] = bool(constraints.get("long_only", base_optimizer_cfg["long_only"]))

        configs = {
            "minvar_baseline_ch1": {
                "objective": "min_variance",
                "max_total_crypto_weight": central_crypto_cap,
            },
            "minvar_no_crypto_control": {
                "objective": "min_variance",
                "max_total_crypto_weight": 0.0,
            },
            "cvar_baseline": {
                "objective": "min_historical_cvar",
                "max_total_crypto_weight": central_crypto_cap,
                "beta": beta,
            },
            "cvar_no_crypto_control": {
                "objective": "min_historical_cvar",
                "max_total_crypto_weight": 0.0,
                "beta": beta,
            },
        }

        beta_sens = tr.get("beta_sensitivity", {})
        if bool(beta_sens.get("enabled", False)):
            for b in beta_sens.get("values", [0.975]):
                b_val = float(b)
                label = str(b_val).replace(".", "")
                configs[f"cvar_baseline_beta_{label}"] = {
                    "objective": "min_historical_cvar",
                    "max_total_crypto_weight": central_crypto_cap,
                    "beta": b_val,
                }

        strategy_returns: dict[str, pd.Series] = {}
        strategy_weights: dict[str, pd.DataFrame] = {}
        strategy_turnover: dict[str, pd.DataFrame] = {}

        for strategy_name, spec in configs.items():
            cfg = _make_constraints_config(base_optimizer_cfg, spec["max_total_crypto_weight"])

            if spec["objective"] == "min_variance":
                series, weights_history, turnover_history = run_min_variance_backtest(
                    returns=returns,
                    optimizer_config=cfg,
                    lookback_window=lookback,
                    rebalance_frequency=rebalance_frequency,
                    holding_return_method=holding_return_method,
                    allow_weekend_rebalances=allow_weekend_rebalances,
                )
            else:
                series, weights_history, turnover_history = run_rolling_backtest(
                    returns=returns,
                    optimizer_fn=minimise_historical_cvar,
                    optimizer_config=cfg,
                    lookback_window=lookback,
                    rebalance_frequency=rebalance_frequency,
                    strategy_name=strategy_name,
                    optimizer_kwargs={"beta": float(spec.get("beta", beta))},
                    holding_return_method=holding_return_method,
                    allow_weekend_rebalances=allow_weekend_rebalances,
                )

            strategy_returns[strategy_name] = series.rename(strategy_name)
            strategy_weights[strategy_name] = weights_history
            strategy_turnover[strategy_name] = turnover_history

        common_start = max(series.index.min() for series in strategy_returns.values())
        common_end = min(series.index.max() for series in strategy_returns.values())
        strategy_returns = {
            k: v.loc[(v.index >= common_start) & (v.index <= common_end)].sort_index()
            for k, v in strategy_returns.items()
        }

        summary_rows: list[dict] = []
        returns_rows: list[pd.DataFrame] = []
        weights_rows: list[pd.DataFrame] = []
        turnover_rows: list[pd.DataFrame] = []

        for strategy_name, series in strategy_returns.items():
            spec = configs[strategy_name]
            turnover_history = strategy_turnover[strategy_name]
            beta_used = float(spec.get("beta", beta))

            summary_rows.append(
                _summary_row(
                    strategy=strategy_name,
                    returns=series,
                    turnover_history=turnover_history,
                    beta=beta_used,
                    risk_free_rate=risk_free_rate,
                    objective=spec["objective"],
                    lookback_window_days=lookback,
                    rebalance_frequency=rebalance_frequency,
                    max_total_crypto_weight=float(spec["max_total_crypto_weight"]),
                    annualization_factor=annualization_factor,
                )
            )

            frame = series.to_frame(name="portfolio_return").reset_index()
            frame = frame.rename(columns={frame.columns[0]: "date"})
            frame["strategy"] = strategy_name
            frame["objective"] = spec["objective"]
            frame["beta"] = beta_used
            frame["max_total_crypto_weight"] = float(spec["max_total_crypto_weight"])
            returns_rows.append(frame[[
                "date",
                "strategy",
                "objective",
                "beta",
                "max_total_crypto_weight",
                "portfolio_return",
            ]])

            w = strategy_weights[strategy_name].reset_index().melt(
                id_vars=["rebalance_date"],
                var_name="ticker",
                value_name="weight",
            )
            w["strategy"] = strategy_name
            w["objective"] = spec["objective"]
            w["beta"] = beta_used
            w["max_total_crypto_weight"] = float(spec["max_total_crypto_weight"])
            weights_rows.append(w[[
                "rebalance_date",
                "strategy",
                "objective",
                "beta",
                "max_total_crypto_weight",
                "ticker",
                "weight",
            ]])

            t = strategy_turnover[strategy_name].reset_index().copy()
            t["strategy"] = strategy_name
            t["objective"] = spec["objective"]
            t["beta"] = beta_used
            t["max_total_crypto_weight"] = float(spec["max_total_crypto_weight"])
            turnover_rows.append(t[[
                "rebalance_date",
                "strategy",
                "objective",
                "beta",
                "max_total_crypto_weight",
                "turnover_one_way",
                "is_initial_rebalance",
                "n_assets_changed",
                "max_abs_weight_change",
            ]])

        summary = pd.DataFrame(summary_rows).sort_values("strategy").reset_index(drop=True)
        returns_panel = pd.concat(returns_rows, ignore_index=True).sort_values(["strategy", "date"])
        weights_panel = pd.concat(weights_rows, ignore_index=True).sort_values(["strategy", "rebalance_date", "ticker"])
        turnover_panel = pd.concat(turnover_rows, ignore_index=True).sort_values(["strategy", "rebalance_date"])

        net_summary = pd.DataFrame(columns=[])
        stress_frames: list[pd.DataFrame] = []

        if bool(costs_cfg.get("enabled", True)):
            cost_rate = bps_to_rate(float(costs_cfg.get("cost_bps", 10.0)))

            net_rows: list[dict] = []
            net_returns: dict[str, pd.Series] = {}
            for _, row in summary.iterrows():
                strategy = str(row["strategy"])
                gross_series = strategy_returns[strategy]
                turnover_series = strategy_turnover[strategy]["turnover_one_way"]

                rebalance_costs = build_rebalance_cost_series(turnover_series, cost_rate=cost_rate)
                net_series, _ = apply_rebalance_costs_to_daily_returns(gross_series, rebalance_costs)
                net_returns[strategy] = net_series

                es = expected_shortfall_historical(net_series, beta=float(row["beta"]))
                perf = compute_all_metrics(
                    net_series,
                    risk_free_rate=risk_free_rate,
                    annualization_factor=annualization_factor,
                )
                ratio = return_over_es(
                    net_series,
                    beta=float(row["beta"]),
                    annualization_factor=annualization_factor,
                )

                net_rows.append(
                    {
                        "strategy": strategy,
                        "objective": row["objective"],
                        "lookback_window_days": int(row["lookback_window_days"]),
                        "rebalance_frequency": row["rebalance_frequency"],
                        "beta": float(row["beta"]),
                        "max_total_crypto_weight": float(row["max_total_crypto_weight"]),
                        "cost_bps": float(costs_cfg.get("cost_bps", 10.0)),
                        "cost_rate": float(cost_rate),
                        "ann_return_gross": float(row["ann_return"]),
                        "ann_volatility_gross": float(row["ann_volatility"]),
                        "sharpe_gross": float(row["sharpe"]),
                        "max_drawdown_gross": float(row["max_drawdown"]),
                        "calmar_gross": float(row["calmar"]),
                        "expected_shortfall_gross": float(row["expected_shortfall"]),
                        "return_over_es_gross": float(row["return_over_es"]),
                        "ann_return_net": perf["ann_return"],
                        "ann_volatility_net": perf["ann_volatility"],
                        "sharpe_net": perf["sharpe"],
                        "max_drawdown_net": perf["max_drawdown"],
                        "calmar_net": perf["calmar"],
                        "expected_shortfall_net": es,
                        "return_over_es_net": ratio,
                        "oos_start": row["oos_start"],
                        "oos_end": row["oos_end"],
                        "n_oos_days": int(row["n_oos_days"]),
                        "n_rebalances": int(row["n_rebalances"]),
                    }
                )

            net_summary = pd.DataFrame(net_rows).sort_values("strategy").reset_index(drop=True)

            stress_frames.append(
                run_historical_stress_windows(
                    strategy_returns=strategy_returns,
                    windows=stress_windows,
                    beta=beta,
                    risk_free_rate=risk_free_rate,
                    scope="gross",
                    annualization_factor=annualization_factor,
                )
            )
            stress_frames.append(
                run_historical_stress_windows(
                    strategy_returns=net_returns,
                    windows=stress_windows,
                    beta=beta,
                    risk_free_rate=risk_free_rate,
                    scope="net",
                    annualization_factor=annualization_factor,
                )
            )
        else:
            stress_frames.append(
                run_historical_stress_windows(
                    strategy_returns=strategy_returns,
                    windows=stress_windows,
                    beta=beta,
                    risk_free_rate=risk_free_rate,
                    scope="gross",
                    annualization_factor=annualization_factor,
                )
            )

        stress_summary = pd.concat(stress_frames, ignore_index=True) if stress_frames else pd.DataFrame()

        summary.to_csv(out_dir / "tail_risk_summary.csv", index=False)
        net_summary.to_csv(out_dir / "tail_risk_summary_net.csv", index=False)
        returns_panel.to_csv(out_dir / "tail_risk_returns.csv", index=False)
        weights_panel.to_csv(out_dir / "tail_risk_weights_panel.csv", index=False)
        turnover_panel.to_csv(out_dir / "tail_risk_turnover_panel.csv", index=False)
        stress_summary.to_csv(out_dir / "stress_summary.csv", index=False)

        print("Chapter 3 tail-risk run complete.")
        print(f"  Strategies evaluated: {len(summary)}")
        print(f"  Output directory     : {out_dir.relative_to(PROJECT_ROOT)}")

    except Exception as error:
        print(f"\nTail-risk run failed. Reason: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
