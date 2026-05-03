"""Robustness experiment utilities for Chapter 2.

This module runs a first-pass, one-factor-at-a-time robustness sweep for the
minimum variance strategy while preserving Chapter 1 defaults.

Design choices:
- Anchors are frozen and always included:
  - baseline_ch1
  - minvar_no_crypto_control
- First pass isolates one dimension at a time (no Cartesian grid).
- Weekly rebalance is supported by the backtest engine, but excluded from the
  default first-pass sweep unless explicitly enabled.
- Cost scenarios are intentionally out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.backtest import run_min_variance_backtest
from src.costs import (
    apply_rebalance_costs_to_daily_returns,
    bps_to_rate,
    build_rebalance_cost_series,
)
from src.metrics import compute_all_metrics

SUMMARY_COLUMNS: list[str] = [
    "experiment_id",
    "dimension_tested",
    "lookback_window_days",
    "max_total_crypto_weight",
    "rebalance_frequency",
    "covariance_method",
    "ann_return",
    "ann_volatility",
    "sharpe",
    "max_drawdown",
    "calmar",
    "mean_turnover",
    "median_turnover",
    "max_turnover",
    "oos_start",
    "oos_end",
    "n_oos_days",
    "n_rebalances",
]


@dataclass(frozen=True)
class RobustnessSpec:
    """Definition of one robustness experiment specification."""

    experiment_id: str
    dimension_tested: str
    family: str
    lookback_window_days: int
    max_total_crypto_weight: float
    rebalance_frequency: str
    covariance_method: str = "sample"
    is_anchor: bool = False


@dataclass
class RunResult:
    """Cached outputs for one parameter signature."""

    returns: pd.Series
    weights_history: pd.DataFrame
    turnover_history: pd.DataFrame


def build_first_pass_specs(
    *,
    base_lookback_window_days: int,
    base_max_total_crypto_weight: float,
    base_rebalance_frequency: str,
    include_weekly_in_first_pass: bool = False,
    lookback_values: list[int] | None = None,
    crypto_cap_values: list[float] | None = None,
    rebalance_values: list[str] | None = None,
    covariance_methods: list[str] | None = None,
    include_no_crypto_anchor_in_covariance_family: bool = True,
) -> list[RobustnessSpec]:
    """Build first-pass one-factor robustness specs.

    Includes two fixed anchors and three one-factor families:
    lookback, crypto cap, and rebalance frequency.
    """
    specs: list[RobustnessSpec] = [
        RobustnessSpec(
            experiment_id="baseline_ch1",
            dimension_tested="anchor",
            family="anchors",
            lookback_window_days=base_lookback_window_days,
            max_total_crypto_weight=base_max_total_crypto_weight,
            rebalance_frequency=base_rebalance_frequency,
            covariance_method="sample",
            is_anchor=True,
        ),
        RobustnessSpec(
            experiment_id="minvar_no_crypto_control",
            dimension_tested="anchor",
            family="anchors",
            lookback_window_days=base_lookback_window_days,
            max_total_crypto_weight=0.00,
            rebalance_frequency=base_rebalance_frequency,
            covariance_method="sample",
            is_anchor=True,
        ),
    ]

    lookback_levels = lookback_values if lookback_values is not None else [126, 252, 504]
    for lookback in lookback_levels:
        specs.append(
            RobustnessSpec(
                experiment_id=f"lookback_{lookback}",
                dimension_tested="lookback",
                family="lookback",
                lookback_window_days=lookback,
                max_total_crypto_weight=base_max_total_crypto_weight,
                rebalance_frequency=base_rebalance_frequency,
                covariance_method="sample",
            )
        )

    cap_levels = crypto_cap_values if crypto_cap_values is not None else [0.00, 0.10, 0.20, 0.25]
    for cap in cap_levels:
        cap_text = f"{cap:.2f}".replace(".", "_")
        specs.append(
            RobustnessSpec(
                experiment_id=f"crypto_cap_{cap_text}",
                dimension_tested="crypto_cap",
                family="crypto_cap",
                lookback_window_days=base_lookback_window_days,
                max_total_crypto_weight=cap,
                rebalance_frequency=base_rebalance_frequency,
                covariance_method="sample",
            )
        )

    rebalance_frequencies = rebalance_values if rebalance_values is not None else ["monthly", "quarterly"]
    rebalance_frequencies = [str(freq).lower().strip() for freq in rebalance_frequencies]

    if include_weekly_in_first_pass and "weekly" not in rebalance_frequencies:
        rebalance_frequencies.append("weekly")
    if not include_weekly_in_first_pass:
        rebalance_frequencies = [freq for freq in rebalance_frequencies if freq != "weekly"]

    for freq in rebalance_frequencies:
        specs.append(
            RobustnessSpec(
                experiment_id=f"rebalance_{freq}",
                dimension_tested="rebalance_frequency",
                family="rebalance",
                lookback_window_days=base_lookback_window_days,
                max_total_crypto_weight=base_max_total_crypto_weight,
                rebalance_frequency=freq,
                covariance_method="sample",
            )
        )

    cov_methods = covariance_methods if covariance_methods is not None else ["sample", "ledoit_wolf"]
    normalized_cov_methods = [str(method).lower().strip() for method in cov_methods]

    for method in normalized_cov_methods:
        specs.append(
            RobustnessSpec(
                experiment_id=f"covariance_{method}_baseline_ch1",
                dimension_tested="covariance_method",
                family="covariance_method",
                lookback_window_days=base_lookback_window_days,
                max_total_crypto_weight=base_max_total_crypto_weight,
                rebalance_frequency=base_rebalance_frequency,
                covariance_method=method,
            )
        )

        if include_no_crypto_anchor_in_covariance_family:
            specs.append(
                RobustnessSpec(
                    experiment_id=f"covariance_{method}_minvar_no_crypto_control",
                    dimension_tested="covariance_method",
                    family="covariance_method",
                    lookback_window_days=base_lookback_window_days,
                    max_total_crypto_weight=0.00,
                    rebalance_frequency=base_rebalance_frequency,
                    covariance_method=method,
                )
            )

    return specs


def _signature(spec: RobustnessSpec) -> tuple[int, float, str, str]:
    return (
        int(spec.lookback_window_days),
        float(spec.max_total_crypto_weight),
        str(spec.rebalance_frequency),
        str(spec.covariance_method),
    )


def _turnover_stats(turnover_history: pd.DataFrame) -> tuple[float, float, float]:
    oos_turnover = turnover_history.loc[
        ~turnover_history["is_initial_rebalance"],
        "turnover_one_way",
    ]
    if oos_turnover.empty:
        return float("nan"), float("nan"), float("nan")
    return float(oos_turnover.mean()), float(oos_turnover.median()), float(oos_turnover.max())


def _to_native_summary_row(
    spec: RobustnessSpec,
    run_result: RunResult,
    risk_free_rate: float,
    annualization_factor: float,
) -> dict[str, Any]:
    series = run_result.returns
    metrics = compute_all_metrics(
        series,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )
    mean_to, med_to, max_to = _turnover_stats(run_result.turnover_history)

    return {
        "experiment_id": spec.experiment_id,
        "dimension_tested": spec.dimension_tested,
        "family": spec.family,
        "is_anchor": spec.is_anchor,
        "sample_scope": "native",
        "lookback_window_days": int(spec.lookback_window_days),
        "max_total_crypto_weight": float(spec.max_total_crypto_weight),
        "rebalance_frequency": spec.rebalance_frequency,
        "covariance_method": spec.covariance_method,
        "ann_return": metrics["ann_return"],
        "ann_volatility": metrics["ann_volatility"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar": metrics["calmar"],
        "mean_turnover": mean_to,
        "median_turnover": med_to,
        "max_turnover": max_to,
        "oos_start": series.index.min().date().isoformat(),
        "oos_end": series.index.max().date().isoformat(),
        "n_oos_days": int(len(series)),
        "n_rebalances": int(len(run_result.turnover_history)),
    }


def _returns_panel_row(spec: RobustnessSpec, returns: pd.Series) -> pd.DataFrame:
    df = returns.to_frame(name="portfolio_return").reset_index()
    df = df.rename(columns={df.columns[0]: "date"})
    df["experiment_id"] = spec.experiment_id
    df["dimension_tested"] = spec.dimension_tested
    df["family"] = spec.family
    df["sample_scope"] = "native"
    df["lookback_window_days"] = int(spec.lookback_window_days)
    df["max_total_crypto_weight"] = float(spec.max_total_crypto_weight)
    df["rebalance_frequency"] = spec.rebalance_frequency
    df["covariance_method"] = spec.covariance_method
    return df[
        [
            "date",
            "experiment_id",
            "dimension_tested",
            "family",
            "sample_scope",
            "lookback_window_days",
            "max_total_crypto_weight",
            "rebalance_frequency",
            "covariance_method",
            "portfolio_return",
        ]
    ]


def _weights_panel_rows(spec: RobustnessSpec, weights_history: pd.DataFrame) -> pd.DataFrame:
    df = weights_history.reset_index().melt(
        id_vars=["rebalance_date"],
        var_name="ticker",
        value_name="weight",
    )
    df["experiment_id"] = spec.experiment_id
    df["dimension_tested"] = spec.dimension_tested
    df["family"] = spec.family
    df["lookback_window_days"] = int(spec.lookback_window_days)
    df["max_total_crypto_weight"] = float(spec.max_total_crypto_weight)
    df["rebalance_frequency"] = spec.rebalance_frequency
    df["covariance_method"] = spec.covariance_method
    return df[
        [
            "rebalance_date",
            "experiment_id",
            "dimension_tested",
            "family",
            "lookback_window_days",
            "max_total_crypto_weight",
            "rebalance_frequency",
            "covariance_method",
            "ticker",
            "weight",
        ]
    ]


def _turnover_panel_rows(spec: RobustnessSpec, turnover_history: pd.DataFrame) -> pd.DataFrame:
    df = turnover_history.reset_index().copy()
    df["experiment_id"] = spec.experiment_id
    df["dimension_tested"] = spec.dimension_tested
    df["family"] = spec.family
    df["lookback_window_days"] = int(spec.lookback_window_days)
    df["max_total_crypto_weight"] = float(spec.max_total_crypto_weight)
    df["rebalance_frequency"] = spec.rebalance_frequency
    df["covariance_method"] = spec.covariance_method
    ordered = [
        "rebalance_date",
        "experiment_id",
        "dimension_tested",
        "family",
        "lookback_window_days",
        "max_total_crypto_weight",
        "rebalance_frequency",
        "covariance_method",
        "turnover_one_way",
        "is_initial_rebalance",
        "n_assets_changed",
        "max_abs_weight_change",
    ]
    return df[ordered]


def _build_family_common_summary(
    native_summary: pd.DataFrame,
    returns_panel: pd.DataFrame,
    turnover_panel: pd.DataFrame,
    risk_free_rate: float,
    annualization_factor: float,
) -> pd.DataFrame:
    """Recompute metrics on common sample per experiment family."""
    rows: list[dict[str, Any]] = []

    returns_panel = returns_panel.copy()
    returns_panel["date"] = pd.to_datetime(returns_panel["date"])

    turnover_panel = turnover_panel.copy()
    turnover_panel["rebalance_date"] = pd.to_datetime(turnover_panel["rebalance_date"])

    family_names = sorted(
        family_name
        for family_name in native_summary["family"].dropna().astype(str).unique().tolist()
        if family_name != "anchors"
    )
    for family in family_names:
        fam_specs = native_summary.loc[native_summary["family"] == family].copy()
        if fam_specs.empty:
            continue

        common_start = pd.to_datetime(fam_specs["oos_start"]).max()
        common_end = pd.to_datetime(fam_specs["oos_end"]).min()
        if common_start > common_end:
            continue

        for _, spec_row in fam_specs.iterrows():
            exp_id = spec_row["experiment_id"]
            exp_returns = returns_panel.loc[
                (returns_panel["experiment_id"] == exp_id)
                & (returns_panel["date"] >= common_start)
                & (returns_panel["date"] <= common_end),
                ["date", "portfolio_return"],
            ].copy()

            if exp_returns.empty:
                continue

            series = exp_returns.set_index("date")["portfolio_return"].sort_index()
            metrics = compute_all_metrics(
                series,
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
            )

            exp_turnover = turnover_panel.loc[
                (turnover_panel["experiment_id"] == exp_id)
                & (turnover_panel["rebalance_date"] >= common_start)
                & (turnover_panel["rebalance_date"] <= common_end)
            ].copy()

            if exp_turnover.empty:
                mean_to = float("nan")
                med_to = float("nan")
                max_to = float("nan")
                n_rebalances = 0
            else:
                oos_turnover = exp_turnover.loc[
                    ~exp_turnover["is_initial_rebalance"],
                    "turnover_one_way",
                ]
                mean_to = float(oos_turnover.mean()) if not oos_turnover.empty else float("nan")
                med_to = float(oos_turnover.median()) if not oos_turnover.empty else float("nan")
                max_to = float(oos_turnover.max()) if not oos_turnover.empty else float("nan")
                n_rebalances = int(len(exp_turnover))

            rows.append(
                {
                    "experiment_id": exp_id,
                    "dimension_tested": spec_row["dimension_tested"],
                    "family": family,
                    "is_anchor": bool(spec_row["is_anchor"]),
                    "sample_scope": "common_family",
                    "lookback_window_days": int(spec_row["lookback_window_days"]),
                    "max_total_crypto_weight": float(spec_row["max_total_crypto_weight"]),
                    "rebalance_frequency": spec_row["rebalance_frequency"],
                    "covariance_method": spec_row.get("covariance_method", "sample"),
                    "ann_return": metrics["ann_return"],
                    "ann_volatility": metrics["ann_volatility"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "calmar": metrics["calmar"],
                    "mean_turnover": mean_to,
                    "median_turnover": med_to,
                    "max_turnover": max_to,
                    "oos_start": series.index.min().date().isoformat(),
                    "oos_end": series.index.max().date().isoformat(),
                    "n_oos_days": int(len(series)),
                    "n_rebalances": n_rebalances,
                    "family_common_start": common_start.date().isoformat(),
                    "family_common_end": common_end.date().isoformat(),
                }
            )

    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS + ["family", "is_anchor", "sample_scope", "family_common_start", "family_common_end"])

    out = pd.DataFrame(rows)
    ordered_cols = SUMMARY_COLUMNS + ["family", "is_anchor", "sample_scope", "family_common_start", "family_common_end"]
    return out[ordered_cols]


def _build_cost_scenario_summary(
    base_summary: pd.DataFrame,
    returns_panel: pd.DataFrame,
    turnover_panel: pd.DataFrame,
    risk_free_rate: float,
    cost_scenarios_bps: list[float],
    annualization_factor: float,
) -> pd.DataFrame:
    """Build net-of-cost summaries for each experiment and cost scenario.

    Costs are applied on rebalance dates as:
        cost_t = turnover_one_way_t * cost_rate
    """
    rows: list[dict[str, Any]] = []

    returns_panel = returns_panel.copy()
    returns_panel["date"] = pd.to_datetime(returns_panel["date"])

    turnover_panel = turnover_panel.copy()
    turnover_panel["rebalance_date"] = pd.to_datetime(turnover_panel["rebalance_date"])

    for _, spec_row in base_summary.iterrows():
        exp_id = spec_row["experiment_id"]
        oos_start = pd.to_datetime(spec_row["oos_start"])
        oos_end = pd.to_datetime(spec_row["oos_end"])

        exp_returns_df = returns_panel.loc[
            (returns_panel["experiment_id"] == exp_id)
            & (returns_panel["date"] >= oos_start)
            & (returns_panel["date"] <= oos_end),
            ["date", "portfolio_return"],
        ].copy()
        if exp_returns_df.empty:
            continue

        gross_series = exp_returns_df.set_index("date")["portfolio_return"].sort_index()

        exp_turnover_df = turnover_panel.loc[
            (turnover_panel["experiment_id"] == exp_id)
            & (turnover_panel["rebalance_date"] >= oos_start)
            & (turnover_panel["rebalance_date"] <= oos_end),
            ["rebalance_date", "turnover_one_way"],
        ].copy()

        turnover_series = (
            exp_turnover_df.set_index("rebalance_date")["turnover_one_way"].sort_index()
            if not exp_turnover_df.empty
            else pd.Series(dtype=float)
        )

        for cost_bps in cost_scenarios_bps:
            rate = bps_to_rate(cost_bps)
            rebalance_costs = build_rebalance_cost_series(turnover_series, cost_rate=rate)
            net_series, daily_costs = apply_rebalance_costs_to_daily_returns(
                gross_series,
                rebalance_costs,
            )
            net_metrics = compute_all_metrics(
                net_series,
                risk_free_rate=risk_free_rate,
                annualization_factor=annualization_factor,
            )

            row = {
                "experiment_id": exp_id,
                "dimension_tested": spec_row["dimension_tested"],
                "family": spec_row.get("family", ""),
                "is_anchor": bool(spec_row.get("is_anchor", False)),
                "sample_scope": spec_row.get("sample_scope", "native"),
                "lookback_window_days": int(spec_row["lookback_window_days"]),
                "max_total_crypto_weight": float(spec_row["max_total_crypto_weight"]),
                "rebalance_frequency": spec_row["rebalance_frequency"],
                "covariance_method": spec_row.get("covariance_method", "sample"),
                "cost_bps": float(cost_bps),
                "cost_rate": float(rate),
                "ann_return_gross": float(spec_row["ann_return"]),
                "ann_volatility_gross": float(spec_row["ann_volatility"]),
                "sharpe_gross": float(spec_row["sharpe"]),
                "max_drawdown_gross": float(spec_row["max_drawdown"]),
                "calmar_gross": float(spec_row["calmar"]),
                "ann_return_net": float(net_metrics["ann_return"]),
                "ann_volatility_net": float(net_metrics["ann_volatility"]),
                "sharpe_net": float(net_metrics["sharpe"]),
                "max_drawdown_net": float(net_metrics["max_drawdown"]),
                "calmar_net": float(net_metrics["calmar"]),
                "cumulative_cost": float(daily_costs.sum()),
                "oos_start": spec_row["oos_start"],
                "oos_end": spec_row["oos_end"],
                "n_oos_days": int(spec_row["n_oos_days"]),
                "n_rebalances": int(spec_row["n_rebalances"]),
            }

            if "family_common_start" in spec_row.index:
                row["family_common_start"] = spec_row["family_common_start"]
            if "family_common_end" in spec_row.index:
                row["family_common_end"] = spec_row["family_common_end"]

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_first_pass_robustness(
    *,
    returns: pd.DataFrame,
    base_optimizer_config: dict[str, Any],
    risk_free_rate: float,
    base_lookback_window_days: int,
    base_rebalance_frequency: str,
    include_weekly_in_first_pass: bool = False,
    lookback_values: list[int] | None = None,
    crypto_cap_values: list[float] | None = None,
    rebalance_values: list[str] | None = None,
    covariance_methods: list[str] | None = None,
    include_no_crypto_anchor_in_covariance_family: bool = True,
    cost_scenarios_bps: list[float] | None = None,
    annualization_factor: float = 252.0,
    holding_return_method: str = "drifted_buy_and_hold",
    allow_weekend_rebalances: bool = True,
) -> dict[str, pd.DataFrame]:
    """Execute all first-pass robustness experiments.

    Returns DataFrames ready to save as CSV outputs.
    """
    specs = build_first_pass_specs(
        base_lookback_window_days=base_lookback_window_days,
        base_max_total_crypto_weight=float(base_optimizer_config["max_crypto_weight"]),
        base_rebalance_frequency=base_rebalance_frequency,
        include_weekly_in_first_pass=include_weekly_in_first_pass,
        lookback_values=lookback_values,
        crypto_cap_values=crypto_cap_values,
        rebalance_values=rebalance_values,
        covariance_methods=covariance_methods,
        include_no_crypto_anchor_in_covariance_family=include_no_crypto_anchor_in_covariance_family,
    )

    cache: dict[tuple[int, float, str, str], RunResult] = {}
    signature_owner: dict[tuple[int, float, str, str], str] = {}

    summary_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    returns_frames: list[pd.DataFrame] = []
    weights_frames: list[pd.DataFrame] = []
    turnover_frames: list[pd.DataFrame] = []

    for spec in specs:
        sig = _signature(spec)

        if sig not in cache:
            optimizer_config = dict(base_optimizer_config)
            optimizer_config["max_crypto_weight"] = float(spec.max_total_crypto_weight)

            portfolio_returns, weights_history, turnover_history = run_min_variance_backtest(
                returns=returns,
                optimizer_config=optimizer_config,
                lookback_window=int(spec.lookback_window_days),
                rebalance_frequency=spec.rebalance_frequency,
                covariance_method=spec.covariance_method,
                holding_return_method=holding_return_method,
                allow_weekend_rebalances=allow_weekend_rebalances,
            )
            cache[sig] = RunResult(
                returns=portfolio_returns,
                weights_history=weights_history,
                turnover_history=turnover_history,
            )
            signature_owner[sig] = spec.experiment_id
            reused_from = ""
        else:
            reused_from = signature_owner[sig]

        result = cache[sig]

        summary_rows.append(
            _to_native_summary_row(
                spec,
                result,
                risk_free_rate,
                annualization_factor,
            )
        )
        returns_frames.append(_returns_panel_row(spec, result.returns))
        weights_frames.append(_weights_panel_rows(spec, result.weights_history))
        turnover_frames.append(_turnover_panel_rows(spec, result.turnover_history))

        metadata_rows.append(
            {
                "experiment_id": spec.experiment_id,
                "dimension_tested": spec.dimension_tested,
                "family": spec.family,
                "is_anchor": spec.is_anchor,
                "lookback_window_days": int(spec.lookback_window_days),
                "max_total_crypto_weight": float(spec.max_total_crypto_weight),
                "rebalance_frequency": spec.rebalance_frequency,
                "covariance_method": spec.covariance_method,
                "sample_scope_supported": "native,common_family",
                "benchmark_policy": "min_variance_only_benchmarks_deferred",
                "transaction_cost_model": "cost_t=turnover_one_way_t*cost_rate_post_processing",
                "explicitly_out_of_scope": "asset_level_spreads,slippage,market_impact,asset_class_specific_costs",
                "annualization_factor": float(annualization_factor),
                "holding_return_method": holding_return_method,
                "reused_result_from": reused_from,
            }
        )

    native_summary = pd.DataFrame(summary_rows)
    native_summary = native_summary[SUMMARY_COLUMNS + ["family", "is_anchor", "sample_scope"]]

    returns_panel = pd.concat(returns_frames, ignore_index=True)
    returns_panel = returns_panel.sort_values(["experiment_id", "date"]).reset_index(drop=True)

    weights_panel = pd.concat(weights_frames, ignore_index=True)
    weights_panel = weights_panel.sort_values(["experiment_id", "rebalance_date", "ticker"]).reset_index(drop=True)

    turnover_panel = pd.concat(turnover_frames, ignore_index=True)
    turnover_panel = turnover_panel.sort_values(["experiment_id", "rebalance_date"]).reset_index(drop=True)

    metadata = pd.DataFrame(metadata_rows)

    common_family_summary = _build_family_common_summary(
        native_summary=native_summary,
        returns_panel=returns_panel,
        turnover_panel=turnover_panel,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor,
    )

    cost_scenarios = cost_scenarios_bps if cost_scenarios_bps is not None else [0.0, 10.0, 25.0, 50.0]
    net_summary = _build_cost_scenario_summary(
        base_summary=native_summary,
        returns_panel=returns_panel,
        turnover_panel=turnover_panel,
        risk_free_rate=risk_free_rate,
        cost_scenarios_bps=cost_scenarios,
        annualization_factor=annualization_factor,
    )
    net_summary_common_family = _build_cost_scenario_summary(
        base_summary=common_family_summary,
        returns_panel=returns_panel,
        turnover_panel=turnover_panel,
        risk_free_rate=risk_free_rate,
        cost_scenarios_bps=cost_scenarios,
        annualization_factor=annualization_factor,
    )

    return {
        "robustness_summary": native_summary,
        "robustness_summary_gross": native_summary,
        "robustness_returns": returns_panel,
        "robustness_metadata": metadata,
        "robustness_weights_panel": weights_panel,
        "robustness_turnover_panel": turnover_panel,
        "robustness_summary_common_family": common_family_summary,
        "robustness_summary_net": net_summary,
        "robustness_summary_common_family_net": net_summary_common_family,
    }
