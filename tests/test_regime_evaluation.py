"""Tests for Chapter 4 Phase 3 regime evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from metrics import expected_shortfall_historical  # noqa: E402
from regime_evaluation import (  # noqa: E402
    COND_PERF_COLUMNS,
    CRYPTO_EXPOSURE_COLUMNS,
    STRESS_MAP_COLUMNS,
    evaluate_regimes_from_frames,
)


STRATEGIES = [
    "minvar_baseline_ch1",
    "minvar_no_crypto_control",
    "cvar_baseline",
    "cvar_no_crypto_control",
]


def _make_cfg() -> dict:
    return {
        "evaluation": {
            "cost_bps": 10.0,
            "min_obs_warning_threshold": 2,
            "es_beta": 0.95,
            "crypto_weight_tolerance": 1e-6,
            "strategies": STRATEGIES,
            "crypto_tickers": ["BTC-USD", "ETH-USD"],
        }
    }


def _make_labels() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "regime_id": [0, 0, 1, 1, 0, 1],
            "regime_name": [
                "Low-stress / Risk-on",
                "Low-stress / Risk-on",
                "High-stress / Risk-off",
                "High-stress / Risk-off",
                "Low-stress / Risk-on",
                "High-stress / Risk-off",
            ],
        }
    )


def _make_returns() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=7, freq="B")
    rows: list[dict] = []

    base = {
        "minvar_baseline_ch1": [0.01, -0.02, 0.01, -0.015, 0.008, -0.005, 0.002],
        "minvar_no_crypto_control": [0.008, -0.018, 0.009, -0.014, 0.007, -0.004, 0.001],
        "cvar_baseline": [0.009, -0.016, 0.011, -0.013, 0.006, -0.003, 0.002],
        "cvar_no_crypto_control": [0.007, -0.015, 0.010, -0.012, 0.005, -0.002, 0.001],
    }

    for strategy, series in base.items():
        for d, ret in zip(dates, series):
            rows.append(
                {
                    "date": d,
                    "strategy": strategy,
                    "portfolio_return": ret,
                }
            )

    return pd.DataFrame(rows)


def _make_turnover() -> pd.DataFrame:
    rebalance_dates = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-07"])
    rows: list[dict] = []
    for strategy in STRATEGIES:
        for d in rebalance_dates:
            rows.append(
                {
                    "rebalance_date": d,
                    "strategy": strategy,
                    "turnover_one_way": 0.0 if d == rebalance_dates[0] else 0.05,
                }
            )
    return pd.DataFrame(rows)


def _make_weights() -> pd.DataFrame:
    rebalance_dates = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-07"])
    rows: list[dict] = []

    for strategy in STRATEGIES:
        for d in rebalance_dates:
            if strategy in {"minvar_no_crypto_control", "cvar_no_crypto_control"}:
                btc = 0.0
                eth = 0.0
            else:
                btc = 0.02 if d == rebalance_dates[0] else 0.03
                eth = 0.01 if d == rebalance_dates[0] else 0.015

            rows.extend(
                [
                    {"rebalance_date": d, "strategy": strategy, "ticker": "BTC-USD", "weight": btc},
                    {"rebalance_date": d, "strategy": strategy, "ticker": "ETH-USD", "weight": eth},
                    {"rebalance_date": d, "strategy": strategy, "ticker": "SPY", "weight": 1.0 - btc - eth},
                ]
            )

    return pd.DataFrame(rows)


def _make_stress_summary() -> pd.DataFrame:
    rows: list[dict] = []
    for strategy in STRATEGIES:
        for scope in ["gross", "net"]:
            rows.append(
                {
                    "window_id": "w1",
                    "window_label": "Window 1",
                    "start_date": "2020-01-01",
                    "end_date": "2020-01-08",
                    "strategy": strategy,
                    "scope": scope,
                    "n_days": 6,
                    "ann_return": 0.1,
                    "ann_volatility": 0.2,
                    "sharpe": 0.5,
                    "max_drawdown": -0.1,
                    "calmar": 1.0,
                    "expected_shortfall": 0.03,
                    "return_over_es": 3.0,
                }
            )
    return pd.DataFrame(rows)



def _run_result():
    cfg = _make_cfg()
    return evaluate_regimes_from_frames(
        labels_df=_make_labels(),
        returns_df=_make_returns(),
        weights_df=_make_weights(),
        stress_summary_df=_make_stress_summary(),
        turnover_df=_make_turnover(),
        cfg=cfg,
    )



def test_alignment_dates_are_inner_joined() -> None:
    result = _run_result()

    labels_dates = set(pd.to_datetime(_make_labels()["date"]))
    returns = _make_returns()

    for strategy in STRATEGIES:
        ret_dates = set(pd.to_datetime(returns.loc[returns["strategy"] == strategy, "date"]))
        expected_n = len(labels_dates.intersection(ret_dates))
        got_n = int(result.conditional_performance.loc[result.conditional_performance["strategy"] == strategy, "n_obs"].sum())
        assert got_n == expected_n



def test_expected_columns_present() -> None:
    result = _run_result()

    assert list(result.conditional_performance.columns) == COND_PERF_COLUMNS
    assert list(result.conditional_performance_net.columns) == COND_PERF_COLUMNS
    assert list(result.crypto_exposure.columns) == CRYPTO_EXPOSURE_COLUMNS
    assert list(result.drawdown_tail_summary.columns) == STRESS_MAP_COLUMNS



def test_n_obs_positive_by_regime() -> None:
    result = _run_result()
    assert (result.conditional_performance["n_obs"] > 0).all()
    assert (result.conditional_performance_net["n_obs"] > 0).all()



def test_no_crypto_controls_within_tolerance() -> None:
    result = _run_result()
    controls = result.crypto_exposure[
        result.crypto_exposure["strategy"].isin(["minvar_no_crypto_control", "cvar_no_crypto_control"])
    ]
    assert (controls["max_crypto_weight"] <= 1e-6).all()



def test_gross_and_net_are_separate_outputs() -> None:
    result = _run_result()

    merged = result.conditional_performance.merge(
        result.conditional_performance_net,
        on=["strategy", "regime_id", "regime_name"],
        suffixes=("_gross", "_net"),
    )

    assert (merged["ann_return_gross"] != merged["ann_return_net"]).any()



def test_expected_shortfall_sign_and_definition_consistent() -> None:
    result = _run_result()

    row = result.conditional_performance[
        (result.conditional_performance["strategy"] == "minvar_baseline_ch1")
        & (result.conditional_performance["regime_id"] == 1)
    ].iloc[0]

    labels = _make_labels()
    returns = _make_returns()
    aligned = returns.merge(labels, on="date", how="inner")
    subset = aligned[
        (aligned["strategy"] == "minvar_baseline_ch1")
        & (aligned["regime_id"] == 1)
    ]["portfolio_return"]

    expected_es = expected_shortfall_historical(subset, beta=0.95)

    assert row["expected_shortfall"] > 0.0
    assert np.isclose(row["expected_shortfall"], expected_es, atol=1e-12)



def test_stress_window_mapping_no_duplicates_or_loss() -> None:
    result = _run_result()
    mapped = result.drawdown_tail_summary

    dupes = mapped.duplicated(subset=["window_id", "strategy", "scope", "regime_id"]).sum()
    assert dupes == 0

    grouped = mapped.groupby(["window_id", "strategy", "scope"], as_index=False).agg(
        n_obs_sum=("n_obs", "sum"),
        mapped_window_n_obs=("mapped_window_n_obs", "first"),
    )

    assert (grouped["n_obs_sum"] == grouped["mapped_window_n_obs"]).all()
    assert (grouped["mapped_window_n_obs"] > 0).all()
