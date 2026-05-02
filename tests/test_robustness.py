"""Tests for Chapter 2 first-pass robustness utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.costs import (
    apply_rebalance_costs_to_daily_returns,
    bps_to_rate,
    build_rebalance_cost_series,
)
from src.robustness import (
    SUMMARY_COLUMNS,
    build_first_pass_specs,
    run_first_pass_robustness,
)


def _make_synthetic_returns(
    n_days: int = 900,
    tickers: list[str] | None = None,
    seed: int = 7,
) -> pd.DataFrame:
    if tickers is None:
        tickers = ["A", "B", "C", "D"]
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2018-01-02", periods=n_days)
    data = rng.normal(loc=0.0003, scale=0.012, size=(n_days, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def _minimal_optimizer_config() -> dict:
    return {
        "tickers": ["A", "B", "C", "D"],
        "crypto_assets": [],
        "long_only": True,
        "max_weight": 1.0,
        "max_crypto_weight": 0.20,
    }


def test_first_pass_specs_include_anchors_and_no_crypto_cap():
    specs = build_first_pass_specs(
        base_lookback_window_days=252,
        base_max_total_crypto_weight=0.20,
        base_rebalance_frequency="monthly",
        include_weekly_in_first_pass=False,
    )
    ids = {spec.experiment_id for spec in specs}

    assert "baseline_ch1" in ids
    assert "minvar_no_crypto_control" in ids
    assert "crypto_cap_0_00" in ids
    assert "rebalance_weekly" not in ids


def test_robustness_outputs_have_expected_schema():
    returns = _make_synthetic_returns()
    outputs = run_first_pass_robustness(
        returns=returns,
        base_optimizer_config=_minimal_optimizer_config(),
        risk_free_rate=0.0,
        base_lookback_window_days=252,
        base_rebalance_frequency="monthly",
        include_weekly_in_first_pass=False,
    )

    summary = outputs["robustness_summary"]
    returns_panel = outputs["robustness_returns"]
    metadata = outputs["robustness_metadata"]
    weights_panel = outputs["robustness_weights_panel"]
    turnover_panel = outputs["robustness_turnover_panel"]
    common_family = outputs["robustness_summary_common_family"]
    net_summary = outputs["robustness_summary_net"]

    for column in SUMMARY_COLUMNS:
        assert column in summary.columns

    assert {"experiment_id", "date", "portfolio_return"}.issubset(set(returns_panel.columns))
    assert {"experiment_id", "rebalance_date", "ticker", "weight"}.issubset(set(weights_panel.columns))
    assert {"experiment_id", "rebalance_date", "turnover_one_way"}.issubset(set(turnover_panel.columns))
    assert {"experiment_id", "benchmark_policy"}.issubset(set(metadata.columns))
    assert "covariance_method" in summary.columns
    assert "covariance_method" in returns_panel.columns
    assert "covariance_method" in weights_panel.columns
    assert "covariance_method" in turnover_panel.columns
    assert "covariance_method" in metadata.columns

    assert len(summary) > 0
    assert len(returns_panel) > 0
    assert len(weights_panel) > 0
    assert len(turnover_panel) > 0

    # Family-level common sample summaries should be available for Chapter 2 comparison.
    assert "sample_scope" in common_family.columns
    assert set(common_family["sample_scope"].unique()) <= {"common_family"}

    if not common_family.empty:
        assert "covariance_method" in common_family.columns

    net_required = {
        "experiment_id",
        "cost_bps",
        "cumulative_cost",
        "ann_return_gross",
        "ann_return_net",
        "sharpe_gross",
        "sharpe_net",
        "max_drawdown_net",
        "calmar_net",
    }
    assert net_required.issubset(set(net_summary.columns))
    assert "covariance_method" in net_summary.columns


def test_turnover_zero_implies_zero_cost():
    dates = pd.bdate_range("2024-01-01", periods=5)
    gross = pd.Series(0.001, index=dates)
    turnover = pd.Series(0.0, index=dates)

    costs = build_rebalance_cost_series(turnover, cost_rate=bps_to_rate(25.0))
    net, daily_costs = apply_rebalance_costs_to_daily_returns(gross, costs)

    assert float(costs.sum()) == 0.0
    assert float(daily_costs.sum()) == 0.0
    pd.testing.assert_series_equal(net, gross)


def test_higher_bps_reduces_net_return_for_same_turnover_path():
    dates = pd.bdate_range("2024-01-01", periods=10)
    gross = pd.Series(0.001, index=dates)
    turnover = pd.Series(0.02, index=dates[::2])

    costs_low = build_rebalance_cost_series(turnover, cost_rate=bps_to_rate(10.0))
    costs_high = build_rebalance_cost_series(turnover, cost_rate=bps_to_rate(50.0))

    net_low, _ = apply_rebalance_costs_to_daily_returns(gross, costs_low)
    net_high, _ = apply_rebalance_costs_to_daily_returns(gross, costs_high)

    assert net_high.sum() < net_low.sum()


def test_baseline_monthly_anchor_present_and_monthly():
    returns = _make_synthetic_returns()
    outputs = run_first_pass_robustness(
        returns=returns,
        base_optimizer_config=_minimal_optimizer_config(),
        risk_free_rate=0.0,
        base_lookback_window_days=252,
        base_rebalance_frequency="monthly",
        include_weekly_in_first_pass=False,
    )
    summary = outputs["robustness_summary"]

    baseline_row = summary.loc[summary["experiment_id"] == "baseline_ch1"].iloc[0]
    assert baseline_row["rebalance_frequency"] == "monthly"
    assert int(baseline_row["lookback_window_days"]) == 252


def test_covariance_method_family_present_and_ledoit_pipeline_runs():
    returns = _make_synthetic_returns()
    outputs = run_first_pass_robustness(
        returns=returns,
        base_optimizer_config=_minimal_optimizer_config(),
        risk_free_rate=0.0,
        base_lookback_window_days=252,
        base_rebalance_frequency="monthly",
        include_weekly_in_first_pass=False,
        covariance_methods=["sample", "ledoit_wolf"],
        include_no_crypto_anchor_in_covariance_family=False,
    )

    summary = outputs["robustness_summary"]
    cov_family = summary.loc[summary["family"] == "covariance_method"].copy()

    assert not cov_family.empty
    assert set(cov_family["covariance_method"].unique()) == {"sample", "ledoit_wolf"}


def test_rebalance_dates_in_panels_belong_to_operational_index():
    returns = _make_synthetic_returns()
    outputs = run_first_pass_robustness(
        returns=returns,
        base_optimizer_config=_minimal_optimizer_config(),
        risk_free_rate=0.0,
        base_lookback_window_days=252,
        base_rebalance_frequency="monthly",
        include_weekly_in_first_pass=False,
    )

    operational_dates = set(returns.index)
    weights_rebalance_dates = set(pd.to_datetime(outputs["robustness_weights_panel"]["rebalance_date"]))
    turnover_rebalance_dates = set(pd.to_datetime(outputs["robustness_turnover_panel"]["rebalance_date"]))

    assert weights_rebalance_dates.issubset(operational_dates)
    assert turnover_rebalance_dates.issubset(operational_dates)


def test_common_family_covariance_method_summary_consistent():
    returns = _make_synthetic_returns()
    outputs = run_first_pass_robustness(
        returns=returns,
        base_optimizer_config=_minimal_optimizer_config(),
        risk_free_rate=0.0,
        base_lookback_window_days=252,
        base_rebalance_frequency="monthly",
        include_weekly_in_first_pass=False,
        covariance_methods=["sample", "ledoit_wolf"],
        include_no_crypto_anchor_in_covariance_family=False,
    )

    common_family = outputs["robustness_summary_common_family"]
    cov_common = common_family.loc[common_family["family"] == "covariance_method"]

    assert not cov_common.empty
    assert set(cov_common["covariance_method"].unique()) == {"sample", "ledoit_wolf"}
    assert set(cov_common["sample_scope"].unique()) == {"common_family"}
