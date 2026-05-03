# Methodological Audit Fixes (v0.4.1)

## 1) Calendar Drift and Mixed 24/7 vs Weekday Assets

Problem:
- Calendar-day preprocessing mixed weekend crypto returns with forward-filled TradFi prices.
- This introduced artificial ETF zero-return rows on weekends and inconsistent annualization semantics.

Fix implemented:
- Added explicit calendar policy with baseline `business_day_aligned`.
- Added `align_prices_to_calendar(...)` in preprocessing.
- Added `dataset_metadata.json` with policy and annualization factor.

Expected impact:
- No weekend rows in baseline processed returns.
- Lookback/rebalance/annualization semantics are coherent with business-day observations.

## 2) Intraperiod Return Model Inconsistency

Problem:
- Daily return path was computed as fixed target weights between monthly rebalances.

Fix implemented:
- Backtest engine now supports:
  - baseline `drifted_buy_and_hold`
  - legacy `constant_target_weights`
- Turnover remains computed against drifted pre-trade weights.

Expected impact:
- Holding-period dynamics and turnover interpretation match monthly rebalance narrative.

## 3) Benchmark Asymmetry

Problem:
- Benchmarks were effectively daily constant-mix while optimized portfolios were monthly walk-forward.

Fix implemented:
- Added benchmark backtest suite with same rebalance conventions and holding-return method.
- Persisted benchmark turnover and weights history.

Expected impact:
- Fairer MinVar/CVaR vs benchmark comparisons.

## 4) Annualization Hardcoding

Problem:
- Metrics relied on fixed 252 in core formulas.

Fix implemented:
- Parameterized annualization in metrics.
- Scripts resolve annualization from dataset metadata/config.

Expected impact:
- Annualized outputs are consistent with the selected calendar policy.

## 5) Chapter 4 Reproducibility (HMM)

Problem:
- Notebook narrative and runtime model could diverge when `hmmlearn` was missing.

Fix implemented:
- Added `hmmlearn` to requirements.
- Added strict fallback behavior:
  - if `primary_model=hmm` and unavailable:
    - raise ImportError when `allow_fallback=false`
    - fallback to KMeans only when explicitly allowed
- Persisted `regime_model_metadata.json` with requested and used model.

Expected impact:
- Chapter 4 is reproducible and self-describing regarding model identity.

## Tests Added

- `tests/test_calendar_policy.py`
- `tests/test_rebalance_dates.py`
- `tests/test_backtest_drifted_returns.py`
- `tests/test_benchmarks_monthly_drifted.py`
- `tests/test_metrics_annualization.py`
- `tests/test_regime_dependencies.py`
- `tests/test_pipeline_smoke.py`

## Before/After Artifacts

- `outputs/audit_fix_comparison.csv`
- `outputs/audit_fix_comparison.md`
