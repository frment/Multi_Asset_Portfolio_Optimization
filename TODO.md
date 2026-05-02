# TODO — Implementation Roadmap

This file tracks the phased implementation plan for the project.
Each phase has a clear goal, concrete tasks, and a definition of done.

Status legend: `[x]` = done · `[ ]` = pending

---

## Phase 0: Repository and Environment Setup ✅

**Goal:** Establish a clean, reproducible project foundation.

### Tasks
- [x] Create project directory structure
- [x] Write `README.md` with motivation, scope, and roadmap
- [x] Write `TODO.md` (this file)
- [x] Create `config/assets.yaml` with asset universe
- [x] Create `config/settings.yaml` with portfolio parameters
- [x] Write `src/config.py` — YAML config loader
- [x] Write `src/utils.py` — basic utility helpers
- [x] Create `requirements.txt` with essential dependencies
- [x] Create `.gitignore`
- [ ] Initialize git repository
- [ ] Create and document virtual environment setup
- [ ] Run `pytest` with a trivial test to confirm the test harness works

### Definition of Done
The repo can be cloned, the environment set up in under 5 minutes, and all scaffold files are in place. `pytest` runs successfully (even if there are no real tests yet).

---

## Phase 1: Data Ingestion and Processed Dataset ✅

**Goal:** Download historical price data and compute clean return series ready for analysis.

### Tasks
- [x] Create `src/data_loader.py`:
  - Downloads adjusted close prices from Yahoo Finance for all tickers in `config/assets.yaml`
  - Validates ticker coverage and date index integrity
  - Saves raw prices to `data/raw/prices_raw.csv`
- [x] Create `src/preprocessing.py`:
  - Handles missing data (forward-fill weekends/holidays, drop leading NaNs)
  - Computes daily simple returns and daily log returns
  - Saves processed datasets to `data/processed/`
- [x] Create `scripts/run_download.py` — entry point for raw data ingestion
- [x] Create `scripts/run_build_dataset.py` — entry point to build processed return files
- [x] Validate output: `prices_clean.csv` (3 043 × 6), `returns_simple.csv` (3 042 × 6), `returns_log.csv` (3 042 × 6), all with zero missing values
- [ ] Add basic unit tests for preprocessing logic (known input → known output)

### Definition of Done
Both scripts run end-to-end with exit code 0. Reproducible raw and processed datasets exist for all 6 assets from 2018-01-01 to present. Processed files can be reloaded without re-downloading.

---

## Phase 2: Metrics and Benchmark Portfolios ✅

**Goal:** Implement performance metrics and establish baseline portfolios that any optimizer must beat.

### Tasks

**Metrics**
- [x] Create `src/metrics.py` with functions:
  - `annualised_return` — compound annualised return
  - `annualised_volatility` — annualised standard deviation
  - `sharpe_ratio` — excess return / annualised vol (rf = 0 for now)
  - `max_drawdown` — largest peak-to-trough decline
  - `calmar_ratio` — annualised return / |max drawdown|
  - `compute_all_metrics` — convenience wrapper returning a dict

**Benchmarks**
- [x] Create `src/benchmarks.py` with functions:
  - Equal Weight (1/N across all 6 assets)
  - 60/40 proxy (60 % SPY / 40 % TLT)
  - Fixed small-crypto (5 % BTC + 5 % ETH + 22.5 % each SPY/QQQ/GLD/TLT)
  - All weights read from `config/settings.yaml` — no hardcoded values
- [x] Create `scripts/run_benchmarks.py` — builds benchmarks, prints metrics table, saves `data/processed/benchmark_summary.csv`
- [ ] Create `notebooks/02_eda.ipynb` with:
  - Cumulative return plots per asset and per benchmark
  - Full-sample and rolling correlation matrices
  - Rolling volatility, rolling BTC-SPY / ETH-SPY correlations
  - Return distribution plots (histograms, QQ plots)
  - Per-asset and per-benchmark performance summary
- [ ] Add unit tests for metrics functions

### Definition of Done
`scripts/run_benchmarks.py` runs end-to-end, produces a readable metrics table, and saves `benchmark_summary.csv`. EDA notebook is pending (not yet built).

---

## Phase 3: Minimum Variance Optimizer ✅ (static result)

**Goal:** Implement the first constrained portfolio optimizer.

**Why Minimum Variance first?** It avoids relying on expected return estimates, which are notoriously noisy at short lookback horizons. Focusing solely on the covariance structure is a more robust starting point.

**Important caveat:** The current optimizer is applied to the *full sample* as a single static calculation. This is useful for verifying correctness and understanding the optimal allocation, but it is **not** an out-of-sample backtest. The rolling walk-forward version is Phase 4.

### Tasks
- [x] Create `src/optimizer.py`:
  - `estimate_covariance` — annualised sample covariance (note in docstring for future shrinkage)
  - `portfolio_variance` — scalar objective `w'Σw` for scipy
  - `validate_weights` — non-raising constraint checker
  - `minimise_variance` — SLSQP optimizer with all constraints from `config/settings.yaml`:
    - long-only
    - weights sum to 1
    - max weight per asset (0.35)
    - max total crypto weight (0.20)
- [x] Validate optimizer on full-sample data — converges, all constraints satisfied
- [ ] Create `notebooks/03_min_variance.ipynb`:
  - Show full-sample optimal weights
  - Compare static Minimum Variance metrics to benchmarks
  - Visualise resulting allocation
- [ ] Add tests: verify constraints hold, verify output shape, test edge cases

### Definition of Done
`minimise_variance()` runs on the full dataset, reads all constraints from config, converges cleanly, and produces a weight vector that satisfies all constraints. Notebook and tests are pending.

---

## Phase 4: Rolling Walk-Forward Backtest ← **NEXT**

**Goal:** Validate strategies out-of-sample using a rolling window methodology.

This is the step that turns a static optimizer result into a real research finding. The key discipline is strict temporal separation: at each rebalance date, only information available *up to that date* is used.

### Tasks
- [ ] Create `src/backtest.py`:
  - Rolling walk-forward engine: at each monthly rebalance date, fit on the trailing lookback window (252 days by default from config), optimise, hold forward until next rebalance
  - Return a time-series of daily portfolio returns and a history of weights
  - Cover all strategies: Equal Weight, 60/40, Fixed Crypto, Minimum Variance
  - Track turnover at each rebalance
  - Explicit look-ahead bias check (assert weights at `t` use only data ≤ `t`)
- [ ] Create `scripts/run_backtest.py` — entry point to run all strategies and save results
- [ ] Save outputs:
  - `data/processed/backtest_returns.csv` — daily returns per strategy
  - `data/processed/backtest_weights.csv` — weight history per strategy
- [ ] Create `notebooks/04_backtest.ipynb`:
  - Cumulative return curves for all strategies
  - Rolling Sharpe ratio and drawdown over time
  - Weight evolution over time (stacked area chart)
  - Summary performance table (OOS metrics)
- [ ] Add tests for the backtest engine (rebalance dates correct, constraints hold at every step, no future data leakage)

### Definition of Done
Walk-forward backtest produces OOS daily return series for all strategies. Look-ahead bias is explicitly verified. A summary table compares all strategies on the same OOS period. Turnover is reported.

---

## Phase 5: Robustness Analysis

**Goal:** Stress-test findings and assess sensitivity to methodological choices.

### Tasks
- [ ] Sensitivity analysis: lookback window (126, 252, 504 days)
- [ ] Sensitivity analysis: max crypto cap (10 %, 15 %, 20 %, 25 %)
- [ ] Sensitivity analysis: rebalance frequency (weekly, monthly, quarterly)
- [ ] Sub-period analysis (pre-COVID, COVID crash, post-COVID, 2022 bear market)
- [ ] Bootstrap confidence intervals for Sharpe ratio differences
- [ ] Add Ledoit-Wolf shrinkage covariance estimator as an alternative to sample covariance
- [ ] Document findings in `notebooks/05_robustness.ipynb`

### Definition of Done
Results are shown to be (or not to be) robust across parameter choices and sub-periods. Key sensitivities are identified and documented. The covariance estimator is improved beyond naive sample covariance.

---

## Phase 6: Advanced Extensions (Future Work)

**Goal:** Explore more sophisticated portfolio construction methods once the foundation is solid.

### Potential Tasks (not committed)
- [ ] Equal Risk Contribution (ERC / Risk Parity) portfolio
- [ ] Maximum Diversification portfolio
- [ ] CVaR / Expected Shortfall optimization
- [ ] Black-Litterman model with subjective views
- [ ] Regime detection (Hidden Markov Models or threshold rules)
- [ ] Volatility forecasting using GARCH models
- [ ] Dynamic allocation overlays (e.g., trend-following signals)
- [ ] Transaction cost modeling
- [ ] Multi-period optimization
- [ ] Interactive dashboard (Streamlit or similar)

### Definition of Done
Each extension is implemented as a self-contained module with tests, compared against the existing backtested benchmarks, and documented in a dedicated notebook.

---

## Notes

- Phases are sequential; complete the current phase's Definition of Done before starting the next.
- Each phase should end with a clean commit and updated documentation.
- Keep `README.md` in sync as new modules are added.
- Tests are consistently deferred but should be written before the project is considered publication-ready.
