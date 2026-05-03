````markdown
# Crypto-Enhanced Multi-Asset Portfolio Optimization

**A Quantitative Risk-Based Approach**

---

## Project Motivation

The growing institutional interest in digital assets raises a practical portfolio construction question:

> Can a small, disciplined allocation to cryptocurrencies improve the risk-adjusted profile of a traditional multi-asset portfolio once realistic constraints, turnover, costs, robustness checks and tail-risk objectives are imposed?

This project approaches that question from a **quantitative risk management and portfolio construction perspective**, not from a crypto-promotional viewpoint.

The emphasis is on:

- **Risk-based portfolio construction**, not return chasing
- **Realistic constraints**: long-only, position caps, crypto sleeve caps
- **Strong benchmarks**: equal weight, 60/40 proxy, fixed-crypto baselines and no-crypto controls
- **Walk-forward out-of-sample validation**
- **Robustness analysis** across lookbacks, crypto caps, rebalance frequencies, covariance estimators and costs
- **Tail-risk analysis** through historical CVaR / Expected Shortfall optimization
- **Transparent limitations and reproducibility**

The goal is to produce a research-grade portfolio study, not a promotional argument for crypto.

---

## Research Question

> Does a bounded and risk-disciplined allocation to BTC and ETH improve the risk-adjusted performance of a traditional multi-asset portfolio when we impose realistic constraints, compare against strong benchmarks, account for turnover/costs, and validate out-of-sample?

The project treats this as a falsifiable research question. A positive result must survive robustness checks, implementation frictions and downside-risk analysis. A negative or fragile result is also informative.

---

## Professional Framing

This repository is structured as a **quantitative portfolio research project**. It prioritizes:

- Out-of-sample portfolio evaluation
- Robustness and sensitivity analysis
- Tail-risk and drawdown-aware evaluation
- Turnover and transaction-cost awareness
- Conservative interpretation of crypto exposure
- Modular, testable, reproducible code
- Notebook-based research reporting in mini-paper style

The intended audience is a quant researcher, portfolio manager, risk specialist, recruiter or collaborator who wants to understand both the research design and the implementation quality.

---

## Current Status

The project has moved beyond MVP scaffold stage.

### Chapter 1 — Baseline OOS Portfolio Construction

Completed.

Implemented:

- Data ingestion and preprocessing
- Daily return construction
- Benchmark portfolios
- Constrained Minimum Variance optimizer
- Rolling walk-forward out-of-sample backtest
- Turnover reporting
- Baseline performance analysis
- Crypto sleeve diagnostics
- Notebook mini-paper:

```text
notebooks/01_backtest_analysis.ipynb
````

### Chapter 2 — Robustness, Costs and Statistical Confidence

Completed / mature.

Implemented:

* One-factor-at-a-time robustness analysis
* No-crypto control
* Lookback sensitivity: 126 / 252 / 504 days
* Crypto cap sensitivity: 0% / 10% / 20% / 25%
* Rebalance frequency sensitivity: monthly / quarterly
* Gross vs net simple transaction-cost layer
* Cost scenarios: 0 / 10 / 25 / 50 bps
* Sample covariance vs Ledoit-Wolf shrinkage
* Bootstrap / confidence layer for pre-registered comparisons
* Common-family sample logic
* Notebook mini-paper:

```text
notebooks/02_robustness_analysis.ipynb
```

### Chapter 3 — Tail Risk and Alternative Risk Objectives

Implemented.

Implemented:

* Generic rolling backtest engine refactor
* Backward-compatible `run_min_variance_backtest` wrapper
* Historical minimum-CVaR / Expected Shortfall optimizer
* MinVar vs CVaR comparison under identical OOS protocol
* No-crypto controls for both objective families
* Historical stress testing
* Gross vs net tail-risk outputs
* Tail-risk metrics:

  * Expected Shortfall
  * Return / ES
  * Max Drawdown
  * Calmar
* Crypto sleeve usage diagnostics
* Notebook mini-paper:

```text
notebooks/03_tail_risk_cvar_analysis.ipynb
```

Key Chapter 3 sanity checks:

* Baseline MinVar outputs are exactly preserved after the generic backtest refactor.
* MinVar and CVaR are compared on the same OOS window:

  * 2018-10-01 to 2026-05-02
  * 2771 observations
* Stress outputs separate `gross` and `net` scopes.
* Crypto exposure remains mostly intermittent rather than structural:

  * MinVar uses crypto sparingly.
  * CVaR uses crypto somewhat more, but not as a permanent allocation.

---

## Current Research Interpretation

The project’s conclusions are deliberately conservative.

At the current stage, the evidence is best read as follows:

* The constrained Minimum Variance baseline is reproducible and implementable.
* The role of crypto is not a simple “always allocate” story.
* Crypto exposure appears more episodic than structural under risk-based optimization.
* Robustness checks matter: results are sensitive to assumptions such as crypto cap, lookback, rebalance frequency and costs.
* Ledoit-Wolf shrinkage is primarily relevant as a stability / regularization tool, not necessarily as a pure performance enhancer.
* CVaR provides a useful alternative risk objective, but must be evaluated against return, drawdown, turnover and net-of-cost trade-offs.
* Historical stress windows are diagnostic, not universal proof.

The project does not claim that crypto is always beneficial. It asks when, how much, and under what constraints crypto survives as a portfolio allocation.

---

## Current Build Focus

### Chapter 4 — Regime Analysis

Completed.

Implemented:

* Regime feature engineering from existing daily returns
* Candidate detection stack (KMeans / HMM) with HMM-2 as primary diagnostic model
* Regime persistence and transition matrix analysis
* Regime-conditional gross/net performance attribution
* MinVar vs CVaR comparison by regime
* Crypto sleeve diagnostics by regime
* Stress windows mapped to detected regimes
* Notebook mini-paper:

```text
notebooks/04_regime_analysis.ipynb
```

Methodological caveat:

* Chapter 4 is diagnostic and in-sample attribution.
* It is not a forecasting, signaling or dynamic-allocation chapter.

### Immediate Next Build

### Chapter 5 — Supervised Risk Forecasting / Overlay

Next.

Objective:

> Test whether risk forecasting signals provide robust out-of-sample value and, only if they do, evaluate a conservative overlay versus static MinVar/CVaR references.

Initial scope:

* Supervised risk targets defined ex-ante
* Walk-forward validation without look-ahead
* Calibration/stability first, Sharpe second
* Conservative overlay rules (if signal quality justifies it)

Not in scope yet:

* Aggressive tactical timing
* Regime-trading claims
* Unconstrained dynamic crypto caps

---

## Asset Universe

| Ticker  | Role                  | Rationale                                           |
| ------- | --------------------- | --------------------------------------------------- |
| BTC-USD | Crypto sleeve         | Largest cryptocurrency by market cap                |
| ETH-USD | Crypto sleeve         | Second-largest crypto asset; different risk profile |
| SPY     | Equity risk           | Broad US equity proxy                               |
| QQQ     | Growth equity risk    | US tech-heavy equity exposure                       |
| GLD     | Defensive diversifier | Gold ETF; defensive / inflation-sensitive exposure  |
| TLT     | Duration risk         | Long-duration US Treasury bond proxy                |

This is a deliberately compact multi-asset universe covering:

* crypto
* equities
* growth equities
* gold
* duration / rates

The compact universe keeps the research interpretable while still allowing meaningful cross-asset portfolio construction.

---

## Core Portfolio Assumptions

| Parameter               | Value            |
| ----------------------- | ---------------- |
| Data frequency          | Daily            |
| Start date              | 2018-01-01       |
| Baseline rebalance      | Monthly          |
| Baseline lookback       | 252 trading days |
| Portfolio type          | Long-only        |
| Weight constraint       | Sum to 1         |
| Max weight per asset    | 35%              |
| Max total crypto weight | 20%              |

These constraints are intentionally conservative. They are designed to avoid unrealistic concentration and to keep crypto exposure bounded.

---

## Benchmark and Strategy Set

The project evaluates several portfolio families and controls.

### Benchmarks

1. **Equal Weight**

   * Allocates equally across all assets.
   * Simple but hard to beat out-of-sample.

2. **60/40 Proxy**

   * 60% SPY / 40% TLT.
   * Traditional institutional reference point.

3. **Fixed Small-Crypto**

   * Hand-specified small allocation to BTC and ETH.
   * Useful to separate “crypto exposure” from optimization.

### Optimization-Based Strategies

1. **Minimum Variance**

   * First risk-based optimizer.
   * Avoids explicit expected-return forecasts.

2. **Minimum Variance, No-Crypto Control**

   * Same optimizer with crypto cap set to zero.
   * Separates optimizer effect from crypto availability.

3. **Historical Minimum-CVaR**

   * Tail-risk objective based on empirical historical scenarios.
   * Focused on Expected Shortfall rather than variance.

4. **Historical Minimum-CVaR, No-Crypto Control**

   * Same CVaR objective without crypto exposure.
   * Used to isolate the crypto sleeve effect under a tail-risk objective.

---

## Methodology Roadmap

### Chapter 1 — Baseline

Completed.

* Data ingestion
* Return construction
* Benchmarks
* Constrained Minimum Variance
* Rolling OOS backtest
* Turnover
* Baseline notebook

### Chapter 2 — Robustness and Implementability

Completed.

* Robustness one-factor-at-a-time
* No-crypto control
* Lookback sensitivity
* Crypto cap sensitivity
* Rebalance frequency sensitivity
* Gross vs net cost layer
* Ledoit-Wolf covariance comparison
* Bootstrap confidence layer
* Robustness notebook

### Chapter 3 — Tail Risk and Alternative Objectives

Implemented.

* Historical CVaR / Expected Shortfall optimization
* MinVar vs CVaR comparison
* No-crypto controls
* Historical stress testing
* Tail-risk metrics
* Crypto sleeve usage under CVaR
* Tail-risk notebook

### Chapter 4 — Regime Analysis

Completed.

* Regime features
* KMeans / HMM regime candidates
* Regime persistence and transition matrix
* Conditional performance (gross and net)
* MinVar vs CVaR by regime
* Crypto sleeve by regime
* Stress windows mapped to regimes
* Regime notebook

### Chapter 5 — Supervised Risk Forecasting / Overlay

Future.

Potential scope:

* Volatility forecasting
* Crash probability
* Risk-aware overlay
* Dynamic de-risking rules

Not implemented yet.

### Premium / Final Extensions

Future.

Potential scope:

* Black-Litterman
* Resampling
* Dashboard
* Richer cost/slippage model
* Expanded asset universe

---

## Repository Structure

Current high-level structure:

```text
./
├── README.md
├── TODO.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── assets.yaml
│   ├── settings.yaml
│   ├── robustness.yaml
│   ├── tail_risk.yaml
│   └── regime_analysis.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── benchmarks.py
│   ├── optimizer.py
│   ├── backtest.py
│   ├── costs.py
│   ├── bootstrap.py
│   ├── covariance.py
│   ├── robustness.py
│   ├── cvar_optimizer.py
│   ├── stress.py
│   ├── regime_features.py
│   ├── regime_detection.py
│   └── regime_evaluation.py
├── scripts/
│   ├── run_download.py
│   ├── run_build_dataset.py
│   ├── run_benchmarks.py
│   ├── run_optimizer.py
│   ├── run_backtest.py
│   ├── run_robustness.py
│   ├── run_statistical_confidence.py
│   ├── run_tail_risk.py
│   └── run_regime_analysis.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── robustness/
│       ├── tail_risk/
│       └── regime_analysis/
├── notebooks/
│   ├── 01_backtest_analysis.ipynb
│   ├── 02_robustness_analysis.ipynb
│   ├── 03_tail_risk_cvar_analysis.ipynb
│   └── 04_regime_analysis.ipynb
└── tests/
    ├── test_backtest.py
    ├── test_bootstrap.py
    ├── test_cvar_optimizer.py
   ├── test_optimizer_covariance.py
    ├── test_robustness.py
    ├── test_stress.py
   ├── test_tail_risk_metrics.py
   ├── test_regime_features.py
   ├── test_regime_detection.py
   └── test_regime_evaluation.py
```

Design note:

* Core logic lives in `src/`.
* Scripts orchestrate reproducible runs.
* Notebooks are for analysis and interpretation.
* Configuration lives in YAML.
* Tests protect critical invariants and refactors.

---

## Key Outputs

### Baseline Outputs

```text
data/processed/prices_clean.csv
data/processed/returns_simple.csv
data/processed/returns_log.csv
data/processed/portfolio_returns.csv
data/processed/weights_history.csv
data/processed/turnover_history.csv
data/processed/backtest_summary.csv
```

### Robustness Outputs

```text
data/processed/robustness/robustness_summary.csv
data/processed/robustness/robustness_summary_gross.csv
data/processed/robustness/robustness_summary_net.csv
data/processed/robustness/robustness_summary_common_family.csv
data/processed/robustness/robustness_summary_common_family_net.csv
data/processed/robustness/robustness_returns.csv
data/processed/robustness/robustness_metadata.csv
data/processed/robustness/robustness_weights_panel.csv
data/processed/robustness/robustness_turnover_panel.csv
data/processed/robustness/confidence_summary.csv
```

### Tail-Risk Outputs

```text
data/processed/tail_risk/tail_risk_summary.csv
data/processed/tail_risk/tail_risk_summary_net.csv
data/processed/tail_risk/tail_risk_returns.csv
data/processed/tail_risk/tail_risk_weights_panel.csv
data/processed/tail_risk/tail_risk_turnover_panel.csv
data/processed/tail_risk/stress_summary.csv
```

### Regime Outputs

```text
data/processed/regime_analysis/regime_features.csv
data/processed/regime_analysis/regime_labels.csv
data/processed/regime_analysis/regime_model_summary.csv
data/processed/regime_analysis/regime_transition_matrix.csv
data/processed/regime_analysis/regime_conditional_performance.csv
data/processed/regime_analysis/regime_conditional_performance_net.csv
data/processed/regime_analysis/regime_crypto_exposure.csv
data/processed/regime_analysis/regime_drawdown_tail_summary.csv
```

---

## Reproducibility

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the pipeline:

```bash
python scripts/run_download.py
python scripts/run_build_dataset.py
python scripts/run_benchmarks.py
python scripts/run_optimizer.py
python scripts/run_backtest.py
```

Run robustness analysis:

```bash
python scripts/run_robustness.py
python scripts/run_statistical_confidence.py
```

Run tail-risk analysis:

```bash
python scripts/run_tail_risk.py
```

Run regime analysis:

```bash
python scripts/run_regime_analysis.py
```

Run tests:

```bash
python -m pytest -q
```

---

## Design Principles

1. **Modularity**

   * Each concern has its own module: data, preprocessing, metrics, optimization, backtesting, costs, robustness and tail risk.

2. **Config-driven research**

   * Parameters live in YAML rather than being scattered through notebooks.

3. **Pure functions first**

   * Prefer small, testable functions over hidden notebook state.

4. **Walk-forward discipline**

   * Portfolio weights are estimated using historical information only.

5. **Benchmark discipline**

   * Every optimizer is compared against simple baselines and controls.

6. **Risk-first interpretation**

   * Sharpe is not enough. Drawdown, Expected Shortfall, turnover and crypto exposure matter.

7. **Honest conclusions**

   * The project documents fragility, limitations and negative evidence.

---

## Important Limitations and Disclaimer

* This is a research and learning project. It is **not investment advice**.
* Results are based on historical data and may not generalize.
* Crypto has limited history and materially different risk characteristics from traditional assets.
* Data comes primarily from Yahoo Finance and is not institutional-grade.
* Transaction costs are currently modeled as simple proportional turnover costs.
* Slippage, market impact, borrow costs, taxes and liquidity constraints are not fully modeled.
* CVaR is estimated empirically from historical windows; no EVT or parametric tail model is used yet.
* Stress windows are diagnostic and historical, not predictive.
* Regime analysis is planned next but not yet implemented.
* This remains a research codebase, not a production portfolio management system.

---

## License

This project is for educational and research purposes.

```
```
