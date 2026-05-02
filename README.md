# Crypto-Enhanced Multi-Asset Portfolio Optimization

**A Quantitative Risk-Based Approach**

---

## Project Motivation

The growing institutional interest in digital assets raises a practical portfolio construction question: can a small, disciplined allocation to cryptocurrencies improve the risk-adjusted performance of a traditional multi-asset portfolio?

This project approaches that question from a **quantitative risk management perspective**, not from a crypto-enthusiast viewpoint. The emphasis is on:

- **Risk-based portfolio construction** (not return chasing)
- **Realistic constraints** (position limits, crypto caps, long-only)
- **Strong benchmarks** (equal weight, 60/40 proxy, fixed-crypto baselines)
- **Robust out-of-sample validation** (walk-forward backtesting)
- **Reproducibility** (config-driven, version-controlled, modular code)

The goal is to produce a research-grade answer, not a promotional argument.

---

## Research Question

> Does a bounded and risk-disciplined allocation to BTC and ETH improve the risk-adjusted performance of a traditional multi-asset portfolio, when we impose realistic constraints, compare against strong benchmarks, and validate out-of-sample?

---

## Professional Framing

This project is structured as a **quantitative portfolio research study**. It prioritizes:

- Sharpe ratio improvements **and** tail risk metrics (CVaR, max drawdown)
- Turnover and transaction cost awareness
- Benchmark-relative evaluation (not absolute returns in isolation)
- Transparent methodology with documented assumptions
- Interview-ready code and documentation

The intent is that a recruiter, portfolio manager, or collaborator can open this repository and quickly understand the research design, current status, and planned next steps.

---

## Current Scope

**Chapter 1 is complete** as the reproducible OOS baseline.

**Chapter 2 is now closed** for robustness, turnover and implementability, simple net-of-cost analysis, covariance stability (sample vs Ledoit-Wolf), a light bootstrap confidence layer, and research-grade reporting notebooks.

What exists today:

- End-to-end data ingestion and preprocessing pipeline
- Risk metrics and benchmark portfolio framework
- Constrained Minimum Variance optimizer (config-driven constraints)
- Rolling walk-forward out-of-sample backtest workflow
- **Turnover reporting**: pre-trade drifted one-way turnover at every rebalance, saved to `turnover_history.csv`
- **Automated test suite** for the backtest engine (rebalance dates, temporal separation, constraint checking, turnover correctness)
- Reproducible scripts to run each stage of the pipeline
- Analysis notebook for baseline OOS evidence (equity curves, drawdowns, weights, quantified interpretation, crypto activation diagnostics)
- Configuration-first architecture with modular, testable functions

What does **not** exist yet:

- Full regime-conditional modeling and richer subperiod inference
- Asset-specific execution/slippage/impact cost modeling
- Risk-space decomposition (for example, marginal contribution to variance)
- Unit tests for preprocessing and metrics modules

The project has moved beyond scaffold stage and now has a closed Chapter 2 with conservative conclusions and explicit limits.

---

## Why Minimum Variance First

Minimum Variance is the first optimizer in this project for practical reasons:

- Expected return estimates are noisy and unstable, especially for short and volatile histories.
- Covariance-driven optimization is usually a more robust starting point than mean-variance with aggressive return assumptions.
- It provides a clean way to validate the data pipeline, portfolio constraints, and rolling walk-forward backtest engine before adding more advanced methods.

The objective is not to claim optimality early. It is to establish a reliable baseline optimizer that can be stress-tested and improved later.

---

## Immediate Next Build

Chapter 3 priorities:

1. Regime and subperiod structure as a first-class modeling layer
2. Expanded confirmatory design (pre-registered contrasts and multiple-testing discipline)
3. Richer implementation layer beyond a flat bps wedge
4. Next objective-family extensions only after the above diagnostics

---

## Initial Asset Universe

| Ticker   | Role                          | Rationale                                        |
|----------|-------------------------------|--------------------------------------------------|
| BTC-USD  | Crypto sleeve                 | Largest cryptocurrency by market cap              |
| ETH-USD  | Crypto sleeve                 | Second-largest; different risk profile than BTC   |
| SPY      | Equity risk                   | Broad US equity market proxy (S&P 500)            |
| QQQ      | Equity risk                   | US tech-heavy equity exposure (Nasdaq-100)        |
| GLD      | Defensive diversifier         | Gold ETF; traditional safe-haven / inflation hedge|
| TLT      | Defensive rates proxy         | Long-duration US Treasury bonds                   |

**Why these assets?**

- BTC and ETH represent the two most liquid, institutionally relevant cryptocurrencies.
- SPY and QQQ capture broad and growth-tilted US equity risk.
- GLD provides commodity/defensive diversification.
- TLT introduces duration risk and acts as a rates hedge in risk-off environments.

This is a minimal but meaningful multi-asset universe that covers equities, crypto, commodities, and fixed income.

---

## Initial Portfolio Assumptions

| Parameter                  | Value              |
|----------------------------|--------------------|
| Data frequency             | Daily              |
| Start date                 | 2018-01-01         |
| Rebalance frequency        | Monthly            |
| Rolling lookback window    | 252 trading days   |
| Portfolio type             | Long-only          |
| Weights constraint         | Sum to 1           |
| Max weight per asset       | 35%                |
| Max total crypto weight    | 20%                |

These constraints are intentionally conservative. They reflect realistic institutional-style limits and prevent the optimizer from concentrating into volatile assets.

---

## Initial Benchmark Set

Before implementing any optimizer, the project establishes strong baselines:

1. **Equal Weight (1/N):** Allocates equally across all assets. A surprisingly hard benchmark to beat in practice.
2. **60/40 Proxy:** 60% SPY + 40% TLT. The classic institutional baseline.
3. **Fixed Small-Crypto:** A hand-specified portfolio with a small fixed allocation to BTC and ETH (e.g., 5% BTC, 5% ETH, rest split among traditional assets).
4. **Minimum Variance:** The first optimization-based portfolio. Minimizes portfolio variance subject to constraints.

Any proposed strategy must demonstrably outperform these baselines on risk-adjusted metrics to be considered meaningful.

---

## Planned Methodology Roadmap

The project is designed to evolve through clear phases:

### Near-term (completed in Chapter 1)
- Data ingestion from Yahoo Finance
- Return computation and cleaned datasets
- Implementation of benchmark portfolios
- Minimum Variance optimization with constraints
- Rolling walk-forward backtest framework
- Turnover reporting (pre-trade drifted one-way turnover at every rebalance)
- Automated backtest engine test suite
- Initial analysis notebook for OOS performance interpretation

### Medium-term (Chapter 2: robustness and depth)
- Equal Risk Contribution (ERC / Risk Parity)
- Maximum Diversification portfolio
- CVaR / Expected Shortfall optimization
- Sensitivity analysis (different lookback windows, constraint levels, rebalance frequencies)
- Bootstrap and Monte Carlo robustness checks

### Longer-term (advanced extensions)
- Black-Litterman model with subjective views
- Regime detection (HMM or threshold-based)
- Volatility forecasting (GARCH family)
- Dynamic allocation overlays
- Transaction cost modeling

These are documented for transparency. They are **not yet implemented**.

---

## Repository Structure (Current)

```
./
├── README.md
├── TODO.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── assets.yaml
│   └── settings.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── benchmarks.py
│   ├── optimizer.py
│   └── backtest.py
├── scripts/
│   ├── run_download.py
│   ├── run_build_dataset.py
│   ├── run_benchmarks.py
│   ├── run_optimizer.py
│   └── run_backtest.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_backtest_analysis.ipynb
└── tests/
```

**Design note:** Business logic lives in `src/`. Scripts orchestrate repeatable tasks. Notebooks are for analysis and narrative interpretation. Configuration is externalized to YAML files, not hardcoded.

---

## What Is Present in Chapter 1

- [x] Data download pipeline and raw market snapshot persistence
- [x] Price cleaning and return engineering (simple and log)
- [x] Performance metrics module (return, volatility, Sharpe, drawdown, Calmar)
- [x] Benchmark portfolio engine and summary outputs
- [x] Constrained Minimum Variance optimizer
- [x] Rolling OOS walk-forward validation for Minimum Variance
- [x] Benchmark comparison on aligned OOS period
- [x] Saved outputs for returns, weights history, and summary metrics
- [x] Analysis notebook documenting interpretation and limitations
- [x] Chapter 1 frozen as reproducible baseline + descriptive OOS evidence + gross implementability

---

## What Will Be Built Next

The immediate next steps are now Chapter 3 / next methodological block (see `TODO.md`):

1. Regime-conditional robustness and subperiod diagnostics
2. Extended confirmatory layer with explicit error-rate control
3. More realistic implementation-cost modeling
4. Additional objective families only after the previous layers are stable

---

## Design Principles

1. **Modularity:** Each concern (data, optimization, backtesting) gets its own module.
2. **Config-driven:** Parameters live in YAML, not scattered through code.
3. **Pure functions first:** Prefer stateless functions over classes unless classes are clearly warranted.
4. **Readability over cleverness:** Code should be understandable by a motivated beginner.
5. **Honest documentation:** Never claim something is done that isn't. Document limitations.
6. **Reproducibility:** Anyone should be able to clone this repo, install dependencies, and reproduce results.
7. **Benchmark discipline:** No strategy is meaningful unless it beats simple baselines.

---

## Important Limitations and Disclaimer

- **This is a research and learning project.** It is not investment advice.
- **Results are based on historical data.** Past performance does not predict future returns.
- **Transaction costs, slippage, and market impact are not modeled in the MVP.** Real-world implementation would require these.
- **The crypto market is highly volatile and has limited history.** Conclusions drawn from 2018–present may not generalize.
- **Data sourced from Yahoo Finance** may contain errors, adjusted-price artifacts, or gaps. It is not institutional-grade data.
- **This remains a research codebase, not a production system.** Chapter 2 is closed, but advanced regime, inference, and execution modeling remain future work.

---

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd Multi_Asset_Portfolio_Optimization

# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## License

This project is for educational and research purposes.
