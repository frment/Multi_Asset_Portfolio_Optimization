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

**This repository is currently in MVP foundation stage.**

What exists today:

- A clean, modular project structure
- Configuration files defining the asset universe, constraints, and settings
- Core scaffold code (`src/`) with config loaders and basic utilities
- A detailed roadmap (`TODO.md`) with phased implementation plans
- This README documenting motivation, design, and scope

What does **not** exist yet:

- Data ingestion pipeline
- Exploratory data analysis
- Portfolio optimizers
- Backtesting engine
- Results, charts, or performance tables

This is an honest starting point. The foundation is deliberately kept simple so each subsequent layer can be built, tested, and understood incrementally.

---

## Why Minimum Variance First

Minimum Variance is the first optimizer in this project for practical reasons:

- Expected return estimates are noisy and unstable, especially for short and volatile histories.
- Covariance-driven optimization is usually a more robust starting point than mean-variance with aggressive return assumptions.
- It provides a clean way to validate the data pipeline, portfolio constraints, and rolling walk-forward backtest engine before adding more advanced methods.

The objective is not to claim optimality early. It is to establish a reliable baseline optimizer that can be stress-tested and improved later.

---

## Immediate Next Build

The next implementation targets are:

1. Market data ingestion
2. Clean processed return series
3. Benchmark portfolios
4. Constrained Minimum Variance optimizer
5. First rolling walk-forward backtest

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

### Near-term (MVP development)
- Data ingestion from Yahoo Finance
- Return computation and basic EDA
- Implementation of benchmark portfolios
- Minimum Variance optimization with constraints
- Rolling walk-forward backtest framework

### Medium-term (robustness and depth)
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
├── README.md                 # This file
├── TODO.md                   # Phased implementation roadmap
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── config/
│   ├── assets.yaml           # Asset universe and categories
│   └── settings.yaml         # Portfolio constraints and parameters
├── src/
│   ├── __init__.py           # Package marker
│   ├── config.py             # Config loading helpers
│   └── utils.py              # General-purpose utilities
├── data/
│   ├── raw/                  # Market data snapshots (created, initially empty)
│   └── processed/            # Cleaned prices/returns (created, initially empty)
├── notebooks/                # Research notebooks (created, initially empty)
├── scripts/                  # Reproducible run scripts (created, initially empty)
└── tests/                    # Unit/integration tests (created, initially empty)
```

**Design note:** Business logic lives in `src/`. Scripts orchestrate repeatable tasks. Notebooks are for exploration and validation only. Configuration is externalized to YAML files, not hardcoded.

---

## What Is Present in This First Layer

- [x] Project README with motivation, scope, and roadmap
- [x] `TODO.md` with concrete phased tasks
- [x] `config/assets.yaml` — asset universe definition
- [x] `config/settings.yaml` — portfolio parameters and constraints
- [x] `src/config.py` — YAML config loader
- [x] `src/utils.py` — basic utility helpers
- [x] `requirements.txt` — essential Python dependencies
- [x] `.gitignore` — standard Python ignores
- [x] `data/raw/` and `data/processed/` directories (empty, for upcoming ingestion)
- [x] `notebooks/`, `scripts/`, and `tests/` directories (empty scaffolding)

---

## What Will Be Built Next

The immediate next steps (Phase 1 in `TODO.md`):

1. Set up a virtual environment and install dependencies
2. Write `src/data_loader.py` to download price data via `yfinance`
3. Write `src/preprocessing.py` to clean prices and compute return series
4. Save processed data to `data/processed/`
5. Add reproducible scripts: `scripts/run_download.py` and `scripts/run_build_dataset.py`
6. Use a notebook later for validation and visualization

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
- **This is an MVP foundation.** Many planned features are not yet built. The roadmap reflects intent, not current capability.

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
