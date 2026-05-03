# Methodology (v0.4.1)

## Calendar

Baseline:
- Policy: `business_day_aligned`.
- Keep dates with observed prices for all TradFi assets (`SPY`, `QQQ`, `GLD`, `TLT`).
- Sample crypto (`BTC-USD`, `ETH-USD`) on those same dates.
- Baseline annualization factor: `252`.

Sensitivity:
- Policy: `calendar_day`.
- TradFi forward-fill allowed explicitly.
- Annualization factor: `365.25`.

## Rebalancing

- Frequency: monthly baseline.
- Rebalance date: first available date of each month in aligned index.
- Lookback: 252 valid observations under chosen calendar policy.
- Weekend rebalances disabled in baseline.

## Holding-Period Return Model

Baseline:
- `drifted_buy_and_hold`.
- Target weights set at rebalance.
- Intra-period weights drift with realized returns.
- Turnover computed against drifted pre-trade weights.

Sensitivity:
- `constant_target_weights` (daily constant-mix approximation).

## Metrics

- Annualization factor is explicit in metric functions.
- Scripts resolve annualization factor from `dataset_metadata.json` first.
- Report tables include calendar/annualization/holding method metadata.

## Costs

- Cost model: `cost_t = turnover_one_way_t * cost_rate` on rebalance dates.
- Report gross and net outputs where applicable.

## Regime Analysis

- Primary model: HMM with 2 states.
- `allow_fallback: false` by default.
- Real model used is persisted to `regime_model_metadata.json`.
- Regime outputs are diagnostic/in-sample attribution, not forecasts.

## Supervised Risk Forecasting and Overlay (Chapter 5)

- Objective: evaluate whether risk-forecasting overlays improve downside control OOS, net of costs.
- Targets are forward-looking and leakage-safe (t+1:t+h windows):
	- realized volatility,
	- forward drawdown / drawdown events,
	- stress-event proxy from forward lower-tail outcomes.
- Features are trailing-only at time t (rolling vol/return/drawdown/correlation/ES, optional regime diagnostics).
- Validation uses walk-forward splits with embargo (default 21 days).
- Model families are conservative and reproducible (naive, EWMA, linear, tree-based).
- Model selection rule: promote model to overlay only if it improves naive baseline in walk-forward validation.
- Overlay design prioritizes dynamic crypto cap and defensive redistribution; no synthetic cash in base implementation.
- Economic evaluation remains primary: drawdown, ES95, Sharpe/Sortino/Calmar, turnover, and net costs.

## Limitations

- Historical data only; no execution slippage model yet.
- Cost model is strategy-level and simplified.
- Regime labels are not tradable signals.
- Chapter 5 overlays are implemented with conservative assumptions and should not be interpreted as stable alpha signals.
