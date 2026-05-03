# Chapter 5 - Supervised Risk Forecasting and Dynamic Overlay

## Scope

Chapter 5 evaluates whether a supervised risk-forecasting layer improves downside control out-of-sample, net of costs, without overfitting.

Explicit non-claims:
- No aggressive return prediction.
- No narrative rescue for crypto.
- No p-hacking with broad model search.

## Targets

Implemented targets in [data/processed/supervised_targets.csv](data/processed/supervised_targets.csv):
- Forward realized volatility (21d, 63d) for portfolio and selected assets.
- Forward drawdown (continuous) and drawdown-event binaries.
- Forward stress-event proxy from lower-tail cumulative returns.

Leakage control:
- Forward targets use only returns from t+1 to t+h.
- Last h rows are NaN by design.

## Features

Implemented leakage-safe features in [data/processed/supervised_features.csv](data/processed/supervised_features.csv):
- Portfolio risk/return/drawdown/ES features.
- TradFi and crypto rolling vol, momentum, drawdown.
- Cross-asset rolling correlations.
- Optional regime columns when available.
- Calendar features (month, quarter).

All rolling features use data available up to t only.

## Validation

Walk-forward validation with embargo in [src/supervised_validation.py](src/supervised_validation.py):
- train window: 756
- test window: 63
- step: 21
- min train size: 504
- embargo: 21 days

No overlap between train/test splits and no future information in training folds.

## Model Set

Conservative model families in [src/supervised_models.py](src/supervised_models.py):
- Volatility regression: naive rolling vol, EWMA, Ridge, Elastic Net, Random Forest.
- Event classification: Logistic, Calibrated Logistic, Random Forest Classifier, Gradient Boosting Classifier.

Selection principle:
- A model is selected for overlay only if it beats naive baseline in walk-forward metrics.
- Otherwise fallback remains naive/logistic baseline.

## Overlay Rule

Overlay logic in [src/risk_overlay.py](src/risk_overlay.py) and [src/overlay_backtest.py](src/overlay_backtest.py):
- Dynamic crypto cap is the primary deployable control.
- De-risking redistributes toward defensive assets (TLT/GLD) when stress probability rises.
- No synthetic cash introduced in base implementation.

Reason codes are persisted per rebalance in [outputs/chapter5/overlay_decisions.csv](outputs/chapter5/overlay_decisions.csv).

## Limitations

- Small effective sample for rare stress events.
- Probability calibration uncertainty in low-event regions.
- Overlay improvements can come from lower gross risk rather than superior forecasting skill.
- Conclusions should stay conservative: tactical crypto use can remain intermittent.
