# Audit Fix Comparison

## Before vs After (MinVar Baseline)
| scenario | n_observations | n_weekend_rows | first_rebalance | number_of_rebalances | annualized_return | annualized_volatility | sharpe | max_drawdown | es95 | average_crypto_weight | median_crypto_weight | max_crypto_weight | rebalances_crypto_gt_2pct | final_cumulative_return |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before_calendar_day_constant_target | 3042 | 869 | 2018-10-01 | 92 | 0.12277583955175286 | 0.10765045020054717 | 1.1405046548623612 | -0.2333251343508661 | 0.013375902280503458 | 0.007591277743775614 | 7.453780215714637e-17 | 0.04763183958661364 | 14.0 | 1.4074237526667464 |
| after_business_day_drifted | 2173 | 0 | 2019-01-01 | 89 | 0.11809890264099754 | 0.10774404251620649 | 1.0961061037155118 | -0.23414979433947725 | 0.015538800540207407 | 0.005348671068721499 | 5.929864591612072e-17 | 0.03229966442458746 | 6.0 | 1.3346340924151616 |

## New Baseline: MinVar with vs without Crypto
- Sharpe (with crypto): 1.0961
- Sharpe (no crypto): 1.0886
- Delta Sharpe: +0.0075

## Answers
1. Main results changed: Yes.
2. MinVar vs 60/40 under new baseline: MinVar Sharpe 1.0961.
3. Incremental crypto effect remains modest: delta Sharpe +0.0075.
4. Crypto exposure remains tactical/intermittent: avg 0.0053, max 0.0323.
5. CVaR conclusions require reading latest Chapter 3 outputs (generated in tail_risk_summary*.csv).
6. Regimes changed with strict HMM reproducibility and are now explicitly recorded in regime_model_metadata.json.