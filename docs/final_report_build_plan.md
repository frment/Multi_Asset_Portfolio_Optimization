# Final Report Build Plan (Fase 1)

Objetivo de este documento: definir un plan ejecutable y auditable para construir, en Fase 2, el Notebook final tipo paper ([notebooks/06_research_synthesis_and_publication_report.ipynb](notebooks/06_research_synthesis_and_publication_report.ipynb)) y la guía completa del código ([docs/codebase_study_guide.md](docs/codebase_study_guide.md)), sin crear aún esos entregables.

## 1) Outputs reales disponibles de Chapters 1-5

### 1.1 Chapter 1 (Baseline, benchmarks y dataset base)

| Path | Tipo | Shape | Columnas principales | Capítulo | Uso propuesto en Notebook 06 | ¿Necesario para tabla/figura final? |
|---|---|---:|---|---|---|---|
| data/processed/dataset_metadata.json | JSON | keys=8 | calendar_policy, annualization_factor, start_date, end_date, n_observations | Ch1 | Cuadro metodológico de muestra, calendario y anualización | Sí (Methodological Pipeline Overview, Methodological Audit Impact) |
| data/processed/prices_clean.csv | CSV | 2174x7 | Date, SPY, QQQ, GLD, TLT, BTC-USD, ETH-USD | Ch1 | Verificación de cobertura y universo de activos | No (apoyo/QA) |
| data/processed/prices_aligned.csv | CSV | 2174x7 | Date, SPY, QQQ, GLD, TLT, BTC-USD, ETH-USD | Ch1 | Evidencia de alineación de calendario | Sí (Methodological Audit Impact) |
| data/processed/returns_simple.csv | CSV | 2173x7 | Date, SPY, QQQ, GLD, TLT, BTC-USD, ETH-USD | Ch1 | Serie base para reconstruir narrativa de riesgo-retorno | No (apoyo) |
| data/processed/returns_log.csv | CSV | 2173x7 | Date, SPY, QQQ, GLD, TLT, BTC-USD, ETH-USD | Ch1 | Apoyo técnico para metodología | No |
| data/processed/portfolio_returns.csv | CSV | 1914x5 | Date, min_variance, equal_weight, sixty_forty, fixed_small_crypto | Ch1 | Curvas de wealth comparativas baseline | Sí (Final Strategy Comparison, cumulative wealth final) |
| data/processed/backtest_summary.csv | CSV | 4x9 | strategy, ann_return, ann_volatility, sharpe, max_drawdown, calmar, calendar_policy, annualization_factor, holding_return_method | Ch1 | Tabla de hallazgos capítulo baseline | Sí (chapter-by-chapter findings, final strategy comparison) |
| data/processed/benchmark_summary.csv | CSV | 3x9 | benchmark, ann_return, ann_volatility, sharpe, max_drawdown, calmar, calendar_policy, annualization_factor, holding_return_method | Ch1 | Evidencia frente a benchmarks | Sí (chapter-by-chapter findings, final strategy comparison) |
| data/processed/weights_history.csv | CSV | 89x7 | rebalance_date, BTC-USD, ETH-USD, SPY, QQQ, GLD, TLT | Ch1 | Distribución temporal de exposición crypto/base | Sí (crypto exposure over time/distribution) |
| data/processed/turnover_history.csv | CSV | 89x6 | rebalance_date, turnover_one_way, is_initial_rebalance, n_assets_changed, max_abs_weight_change, holding_return_method | Ch1 | Fricción implementable y estabilidad | Sí (Economic Attribution) |
| data/processed/benchmark_weights_history.csv | CSV | 267x8 | rebalance_date, SPY, QQQ, GLD, TLT, BTC-USD, ETH-USD, strategy | Ch1 | Referencia de exposición de benchmarks | Sí (crypto exposure summary) |
| data/processed/benchmark_turnover_history.csv | CSV | 267x7 | rebalance_date, strategy, turnover_one_way, is_initial_rebalance, n_assets_changed, max_abs_weight_change, holding_return_method | Ch1 | Comparación de fricción vs estrategias optimizadas | Sí (Economic Attribution) |
| data/processed/min_variance_weights_static.csv | CSV | 1x7 | rebalance_id, BTC-USD, ETH-USD, SPY, QQQ, GLD, TLT | Ch1 | Foto de solución estática de referencia | No |
| data/processed/min_variance_returns_static.csv | CSV | 2173x2 | Date, min_variance_static | Ch1 | Sensibilidad de baseline estático vs rolling | No |

### 1.2 Chapter 2 (Robustness, costes y confianza estadística)

| Path | Tipo | Shape | Columnas principales | Capítulo | Uso propuesto en Notebook 06 | ¿Necesario para tabla/figura final? |
|---|---|---:|---|---|---|---|
| data/processed/robustness/robustness_metadata.csv | CSV | 15x15 | experiment_id, dimension_tested, family, is_anchor, lookback_window_days, max_total_crypto_weight | Ch2 | Diccionario experimental y trazabilidad | Sí (Robustness and Evidence Scorecard) |
| data/processed/robustness/robustness_summary.csv | CSV | 15x21 | experiment_id, dimension_tested, ann_return, ann_volatility, sharpe, max_drawdown, calmar | Ch2 | Resultados brutos por experimento | Sí (chapter-by-chapter findings) |
| data/processed/robustness/robustness_summary_gross.csv | CSV | 15x21 | métricas gross por experimento | Ch2 | Línea base pre-costes | Sí (robustness scorecard) |
| data/processed/robustness/robustness_summary_net.csv | CSV | 60x26 | cost_bps, ann_return_gross/net, sharpe_gross/net, drawdown_gross/net | Ch2 | Costes e implementabilidad | Sí (Economic Attribution, robustness scorecard) |
| data/processed/robustness/robustness_summary_common_family.csv | CSV | 13x23 | métricas con ventana común familiar | Ch2 | Comparación justa por familia | Sí (claim vs evidence) |
| data/processed/robustness/robustness_summary_common_family_net.csv | CSV | 52x28 | métricas net/gross en ventana común | Ch2 | Evidencia robusta neta de costes | Sí (claim vs evidence, robustness scorecard) |
| data/processed/robustness/robustness_returns.csv | CSV | 28580x10 | date, experiment_id, sample_scope, portfolio_return | Ch2 | Curvas robustez agregadas | Sí (cumulative wealth final, robustness visual) |
| data/processed/robustness/robustness_turnover_panel.csv | CSV | 1270x12 | rebalance_date, experiment_id, turnover_one_way | Ch2 | Atribución económica por recambio | Sí (Economic Attribution) |
| data/processed/robustness/robustness_weights_panel.csv | CSV | 7620x10 | rebalance_date, experiment_id, ticker, weight | Ch2 | Sensibilidad de exposición crypto/no-crypto | Sí (crypto exposure summary) |
| data/processed/robustness/confidence_summary.csv | CSV | 5x16 | comparison_id, strategy_a, strategy_b, metric_compared, point_estimate_difference, ci_lower, ci_upper | Ch2 | Bloque inferencial (evidencia estadística) | Sí (claim vs evidence, robustness scorecard) |

### 1.3 Chapter 3 (Tail risk y CVaR)

| Path | Tipo | Shape | Columnas principales | Capítulo | Uso propuesto en Notebook 06 | ¿Necesario para tabla/figura final? |
|---|---|---:|---|---|---|---|
| data/processed/tail_risk/tail_risk_summary.csv | CSV | 4x20 | strategy, objective, beta, ann_return, ann_volatility, sharpe, max_drawdown, expected_shortfall | Ch3 | Comparación MinVar vs CVaR en riesgo extremo | Sí (chapter-by-chapter findings, final strategy comparison) |
| data/processed/tail_risk/tail_risk_summary_net.csv | CSV | 4x26 | coste + métricas gross/net + ES gross/net | Ch3 | Impacto neto de costes bajo objetivo tail | Sí (Economic Attribution, claim vs evidence) |
| data/processed/tail_risk/tail_risk_returns.csv | CSV | 7656x6 | date, strategy, objective, beta, max_total_crypto_weight, portfolio_return | Ch3 | Curvas y drawdowns de estrategias tail | Sí (cumulative wealth final, drawdowns final) |
| data/processed/tail_risk/tail_risk_turnover_panel.csv | CSV | 356x9 | rebalance_date, strategy, turnover_one_way | Ch3 | Coste de implementación de enfoque tail | Sí (Economic Attribution) |
| data/processed/tail_risk/tail_risk_weights_panel.csv | CSV | 2136x7 | rebalance_date, strategy, ticker, weight | Ch3 | Exposición crypto estructural vs táctica | Sí (crypto exposure over time/distribution) |
| data/processed/tail_risk/stress_summary.csv | CSV | 24x14 | window_id, strategy, scope, ann_return, sharpe, max_drawdown, expected_shortfall | Ch3 | Síntesis de comportamiento en ventanas de estrés | Sí (Tail Risk, Regimes and Overlay Synthesis) |

### 1.4 Chapter 4 (Regime analysis)

| Path | Tipo | Shape | Columnas principales | Capítulo | Uso propuesto en Notebook 06 | ¿Necesario para tabla/figura final? |
|---|---|---:|---|---|---|---|
| data/processed/regime_analysis/regime_model_metadata.json | JSON | keys=9 | primary_model_requested, model_used, allow_fallback, n_states, feature_set | Ch4 | Transparencia metodológica del detector de regímenes | Sí (Methodological Pipeline Overview) |
| data/processed/regime_analysis/regime_features.csv | CSV | 2111x11 | date, realized_vol_*, drawdown_*, corr_*, momentum_* | Ch4 | Base explicativa de régimen (no señal táctica directa) | No (apoyo) |
| data/processed/regime_analysis/regime_labels.csv | CSV | 2111x3 | date, regime_id, regime_name | Ch4 | Segmentación temporal para síntesis final | Sí (Tail Risk, Regimes and Overlay Synthesis) |
| data/processed/regime_analysis/regime_model_summary.csv | CSV | 2x17 | state_id, state_name, n_days, share_pct, mean_duration_days | Ch4 | Tabla de persistencia e interpretación de estados | Sí (chapter-by-chapter findings) |
| data/processed/regime_analysis/regime_transition_matrix.csv | CSV | 2x3 | from_state, Low-stress / Risk-on, High-stress / Risk-off | Ch4 | Figura/tablas de dinámica de régimen | Sí (Tail Risk, Regimes and Overlay Synthesis) |
| data/processed/regime_analysis/regime_conditional_performance.csv | CSV | 8x12 | strategy, regime_name, ann_return, sharpe, expected_shortfall | Ch4 | Atribución condicional gross | Sí (claim vs evidence) |
| data/processed/regime_analysis/regime_conditional_performance_net.csv | CSV | 8x12 | estrategia por régimen neta | Ch4 | Atribución condicional neta | Sí (claim vs evidence) |
| data/processed/regime_analysis/regime_crypto_exposure.csv | CSV | 8x10 | strategy, regime_name, mean_crypto_weight, median_crypto_weight, share_crypto_gt_2pct | Ch4 | Exposición crypto estructural vs táctica por régimen | Sí (crypto exposure summary) |
| data/processed/regime_analysis/regime_drawdown_tail_summary.csv | CSV | 40x19 | window_id, strategy, scope, regime_name, max_drawdown, expected_shortfall | Ch4 | Puente entre regímenes y eventos de cola | Sí (Tail Risk, Regimes and Overlay Synthesis) |

### 1.5 Chapter 5 (Supervised overlay)

| Path | Tipo | Shape | Columnas principales | Capítulo | Uso propuesto en Notebook 06 | ¿Necesario para tabla/figura final? |
|---|---|---:|---|---|---|---|
| data/processed/supervised_features.csv | CSV | 2173x34 | Date + bloque de features de retorno/vol/correlación/régimen | Ch5 | Contexto de diseño de señales supervisadas | No (apoyo metodológico) |
| data/processed/supervised_targets.csv | CSV | 2173x13 | Date + targets de vol/drawdown/eventos | Ch5 | Definición de objetivos ex-ante | No (apoyo metodológico) |
| outputs/chapter5/chapter5_metadata.json | JSON | keys=16 | chapter, calendar_policy, annualization_factor, validation, models_attempted, models_selected, output_paths | Ch5 | Auditoría de pipeline Ch5 | Sí (Methodological Pipeline Overview, Methodological Audit Impact) |
| outputs/chapter5/model_diagnostics.csv | CSV | 55x7 | target_col, task, model_name, n_samples, n_splits, n_predictions, status | Ch5 | Calidad de ejecución por modelo/target | Sí (robustness scorecard) |
| outputs/chapter5/model_scores.csv | CSV | 55x23 | métricas de clasificación/regresión OOS | Ch5 | Tabla de desempeño y selección prudente | Sí (chapter-by-chapter findings, claim vs evidence) |
| outputs/chapter5/model_selection.csv | CSV | 12x5 | target_col, task, best_model, selected_model, improved_vs_naive | Ch5 | Evidencia de selección conservadora | Sí (claim vs evidence) |
| outputs/chapter5/calibration_tables.csv | CSV | 150x6 | target_col, model_name, decile, mean_score, event_rate, n | Ch5 | Figura/tablas de calibración | Sí (robustness scorecard) |
| outputs/chapter5/forecast_bucket_analysis.csv | CSV | 200x6 | target_col, model_name, bucket, mean_pred, mean_realized, n | Ch5 | Coherencia predicción-realizado | Sí (robustness scorecard) |
| outputs/chapter5/overlay_backtest_summary.csv | CSV | 6x22 | strategy, ann_return, sharpe, max_drawdown, es95, turnover, total_costs, avg_crypto_weight | Ch5 | Comparación final de estrategias con overlay | Sí (final strategy comparison, economic attribution) |
| outputs/chapter5/overlay_daily_returns.csv | CSV | 11484x3 | date, daily_return, strategy | Ch5 | Curvas finales de wealth y drawdown | Sí (cumulative wealth final, drawdowns final) |
| outputs/chapter5/overlay_weights.csv | CSV | 534x8 | rebalance_date, BTC-USD, ETH-USD, SPY, QQQ, GLD, TLT, strategy | Ch5 | Exposición crypto en el tiempo | Sí (crypto exposure over time/distribution) |
| outputs/chapter5/overlay_turnover.csv | CSV | 534x4 | rebalance_date, turnover_one_way, is_initial_rebalance, strategy | Ch5 | Capa de fricción y costes | Sí (Economic Attribution) |
| outputs/chapter5/overlay_decisions.csv | CSV | 534x10 | date, risk_scale, crypto_cap, de_risk_flag, reason, adjusted_crypto_weight | Ch5 | Conteo de razones de overlay y trazabilidad de decisiones | Sí (overlay reason counts, crypto exposure summary) |
| outputs/chapter5/predictions_*.csv (55 archivos) | CSV (familia) | 3087-3969 x 8 (según target) | date, y_true, y_pred, y_prob, split_id, model_name, task, target_col | Ch5 | Material granular para anexos/validación de claims | Opcional (anexo/reproducibilidad) |
| outputs/chapter5_debug/model_scores.csv | CSV | 3x23 | métricas debug | Ch5 debug | QA local de pipeline reducido | No (no usar como evidencia final) |
| outputs/chapter5_debug/model_diagnostics.csv | CSV | 3x7 | estado debug | Ch5 debug | QA local | No |
| outputs/chapter5_debug/model_selection.csv | CSV | 12x5 | selección debug | Ch5 debug | QA local | No |
| outputs/chapter5_debug/calibration_tables.csv | CSV | 10x6 | calibración debug | Ch5 debug | QA local | No |
| outputs/chapter5_debug/forecast_bucket_analysis.csv | CSV | 10x6 | buckets debug | Ch5 debug | QA local | No |
| outputs/chapter5_debug/predictions_*.csv (3 archivos) | CSV (familia) | 189x8 | esquema de predicciones debug | Ch5 debug | QA local | No |

### 1.6 Metodological audit (cross-chapter)

| Path | Tipo | Shape | Columnas principales | Capítulo | Uso propuesto en Notebook 06 | ¿Necesario para tabla/figura final? |
|---|---|---:|---|---|---|---|
| outputs/audit_fix_comparison.csv | CSV | 2x15 | scenario, annualized_return, annualized_volatility, sharpe, max_drawdown, es95, average_crypto_weight | Cross | Tabla before/after de endurecimiento metodológico | Sí (audit before/after, Methodological Audit Impact) |
| outputs/audit_fix_metadata.json | JSON | keys=3 | comparison_csv, comparison_report, delta_sharpe_new_baseline | Cross | Metadatos del experimento de auditoría | Sí (Methodological Audit Impact) |
| outputs/audit_fix_comparison.md | MD | n/a | reporte textual | Cross | Soporte narrativo y trazabilidad | Opcional |

## 2) Módulos reales en src/

| Archivo | Descripción (1 línea) | Rol metodológico |
|---|---|---|
| src/__init__.py | Inicializa paquete `src`. | Soporte de importación y organización del código. |
| src/backtest.py | Motor de backtest walk-forward con rebalanceo y cálculo de turnover. | Núcleo Ch1; base temporal consistente para comparativas posteriores. |
| src/benchmarks.py | Construye benchmarks (equal-weight, 60/40, fixed-small-crypto) bajo la misma convención de rebalanceo. | Contrafactuales comparables para claims de valor añadido. |
| src/bootstrap.py | Implementa bootstrap por bloques para intervalos de confianza. | Capa inferencial de Ch2 (evidencia estadística, no solo point estimates). |
| src/chapter5_runtime.py | Utilidades de runtime/logging para ejecución de Chapter 5. | Trazabilidad y reproducibilidad operativa de Ch5. |
| src/config.py | Carga y resuelve configuración YAML del proyecto. | Control explícito de supuestos experimentales. |
| src/costs.py | Aplica modelo lineal de costes por turnover. | Implementabilidad económica de resultados. |
| src/cvar_optimizer.py | Resuelve optimización Min-CVaR histórica con restricciones. | Núcleo Ch3 para objetivo de cola. |
| src/data_loader.py | Descarga/carga datos de precios del universo de activos. | Punto de entrada de datos, reproducibilidad de muestra. |
| src/metrics.py | Calcula métricas de performance y riesgo (Sharpe, MDD, ES, etc.). | Lenguaje métrico común en Chapters 1-5. |
| src/model_evaluation.py | Evalúa modelos supervisados (clasificación y regresión) OOS. | Evidencia cuantitativa de calidad predictiva en Ch5. |
| src/optimizer.py | Optimizador de mínima varianza con restricciones y covarianza configurable. | Estrategia base de referencia (Ch1/Ch2). |
| src/overlay_backtest.py | Ejecuta backtest de la estrategia con overlay de riesgo. | Validación end-to-end del overlay en Ch5. |
| src/preprocessing.py | Limpia, alinea calendario y construye retornos. | Higiene metodológica de datos base (incluye política de calendario). |
| src/regime_detection.py | Detecta regímenes (HMM/KMeans) y genera transiciones/etiquetas. | Núcleo de diagnóstico Ch4 (estructura de mercado). |
| src/regime_evaluation.py | Evalúa desempeño y exposición por régimen detectado. | Atribución condicional para claims prudentes. |
| src/regime_features.py | Construye features causales de régimen. | Base informativa para segmentación de estrés/riesgo. |
| src/risk_overlay.py | Define reglas de overlay (crypto cap dinámico, de-risking, escalado). | Mecanismo de decisión táctico-conservador en Ch5. |
| src/robustness.py | Ejecuta barridos de sensibilidad (lookback, cap crypto, frecuencia, etc.). | Núcleo de robustez Ch2. |
| src/stress.py | Evalúa estrategias en ventanas históricas de estrés. | Validación de cola y resiliencia. |
| src/supervised_features.py | Construye panel de features supervisadas sin leakage. | Entrada causal para modelos de riesgo Ch5. |
| src/supervised_models.py | Entrena/valida zoo de modelos y baselines. | Producción de señales predictivas OOS Ch5. |
| src/supervised_targets.py | Construye targets forward de vol/drawdown/eventos. | Definición ex-ante del problema predictivo Ch5. |
| src/supervised_validation.py | Genera splits walk-forward con embargo temporal. | Barrera principal anti-leakage en Ch5. |
| src/utils.py | Utilidades generales (paths/directorios). | Infraestructura compartida. |

## 3) Scripts reales en scripts/

| Script | Qué ejecuta | Outputs principales inferibles |
|---|---|---|
| scripts/run_all_chapters.py | Orquesta pipeline Chapter 1-4 (y opcionalmente 5 según configuración). | Outputs agregados de `data/processed/*` y análisis por capítulo. |
| scripts/run_audit_fix_comparison.py | Compara baseline metodológico nuevo vs sensibilidad previa de calendario/política de holding. | `outputs/audit_fix_comparison.csv`, `outputs/audit_fix_comparison.md`, `outputs/audit_fix_metadata.json`. |
| scripts/run_backtest.py | Ejecuta backtest baseline de estrategia optimizada y series asociadas. | Resúmenes y series en `data/processed/` (backtest, pesos, turnover). |
| scripts/run_benchmarks.py | Ejecuta y resume benchmarks bajo convención comparable. | `data/processed/benchmark_summary.csv` y paneles benchmark. |
| scripts/run_build_dataset.py | Construye dataset procesado desde precios. | `prices_clean.csv`, `prices_aligned.csv`, `returns_simple.csv`, `returns_log.csv`, `dataset_metadata.json`. |
| scripts/run_chapter5.py | Orquesta Chapter 5 completo (targets -> features -> models -> overlay). | Archivos en `data/processed/supervised_*.csv` y `outputs/chapter5/*`. |
| scripts/run_download.py | Descarga datos crudos del universo de activos. | Datos crudos en `data/raw/` (según configuración). |
| scripts/run_optimizer.py | Ejecuta optimización estática de referencia. | `min_variance_weights_static.csv`, `min_variance_returns_static.csv`. |
| scripts/run_regime_analysis.py | Ejecuta pipeline de regímenes (features, detección, evaluación). | `data/processed/regime_analysis/*`. |
| scripts/run_risk_overlay.py | Ejecuta overlay de riesgo y backtest asociado. | `outputs/chapter5/overlay_*.csv`. |
| scripts/run_robustness.py | Ejecuta análisis de robustez multi-experimento. | `data/processed/robustness/*`. |
| scripts/run_statistical_confidence.py | Ejecuta bootstrap/confianza para comparaciones predefinidas. | `data/processed/robustness/confidence_summary.csv`. |
| scripts/run_supervised_features.py | Genera features supervisadas Ch5. | `data/processed/supervised_features.csv`. |
| scripts/run_supervised_models.py | Entrena/evalúa modelos y exporta scores/predicciones. | `outputs/chapter5/model_scores.csv`, `model_selection.csv`, `predictions_*.csv`, tablas de calibración. |
| scripts/run_supervised_targets.py | Genera targets supervisadas forward. | `data/processed/supervised_targets.csv`. |
| scripts/run_tail_risk.py | Ejecuta comparativas MinVar vs CVaR y stress tests. | `data/processed/tail_risk/*`. |

## 4) Tests reales en tests/

| Test file | Qué protege | Por qué importa metodológicamente |
|---|---|---|
| tests/test_backtest.py | Lógica temporal de backtest, restricciones y turnover. | Evita look-ahead y asegura comparabilidad de resultados OOS. |
| tests/test_backtest_drifted_returns.py | Diferencia entre drifted buy-and-hold y esquemas alternativos. | Garantiza consistencia del método baseline auditado. |
| tests/test_benchmarks_monthly_drifted.py | Reglas de benchmarks con rebalanceo mensual drifted. | Evita comparaciones injustas contra estrategias optimizadas. |
| tests/test_bootstrap.py | Esquema/reproducibilidad del bootstrap y CIs. | Valida robustez inferencial de Chapter 2. |
| tests/test_calendar_policy.py | Política de calendario (alineación business day, sin sesgos de fin de semana). | Protege la integridad de métricas anualizadas y retornos. |
| tests/test_chapter5_pipeline_smoke.py | Smoke test end-to-end de pipeline Chapter 5. | Comprueba reproducibilidad mínima del bloque supervisado. |
| tests/test_cvar_optimizer.py | Restricciones y estabilidad del optimizador CVaR. | Base de credibilidad de resultados tail risk. |
| tests/test_metrics_annualization.py | Correcta anualización de métricas. | Evita interpretaciones erróneas de Sharpe/retornos. |
| tests/test_optimizer_covariance.py | Comportamiento de covarianza/optimizer bajo distintas opciones. | Sustenta claims de estabilidad/robustez del optimizador. |
| tests/test_overlay_backtest.py | Esquema y consistencia de outputs del overlay backtest. | Garantiza trazabilidad de resultados tácticos en Ch5. |
| tests/test_pipeline_smoke.py | Smoke test Chapter 1-4. | Asegura ejecutabilidad transversal de pipeline principal. |
| tests/test_rebalance_dates.py | Cálculo de fechas de rebalanceo y condiciones de lookback. | Protege validez temporal de toda la investigación. |
| tests/test_regime_dependencies.py | Dependencias/fallback de detección de regímenes. | Mantiene robustez operativa entre entornos. |
| tests/test_regime_detection.py | Etiquetado y estructura de transiciones de regímenes. | Fundamenta la lectura de estados de mercado. |
| tests/test_regime_evaluation.py | Métricas condicionales por régimen. | Soporta síntesis prudente de comportamiento por entorno. |
| tests/test_regime_features.py | Construcción causal y calidad de features de régimen. | Evita leakage en análisis de régimen. |
| tests/test_risk_overlay.py | Reglas de overlay y consistencia de pesos/caps. | Defiende que la capa táctica cumple reglas conservadoras. |
| tests/test_robustness.py | Esquema y coherencia de outputs de robustez. | Garantiza trazabilidad de sensibilidad Ch2. |
| tests/test_stress.py | Evaluación en ventanas de estrés y manejo de casos límite. | Refuerza validez de claims en escenarios extremos. |
| tests/test_supervised_features.py | Features causales y sin información futura. | Barrera clave anti-leakage en Ch5. |
| tests/test_supervised_models.py | Esquema/consistencia de predicciones y scores OOS. | Mantiene comparabilidad entre modelos. |
| tests/test_supervised_targets.py | Targets forward correctamente alineadas. | Evita contaminar entrenamiento con futuro. |
| tests/test_tail_risk_metrics.py | Cálculo de ES y ratios derivados. | Sostiene credibilidad de resultados de cola. |
| tests/test_time_series_validation.py | Splits walk-forward con embargo y no solapamiento indebido. | Garantiza validez temporal del aprendizaje supervisado. |

## 5) Estructura exacta propuesta de Notebook 06

Sección a sección (orden exacto y objetivo editorial):

1. Abstract / Executive Summary
   - 5-8 bullets con pregunta, hallazgo principal, fuerza de evidencia y límites.
2. Research Question Revisited
   - Reformulación falsable y criterios de éxito/fracaso.
3. Methodological Pipeline Overview
   - Diagrama único del flujo Ch1 -> Ch5 + tabla de supuestos clave (calendario, anualización, costes, OOS).
4. Chapter-by-Chapter Findings
   - Subtabla por capítulo con 3-5 hallazgos y nivel de confianza.
5. Final Claim vs Evidence Table
   - Claims finales, evidencia cuantitativa, fuerza de evidencia y caveats.
6. Economic Attribution
   - Retorno bruto vs neto, turnover, costes acumulados, impacto marginal de overlay.
7. Crypto Exposure: Structural vs Tactical
   - Peso medio/mediano/máximo, frecuencia de exposición >2%, evolución temporal.
8. Tail Risk, Regimes and Overlay Synthesis
   - Integración de señales de cola, regímenes y decisiones de overlay.
9. Methodological Audit Impact
   - Before/after de auditoría metodológica y efecto en métricas clave.
10. Robustness and Evidence Scorecard
   - Scorecard con robustez transversal (sensibilidad + inferencia + calibración).
11. Final Strategy Comparison
   - Tabla final de estrategias candidatas y ranking prudente.
12. Limitations
   - Límites de muestra, universo, costes, regime-dependence, estabilidad temporal.
13. Final Conclusion
   - Conclusión conservadora y condicionada por evidencia disponible.
14. Publication-Ready Summary
   - Resumen 1 página, lenguaje ejecutivo, cifras clave y disclaimers.
15. References
   - Referencias a notebooks 01-05, scripts de pipeline y outputs usados.

## 6) Estructura exacta propuesta de docs/codebase_study_guide.md

Orden propuesto (exacto):

1. Repo Overview
2. Pipeline End-to-End (de datos crudos a outputs de publicación)
3. Configs (`config/*.yaml`) y cómo cambian los experimentos
4. Módulos `src/` (todos, uno a uno)
5. Scripts `scripts/` (todos, orden recomendado de ejecución)
6. Tests `tests/` (todos, por bloque metodológico)
7. Data y Outputs (`data/`, `outputs/`) con mapa de artefactos
8. Reproducción (entorno, comandos mínimos, rutas esperadas)
9. Debugging (fallos típicos, dónde mirar, cómo validar integridad)
10. Extensiones futuras (sin romper baseline auditado)
11. Plan de estudio para el usuario (ruta de aprendizaje de 2-3 semanas)

## 7) Tablas que se podrán construir en Notebook 06

Tablas mínimas comprometidas y fuentes:

1. Chapter-by-chapter findings
   - Fuentes: `backtest_summary.csv`, `robustness_summary*.csv`, `tail_risk_summary*.csv`, `regime_model_summary.csv`, `model_scores.csv`, `overlay_backtest_summary.csv`.
2. Claim vs evidence
   - Fuentes: `confidence_summary.csv`, `tail_risk_summary_net.csv`, `regime_conditional_performance_net.csv`, `model_selection.csv`, `audit_fix_comparison.csv`.
3. Final strategy comparison
   - Fuentes: `backtest_summary.csv`, `benchmark_summary.csv`, `tail_risk_summary.csv`, `overlay_backtest_summary.csv`.
4. Economic attribution
   - Fuentes: `robustness_summary_net.csv`, `tail_risk_summary_net.csv`, `overlay_backtest_summary.csv`, `overlay_turnover.csv`, `turnover_history.csv`.
5. Crypto exposure summary
   - Fuentes: `weights_history.csv`, `tail_risk_weights_panel.csv`, `regime_crypto_exposure.csv`, `overlay_weights.csv`, `overlay_decisions.csv`.
6. Audit before/after
   - Fuente principal: `outputs/audit_fix_comparison.csv` (+ metadata JSON).
7. Robustness scorecard
   - Fuentes: `robustness_summary_common_family_net.csv`, `confidence_summary.csv`, `calibration_tables.csv`, `forecast_bucket_analysis.csv`, `model_diagnostics.csv`.

## 8) Figuras que se podrán construir en Notebook 06

Figuras mínimas comprometidas y fuentes:

1. Pipeline diagram
   - Fuente: estructura lógica del pipeline + `chapter5_metadata.json` + `dataset_metadata.json`.
2. Cumulative wealth final
   - Fuentes: `portfolio_returns.csv`, `tail_risk_returns.csv`, `overlay_daily_returns.csv`.
3. Drawdowns final
   - Fuentes: mismas series de retornos de la figura anterior.
4. Crypto exposure over time
   - Fuentes: `weights_history.csv`, `tail_risk_weights_panel.csv`, `overlay_weights.csv`.
5. Crypto exposure distribution
   - Fuentes: `regime_crypto_exposure.csv`, `overlay_weights.csv`, `overlay_decisions.csv`.
6. Economic attribution bars
   - Fuentes: `robustness_summary_net.csv`, `tail_risk_summary_net.csv`, `overlay_backtest_summary.csv`.
7. Overlay reason counts (si Chapter 5 tiene outputs válidos)
   - Fuente: `overlay_decisions.csv` (columna `reason`).

## 9) Outputs faltantes o riesgos

1. No se observan en `outputs/` artefactos finales consolidados cross-chapter (único tablero integrado); el Notebook 06 deberá consolidarlos explícitamente.
2. Las series de retornos están en formatos/horizontes distintos (`portfolio_returns.csv`, `tail_risk_returns.csv`, `overlay_daily_returns.csv`), por lo que existe riesgo de comparaciones no alineadas si no se recorta a ventana común.
3. Chapter 5 tiene outputs debug (`outputs/chapter5_debug/*`) que no deben mezclarse con evidencia final; riesgo de contaminación si no se filtra.
4. La familia `predictions_*.csv` es amplia (55 archivos) y puede sobrecargar el notebook si se intenta visualizar completa; recomendable usar agregados (`model_scores`, calibración, buckets).
5. Algunas afirmaciones de causalidad económica pueden requerir una descomposición adicional (retorno bruto, coste, turnover) a partir de varias tablas, no una sola fuente.

## 10) Plan de Fase 2 (concreto, sin nuevos modelos)

1. Definir contrato de datos para Notebook 06
   - Lista de artefactos obligatorios, opcionales y excluidos (debug).
2. Construir capa de carga y normalización de tablas
   - Estandarizar nombres humanos de columnas, tipos, fechas y unidades.
3. Crear dataset maestro de comparación final
   - Unir baseline/robustness/tail/overlay en ventana temporal común para comparaciones justas.
4. Generar tablas editoriales (no técnicas) del paper
   - Chapter findings, claim-evidence, strategy comparison, scorecard.
5. Generar figuras editoriales
   - Wealth, drawdown, exposición crypto temporal/distribución, atribución económica, diagrama pipeline.
6. Redactar narrativa prudente por sección
   - Mensajes defendibles: qué evidencia es fuerte, cuál es condicional y qué no se puede concluir.
7. Incorporar bloque Methodological Audit Impact
   - Before/after limpio y legible para auditoría.
8. Integrar sección de limitaciones y conclusión final
   - Evitar overclaiming; dejar explícitas fronteras del estudio.
9. Añadir resumen publication-ready y referencias
   - Texto ejecutivo de 1 página + referencias internas a capítulos/outputs.
10. Validación final de reproducibilidad del notebook
   - Verificar que todas las celdas corren con outputs existentes y que no requiere nuevos entrenamientos.

---

## 11) Core vs optional evidence

### 11.1 Core evidence (obligatorio en Notebook 06)

Todos los outputs listados en secciones 1.1–1.5 marcados como **"Sí"** en la columna *¿Necesario para tabla/figura final?*. Son la base no negociable de claims y figuras.

### 11.2 Optional / appendix evidence

- `outputs/chapter5/predictions_*.csv` (55 archivos) — válidos para anexo o reproducibilidad, no para el cuerpo principal.
- `outputs/audit_fix_comparison.md` — soporte narrativo; la tabla CSV es el core.
- `data/processed/robustness/robustness_returns.csv` — útil para figuras adicionales de robustez, opcional en sección principal.

### 11.3 No usar como evidencia final

> **⛔ `outputs/chapter5_debug/*` — EXCLUIDO explícitamente.**
> Contiene outputs de un pipeline reducido (3 modelos, submuestra). No deben aparecer en ninguna tabla, figura, narración ni comparativa del Notebook 06. Riesgo de contaminación si se cargan con glob sin filtro explícito.

---

## 12) Common-window rule

- Todas las curvas de wealth y drawdown finales (sección 5.11 y figura Cumulative wealth/Drawdowns) deben recortarse a la **ventana temporal común** de las series que se comparan antes de calcular cualquier métrica.
- **No mezclar** métricas full-period con curvas common-window en la misma tabla o figura sin una nota explícita.
- Si se construye la tabla final de estrategias (sección 5.11), indicar en el encabezado o footnote si las métricas provienen de:
  - **"Original outputs"** — métricas calculadas sobre el período completo de cada serie sin recorte adicional.
  - **"Common-window recomputed"** — métricas recomputadas sobre la ventana compartida en el propio Notebook 06.
- En Fase 2, el dataset maestro de comparación final (paso 3 del plan) debe resolver este recorte antes de cualquier figura.

---

## 13) Human display-name mapping

Usar en tablas, leyendas y markdown del Notebook 06. Evitar nombres técnicos crudos.

### 13.1 Estrategias

| Nombre técnico (columna/índice) | Nombre legible |
|---|---|
| `min_variance` | Minimum Variance |
| `equal_weight` | Equal Weight |
| `sixty_forty` | 60/40 |
| `fixed_small_crypto` | Fixed Small Crypto |
| `min_variance_cvar` | MinVar + CVaR |
| `minvar_no_overlay` | MinVar (No Overlay) |
| `minvar_naive_overlay` | MinVar + Naive Overlay |
| `minvar_ml_overlay` | MinVar + ML Overlay |
| `minvar_combined_overlay` | MinVar + Combined Overlay |
| `spy_only` | SPY Only |
| `equal_weight_bench` | Equal Weight Benchmark |
| `sixty_forty_bench` | 60/40 Benchmark |

### 13.2 Métricas

| Nombre técnico | Nombre legible |
|---|---|
| `ann_return` | Ann. Return |
| `ann_volatility` | Ann. Volatility |
| `sharpe` | Sharpe Ratio |
| `max_drawdown` | Max Drawdown |
| `calmar` | Calmar Ratio |
| `expected_shortfall` / `es95` | CVaR (95%) |
| `turnover_one_way` | One-Way Turnover |
| `total_costs` | Total Costs (bps) |
| `avg_crypto_weight` | Avg. Crypto Weight |

### 13.3 Modelos supervisados

| Nombre técnico | Nombre legible |
|---|---|
| `naive_rolling_vol` | Naive Rolling Vol |
| `logistic_regression` | Logistic Regression |
| `random_forest_classifier` | Random Forest |
| `gradient_boosting_classifier` | Gradient Boosting |
| `ridge_regression` | Ridge Regression |
| `random_forest_regressor` | Random Forest (Reg.) |
| `gradient_boosting_regressor` | Gradient Boosting (Reg.) |

---

## 14) Phase 2 acceptance criteria

El Notebook 06 se considera listo cuando cumple **todos** los criterios siguientes:

- [ ] Corre de principio a fin (`Run All`) sin entrenar modelos ni requerir outputs ausentes.
- [ ] No carga ningún archivo de `outputs/chapter5_debug/`.
- [ ] Todos los nombres de estrategias, métricas y modelos usan el mapping de la sección 13.
- [ ] Ningún gráfico tiene leyendas solapadas o ilegibles.
- [ ] Ninguna tabla supera 10 columnas sin justificación editorial.
- [ ] Cada figura y tabla principal incluye una celda markdown de lectura inmediatamente debajo.
- [ ] El cuerpo del notebook incluye explícitamente:
  - Sección **Claim vs Evidence** (qué se afirma y qué evidencia lo sustenta).
  - Sección **Methodological Audit Impact** (before/after de auditoría).
  - Sección **Crypto Structural vs Tactical** (exposición media, mediana, frecuencia >2% por régimen).
  - Sección **Limitations** (muestra, universo, costes, régimen-dependencia, estabilidad temporal).
  - Sección **Final Conclusion** (conservadora, condicionada por outputs existentes).

---

## 15) Visual standards

Estándares mínimos para todas las figuras del Notebook 06:

- **Máximo 5 estrategias por gráfico de líneas.** Si hay más, dividir en subplots o agrupar en narrativa.
- **Leyendas fuera del área de plot** (`bbox_to_anchor`) cuando hay solapamiento con curvas.
- **Porcentajes formateados** con un decimal: `12.3%`, no `0.123` ni `12.345678%`.
- **Tablas: máximo 8–10 columnas** en el cuerpo del notebook; tablas más amplias van a anexo o se rotan con `style`.
- **Captions** en markdown debajo de cada figura principal (1-2 frases que digan qué muestra la figura y la lectura clave).
- **Sin nombres técnicos crudos** en ejes, leyendas ni encabezados de tabla (aplicar mapping sección 13).

---

## 16) References plan

Referencias metodológicas mínimas a incluir como lista legible en la sección *References* del Notebook 06 (sección 5.15):

| Referencia | Relevancia |
|---|---|
| Markowitz (1952) — *Portfolio Selection* | Optimización media-varianza base |
| Michaud (1989) — *The Markowitz Optimization Enigma* | Límites de optimización clásica |
| DeMiguel et al. (2009) — *Optimal vs Naive Diversification* | Justificación de equal-weight benchmark |
| Ledoit & Wolf (2004) — *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* | Estimación robusta de covarianza |
| Rockafellar & Uryasev (2000) — *Optimization of Conditional Value-at-Risk* | Base del optimizador CVaR |
| Hamilton (1989) — *A New Approach to the Economic Analysis of Nonstationary Time Series* | Fundamento HMM para detección de regímenes |
| Ang & Bekaert (2002) — *International Asset Allocation with Regime Shifts* | Regímenes y asignación de activos |
| Lo (2002) — *The Statistics of Sharpe Ratios* | Inferencia sobre Sharpe |
| Bailey et al. (2014) — *The Deflated Sharpe Ratio* | Ajuste por selección de estrategias |
| Platanakis & Urquhart (2020) — *Should Investors Include Bitcoin in Their Portfolio?* | Evidencia empírica sobre crypto en carteras |
| Liu & Tsyvinski (2021) — *Risks and Returns of Cryptocurrency* | Factores de riesgo de criptoactivos |

---

## 17) Expected final thesis

Tesis tentativa prudente, condicionada a los outputs actuales:

> **La construcción de cartera basada en riesgo (mínima varianza) aporta más valor que la simple inclusión de criptoactivos.**

Desglose de claims:

1. **MinVar/risk-based construction** mejora el perfil riesgo-retorno (Sharpe, drawdown) respecto a benchmarks pasivos incluso sin crypto, y la mejora es más consistente entre experimentos de robustez que la aportación de BTC/ETH.
2. **BTC/ETH no emergen como asignación estructural robusta**: la exposición media es baja, variable según régimen y sensible al lookback; el optimizador tiende a asignar pesos mínimos o nulos en ventanas de estrés.
3. **Crypto permanece como exposición táctica/intermitente**: su valor aparece principalmente en regímenes *Low-stress/Risk-on* y desaparece o se reduce explícitamente bajo las reglas del overlay.
4. **CVaR, regímenes y overlay no cambian materialmente la conclusión**: dado los outputs actuales, las estrategias con capa adicional (tail, overlay) producen mejoras marginales sobre MinVar puro, pero no revierten el hallazgo central. La evidencia de mejora del overlay es condicional a la calidad predictiva de los modelos OOS, que es modesta.

> Esta tesis debe revisarse si se regeneran outputs con muestra extendida o universo distinto.

---

Estado: Fase 1 completada (plan ejecutable creado, secciones 11–17 añadidas). Fase 2 no iniciada.