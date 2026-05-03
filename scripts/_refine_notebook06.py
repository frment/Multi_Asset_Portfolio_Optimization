from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path


NOTEBOOK = Path("notebooks/06_research_synthesis_and_publication_report.ipynb")


def as_lines(source: str) -> list[str]:
	return source.strip("\n").splitlines(keepends=True)


def cell_source(nb: dict, index: int) -> str:
	return "".join(nb["cells"][index - 1].get("source", []))


def set_source(nb: dict, index: int, source: str) -> None:
	nb["cells"][index - 1]["source"] = as_lines(source)
	if nb["cells"][index - 1].get("cell_type") == "code":
		nb["cells"][index - 1]["execution_count"] = None
		nb["cells"][index - 1]["outputs"] = []


def replace_in_source(nb: dict, old: str, new: str) -> None:
	for cell in nb["cells"]:
		src = "".join(cell.get("source", []))
		if old in src:
			cell["source"] = src.replace(old, new).splitlines(keepends=True)
			if cell.get("cell_type") == "code":
				cell["execution_count"] = None
				cell["outputs"] = []


def recursive_replace(obj, replacements: dict[str, str]):
	if isinstance(obj, str):
		for old, new in replacements.items():
			obj = obj.replace(old, new)
		return obj
	if isinstance(obj, list):
		return [recursive_replace(item, replacements) for item in obj]
	if isinstance(obj, dict):
		return {key: recursive_replace(value, replacements) for key, value in obj.items()}
	return obj


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

set_source(
	nb,
	1,
	"""
# Reporte final de investigación

**Optimización de carteras multi-activo con crypto**

Este documento presenta la síntesis final del estudio en formato de research note aplicada. Consume únicamente outputs ya generados y auditados; no entrena modelos, no rehace pipelines pesados y excluye por diseño cualquier artefacto de `outputs/chapter5_debug/` como evidencia.

**Nota metodológica.** Salvo que se indique lo contrario, todas las métricas y figuras usan la metodología final auditada del proyecto, con calendario de días hábiles, retorno tipo drifted buy-and-hold, costes explícitos y comparación fuera de muestra sobre ventanas homogéneas.

**Convención de nombres.** En esta nota, `baseline final` se refiere a la especificación auditada definitiva del proyecto. La comparación histórica previa al endurecimiento metodológico queda relegada a trazabilidad secundaria y no forma parte del bloque visual principal.
""",
)

set_source(
	nb,
	10,
	"""
from IPython.display import Markdown, display

summary_points = [
	"El estudio usa outputs ya generados para sintetizar los bloques empíricos; este reporte no entrena modelos ni ejecuta pipelines pesados.",
	"La comparación central distingue tres efectos: construcción MinVar, permiso a BTC/ETH y capas posteriores de control de riesgo.",
	"MinVar mejora frente al 60/40 en la ventana fuera de muestra, pero MinVar sin crypto queda muy cerca de MinVar.",
	"La diferencia atribuible a crypto es pequeña y debe juzgarse junto con el intervalo bootstrap, costes, turnover y exposición efectiva.",
	"CVaR, regímenes y modelos supervisados aportan diagnósticos útiles, pero no cambian de forma material la conclusión principal.",
	"La conclusión no es universal: depende de la muestra, del universo de activos, de los costes asumidos y de la especificación de rebalanceo.",
]

display(Markdown("\n".join(f"- {point}" for point in summary_points)))
""",
)

source14 = cell_source(nb, 14)
source14 = source14.replace("# shown once in Figure 6.1", "# shown once in Figure 1")
source14 = source14.replace('display(Markdown("### Tabla A. Diseño empírico del estudio"))', 'display(Markdown("### Tabla 1 — Diseño empírico del estudio"))')
source14 = source14.replace("Figura 6.1", "Figura 1")
set_source(nb, 14, source14)

source17 = cell_source(nb, 17)
for old, new in {
	"Capítulo 1": "Baseline",
	"Capítulo 2": "Robustez",
	"Capítulo 3": "CVaR y cola",
	"Capítulo 4": "Regímenes",
	"Capítulo 5": "Señales y overlay",
	"Capítulo": "Bloque",
	"findings_df = pd.DataFrame(rows, columns=[\"Bloque\",": "findings_df = pd.DataFrame(rows, columns=[\"Bloque\",",
}.items():
	source17 = source17.replace(old, new)
set_source(nb, 17, source17)

source24 = cell_source(nb, 24)
source24 = source24.replace('display(Markdown("### Tabla C1. Performance y riesgo"))', 'display(Markdown("### Tabla 4 — Comparación final de estrategias"))')
source24 = source24.replace('display(Markdown("### Tabla C2. Implementabilidad y exposición"))', 'display(Markdown("### Tabla 5 — Atribución económica e implementabilidad"))')
set_source(nb, 24, source24)

source26 = cell_source(nb, 26)
palette_block = '''COLOR = {
	"minvar": "#1f4e79",
	"minvar_no_crypto": "#8a6f3d",
	"sixty_forty": "#6c757d",
	"equal_weight": "#9aa0a6",
	"fixed_crypto": "#7b5ea7",
	"cvar": "#7a3b69",
	"cvar_no_crypto": "#b07aa1",
	"overlay": "#2f7f7f",
	"overlay_simple": "#3d8b8b",
	"overlay_ml": "#1f6f8b",
	"overlay_stress": "#b86b3c",
	"risk_contained": "#4f8a5f",
	"risk_contained_light": "#dcebdc",
	"high_stress": "#b85c5c",
	"high_stress_light": "#f3d7d7",
	"positive": "#4f8a5f",
	"negative": "#b85c5c",
	"mixed": "#c9a646",
	"neutral": "#d9dee3",
	"threshold": "#555555",
	"zero": "#333333",
	"grid": "#e6e6e6",
	"background": "#f7f7f7",
}

VIS_COLORS = {
	"60/40 tradicional": COLOR["sixty_forty"],
	"MinVar": COLOR["minvar"],
	"MinVar sin crypto": COLOR["minvar_no_crypto"],
	"Equal Weight": COLOR["equal_weight"],
	"Fixed Small Crypto": COLOR["fixed_crypto"],
	"CVaR": COLOR["cvar"],
	"CVaR sin crypto": COLOR["cvar_no_crypto"],
	"Overlay riesgo simple": COLOR["overlay_simple"],
	"Overlay riesgo ML": COLOR["overlay_ml"],
	"Overlay estrés": COLOR["overlay_stress"],
	"Overlay combinado": COLOR["overlay"],
}

VIS_REGIME_COLORS = {
	"Riesgo contenido": COLOR["risk_contained"],
	"Estrés alto": COLOR["high_stress"],
}
'''
source26 = re.sub(r'VIS_COLORS = \{.*?\}\n\nVIS_REGIME_NAMES =', palette_block + "\nVIS_REGIME_NAMES =", source26, flags=re.S)
source26 = re.sub(r'VIS_REGIME_COLORS = \{.*?\}\n\n', "", source26, flags=re.S)
source26 = source26.replace('"axes.facecolor": "#fbfbf8",', '"axes.facecolor": COLOR["background"],')
source26 = source26.replace('"Chapter 5 dir present:"', '"Directorio de overlay supervisado presente:"')

figure1 = r'''
fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def add_box(x, y, w, h, title, body, face, edge="#c7c7c7", title_size=10.5, body_size=8.7):
	box = FancyBboxPatch(
		(x, y), w, h,
		boxstyle="round,pad=0.012,rounding_size=0.018",
		linewidth=1.1,
		edgecolor=edge,
		facecolor=face,
	)
	ax.add_patch(box)
	ax.text(x + w / 2, y + h - 0.022, title, ha="center", va="top", fontsize=title_size, weight="bold", color="#1f1f1f")
	ax.text(x + 0.012, y + h - 0.052, body, ha="left", va="top", fontsize=body_size, color="#333333", linespacing=1.23)
	return box


def add_arrow(start, end, color=COLOR["threshold"], lw=1.2, alpha=0.9):
	ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=lw, color=color, alpha=alpha))


ax.text(0.5, 0.965, "Figura 1. Mapa reproducible del estudio: de código a evidencia", ha="center", va="top", fontsize=16, weight="bold")
ax.text(0.5, 0.934, "Cómo el proyecto transforma datos, configuración y módulos reproducibles en evidencia empírica y conclusión final", ha="center", va="top", fontsize=10.5, color="#555555")

layer_style = {"fontsize": 10, "weight": "bold", "color": "#444444"}
ax.text(0.025, 0.875, "1. Datos y configuración", ha="left", va="center", **layer_style)
ax.text(0.025, 0.705, "2. Motor cuantitativo común", ha="left", va="center", **layer_style)
ax.text(0.025, 0.520, "3. Bloques empíricos", ha="left", va="center", **layer_style)
ax.text(0.025, 0.333, "4. Outputs / evidencia", ha="left", va="center", **layer_style)
ax.text(0.025, 0.142, "5. Síntesis final", ha="left", va="center", **layer_style)

data_boxes = [
	add_box(0.12, 0.805, 0.22, 0.105, "Datos de mercado", "BTC\nETH\nSPY, QQQ, GLD, TLT", "#ffffff", COLOR["neutral"]),
	add_box(0.39, 0.805, 0.22, 0.105, "Configuración", "settings.yaml\nregime_analysis.yaml\nsupervised_risk_overlay.yaml", "#ffffff", COLOR["neutral"]),
	add_box(0.66, 0.805, 0.22, 0.105, "Calendario final", "business-day aligned\nrebalanceo mensual\ndrifted buy-and-hold", "#ffffff", COLOR["neutral"]),
]

engine_boxes = [
	add_box(0.12, 0.635, 0.22, 0.112, "Preparación de datos", "src/data_loader.py\nsrc/preprocessing.py", "#eef4f8", COLOR["minvar"]),
	add_box(0.39, 0.635, 0.22, 0.112, "Optimización y backtest", "src/optimizer.py\nsrc/backtest.py\nsrc/costs.py", "#eef4f8", COLOR["minvar"]),
	add_box(0.66, 0.635, 0.22, 0.112, "Métricas y comparación", "src/metrics.py\nsrc/benchmarks.py\nsrc/covariance.py", "#eef4f8", COLOR["minvar"]),
]

block_x = [0.06, 0.245, 0.43, 0.615, 0.80]
block_w = 0.155
blocks = [
	("Bloque 1 — Baseline", "MinVar vs 60/40\npesos y drawdowns\nscripts/run_backtest.py\nscripts/run_benchmarks.py", COLOR["sixty_forty"]),
	("Bloque 2 — Robustez", "control sin crypto\nsensibilidad / bootstrap\nsrc/robustness.py\nsrc/bootstrap.py\nscripts/run_robustness.py", COLOR["minvar_no_crypto"]),
	("Bloque 3 — CVaR y cola", "ES95\nstress windows\nsrc/cvar_optimizer.py\nsrc/stress.py\nscripts/run_tail_risk.py", COLOR["cvar"]),
	("Bloque 4 — Regímenes", "HMM / fallback\nperformance condicional\nsrc/regime_detection.py\nsrc/regime_evaluation.py\nscripts/run_regime_analysis.py", COLOR["high_stress"]),
	("Bloque 5 — Señales y overlay", "targets supervisados\nmodelos OOS\noverlay de riesgo\nsrc/supervised_models.py\nsrc/risk_overlay.py\nsrc/overlay_backtest.py\nscripts/run_chapter5.py", COLOR["overlay"]),
]
for x, (title, body, edge) in zip(block_x, blocks):
	add_box(x, 0.425, block_w, 0.155, title, body, "#ffffff", edge, title_size=9.1, body_size=7.5)

outputs = [
	("Baseline", "portfolio returns\nweights history\nbacktest summary"),
	("Robustez", "robustness outputs\nconfidence_summary.csv\ncrypto vs no-crypto delta"),
	("CVaR", "tail_risk_summary_net.csv\nstress summaries\nCVaR vs MinVar"),
	("Regímenes", "regime labels\nconditional performance\nregime metadata"),
	("Señales y overlay", "model_scores.csv\nmodel_selection.csv\noverlay_decisions.csv\noverlay_backtest_summary.csv"),
]
for x, (title, body) in zip(block_x, outputs):
	add_box(x, 0.235, block_w, 0.135, title, body, "#fbfbfb", COLOR["neutral"], title_size=9.0, body_size=7.4)

final_box = add_box(
	0.16, 0.055, 0.68, 0.135,
	"Síntesis final del estudio",
	"hallazgos por bloque; evidencia frente a hipótesis; comparación final de estrategias; historia visual; limitaciones; conclusión final\n\nMinVar aporta valor; crypto estructural no queda respaldada; crypto aparece como exposición táctica/intermitente.",
	"#f4f7f4", COLOR["risk_contained"], title_size=11.5, body_size=9.0,
)

for x in [0.23, 0.50, 0.77]:
	add_arrow((x, 0.805), (x, 0.748))
for x in [0.23, 0.50, 0.77]:
	add_arrow((x, 0.635), (x, 0.585))
for x in block_x:
	add_arrow((x + block_w / 2, 0.425), (x + block_w / 2, 0.370))
for x in block_x:
	add_arrow((x + block_w / 2, 0.235), (0.50, 0.190), alpha=0.55)

add_arrow((0.50, 0.635), (block_x[0] + block_w / 2, 0.580), lw=1.0, alpha=0.75)
add_arrow((0.50, 0.635), (block_x[1] + block_w / 2, 0.580), lw=1.0, alpha=0.75)
add_arrow((0.50, 0.635), (block_x[2] + block_w / 2, 0.580), lw=1.0, alpha=0.75)
add_arrow((0.50, 0.635), (block_x[3] + block_w / 2, 0.580), lw=1.0, alpha=0.75)
add_arrow((0.50, 0.635), (block_x[4] + block_w / 2, 0.580), lw=1.0, alpha=0.75)

ax.text(0.12, 0.607, "métricas, costes y benchmarks alimentan los contrastes", fontsize=8.6, color="#555555")
ax.text(0.71, 0.208, "outputs finales del bloque supervisado\nalimentan overlay y comparación final", fontsize=8.1, color="#555555", ha="center")

plt.show()

vis_show_reading(
	"Esta figura resume cómo el proyecto transforma datos de mercado y configuración reproducible en evidencia empírica. El análisis no depende de una única tabla final, sino de una cadena de módulos, scripts y outputs que alimentan los bloques del estudio.",
	"El motor común de datos, optimización, backtest, métricas y costes alimenta todos los bloques empíricos. Cada bloque añade una prueba distinta: baseline, robustez, CVaR, regímenes y overlay supervisado.",
	"La conclusión final no sale solo de comparar MinVar contra 60/40. Sale de contrastar si el efecto crypto sobrevive a controles sin crypto, costes, bootstrap, riesgo de cola, estados de mercado y señales supervisadas.",
	"El mapa muestra trazabilidad metodológica, no validez universal. La evidencia sigue dependiendo del universo, periodo, datos y supuestos de implementación.",
)
'''
source26 = re.sub(r'fig, ax = plt\.subplots\(figsize=\(16, 3\.8\)\).*?\)\n$', figure1.strip() + "\n", source26, flags=re.S)
set_source(nb, 26, source26)

source30 = cell_source(nb, 30)
source30 = source30.replace("Figura 6.3", "Figura 3")
source30 = source30.replace('color="#1f4e79"', 'color=COLOR["minvar"]')
source30 = source30.replace('color="#222222"', 'color=COLOR["zero"]')
source30 = source30.replace('color="#777777"', 'color=COLOR["threshold"]')
source30 = source30.replace('color="#8c6d31"', 'color=COLOR["minvar_no_crypto"]')
source30 = source30.replace('color="#444444"', 'color=COLOR["threshold"]')
source30 = source30.replace('cell.set_facecolor("#e9eef3")', 'cell.set_facecolor(COLOR["neutral"])')
source30 = source30.replace(
	'"El bootstrap solo evalúa estabilidad muestral aproximada; no convierte una mejora pequeña en asignación estructural.",',
	'"El bootstrap es una forma humana de preguntar si el delta observado parece estable o si puede ser ruido de muestra: positivo no significa material; material no significa robusto; robusto no significa estructural.",',
)
set_source(nb, 30, source30)

source34 = cell_source(nb, 34)
source34 = source34.replace("Figura 6.5", "Figura 5")
source34 = source34.replace(
	'family_order = [fam for fam in ["lookback", "crypto_cap", "covariance_method", "rebalance_frequency", "cost_sensitivity"] if fam in rob["family"].unique()]',
	'family_labels = {"lookback": "Ventana de estimación", "crypto_cap": "Límite máximo crypto", "covariance_method": "Método de covarianza", "rebalance_frequency": "Frecuencia de rebalanceo", "cost_sensitivity": "Costes de transacción"}\n    family_order = [fam for fam in ["lookback", "crypto_cap", "covariance_method", "rebalance_frequency", "cost_sensitivity"] if fam in rob["family"].unique()]',
)
source34 = source34.replace('fig, axes = plt.subplots(2, 2, figsize=(15.8, 8.4))', 'display_order = [family_labels.get(fam, str(fam).replace("_", " ").title()) for fam in family_order]\n\n    fig, axes = plt.subplots(2, 2, figsize=(15.8, 8.4))')
source34 = source34.replace('axes[0, 0].bar(delta_sharpe.index, delta_sharpe.values, color=["#1f4e79" if v >= 0 else "#8c6d31" for v in delta_sharpe.values])', 'axes[0, 0].bar(display_order[:len(delta_sharpe)], delta_sharpe.values, color=[COLOR["positive"] if v >= 0 else COLOR["negative"] for v in delta_sharpe.values])')
source34 = source34.replace('axes[0, 1].bar(delta_return.index, delta_return.values, color=["#1f4e79" if v >= 0 else "#8c6d31" for v in delta_return.values])', 'axes[0, 1].bar(display_order[:len(delta_return)], delta_return.values, color=[COLOR["positive"] if v >= 0 else COLOR["negative"] for v in delta_return.values])')
source34 = source34.replace('axes[1, 0].bar(delta_maxdd.index, delta_maxdd.values, color=["#1f4e79" if v >= 0 else "#8c6d31" for v in delta_maxdd.values])', 'axes[1, 0].bar(display_order[:len(delta_maxdd)], delta_maxdd.values, color=[COLOR["negative"] if v > 0 else COLOR["positive"] for v in delta_maxdd.values])')
source34 = source34.replace('color="#222222"', 'color=COLOR["zero"]')
source34 = source34.replace('cell.set_facecolor("#e9eef3")', 'cell.set_facecolor(COLOR["neutral"])')
source34 = source34.replace('"Deltas entre MinVar con crypto y MinVar sin crypto por familias de robustez."', '"Deltas frente al control sin crypto por familias comparables: ventana, límite máximo crypto, método de covarianza, frecuencia y costes."')
set_source(nb, 34, source34)

set_source(
	nb,
	36,
	r'''
if VIS_TAIL_NET is None or VIS_TAIL_NET.empty:
	vis_note("No está disponible `tail_risk_summary_net.csv`.")
	vis_show_reading(
		"La comparación entre MinVar y CVaR en retorno, cola y drawdown.",
		"La figura no puede renderizarse porque falta el resumen neto de tail risk.",
		"Sin esta tabla no puede evaluarse si CVaR domina o no a MinVar.",
		"La visual depende del resumen neto y de `stress_summary.csv` para estrés.",
	)
else:
	tail = VIS_TAIL_NET.copy()
	tail["Estrategia"] = tail["strategy"].map(vis_h_strategy)
	ordered = ["MinVar", "MinVar sin crypto", "CVaR", "CVaR sin crypto"]
	tail = tail.loc[tail["Estrategia"].isin(ordered)].copy()
	tail["order"] = tail["Estrategia"].map({name: idx for idx, name in enumerate(ordered)})
	tail = tail.sort_values("order")

	cvar_row = vis_pick_row(VIS_TAIL_NET, "strategy", "cvar_baseline")
	minvar_row = vis_pick_row(VIS_TAIL_NET, "strategy", "minvar_baseline_ch1")
	deltas = []
	if not cvar_row.empty and not minvar_row.empty:
		deltas = [
			("Delta Sharpe", vis_metric(cvar_row, "sharpe_net") - vis_metric(minvar_row, "sharpe_net"), "más alto es mejor"),
			("Delta ES95", (vis_metric(cvar_row, "expected_shortfall_net") - vis_metric(minvar_row, "expected_shortfall_net")) * 100.0, "negativo reduce pérdida"),
			("Delta MaxDD", (abs(vis_metric(cvar_row, "max_drawdown_net")) - abs(vis_metric(minvar_row, "max_drawdown_net"))) * 100.0, "negativo reduce pérdida"),
		]

	fig = plt.figure(figsize=(16.5, 7.4), constrained_layout=True)
	grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.86], width_ratios=[1.25, 1.05, 1.05])
	ax_scatter = fig.add_subplot(grid[:, 0])
	ax_table = fig.add_subplot(grid[:, 1])
	ax_cards = fig.add_subplot(grid[:, 2])

	for row in tail.itertuples():
		ax_scatter.scatter(row.ann_return_net * 100.0, row.expected_shortfall_net * 100.0, s=105, color=VIS_COLORS.get(row.Estrategia), label=row.Estrategia)
		ax_scatter.annotate(row.Estrategia, (row.ann_return_net * 100.0, row.expected_shortfall_net * 100.0), textcoords="offset points", xytext=(6, 6), fontsize=9)
	ax_scatter.set_title("Panel A. Retorno anual vs ES95", fontsize=12)
	ax_scatter.set_xlabel("Retorno anual (%)")
	ax_scatter.set_ylabel("ES95 (% pérdida; menor es mejor)")
	ax_scatter.grid(True, alpha=0.25, color=COLOR["grid"])

	compact = tail[["Estrategia", "sharpe_net", "expected_shortfall_net", "max_drawdown_net"]].copy()
	compact["Sharpe"] = compact["sharpe_net"].map(lambda x: f"{x:.3f}")
	compact["ES95"] = compact["expected_shortfall_net"].map(lambda x: f"{x * 100:.2f}%")
	compact["MaxDD"] = compact["max_drawdown_net"].map(lambda x: f"{abs(x) * 100:.1f}%")
	compact_view = compact[["Estrategia", "Sharpe", "ES95", "MaxDD"]]
	ax_table.axis("off")
	table = ax_table.table(cellText=compact_view.values, colLabels=compact_view.columns, cellLoc="left", colLoc="left", loc="center")
	table.auto_set_font_size(False)
	table.set_fontsize(9.5)
	table.scale(1.0, 1.48)
	for row_idx in range(len(compact_view) + 1):
		for col_idx in range(len(compact_view.columns)):
			cell = table[row_idx, col_idx]
			cell.set_edgecolor("#dddddd")
			if row_idx == 0:
				cell.set_facecolor(COLOR["neutral"])
				cell.set_text_props(weight="bold")
	ax_table.set_title("Panel B. Riesgo de cola y drawdown", fontsize=12, pad=8)

	ax_cards.axis("off")
	ax_cards.set_title("Panel C. Deltas CVaR - MinVar", fontsize=12, pad=8)
	if not deltas:
		ax_cards.text(0.5, 0.5, "Sin deltas CVaR-MinVar", ha="center", va="center", fontsize=11)
	else:
		y_positions = [0.72, 0.46, 0.20]
		for (label, value, note), y in zip(deltas, y_positions):
			better = value > 0 if label == "Delta Sharpe" else value < 0
			face = COLOR["risk_contained_light"] if better else COLOR["high_stress_light"]
			edge = COLOR["positive"] if better else COLOR["negative"]
			box = FancyBboxPatch((0.08, y), 0.84, 0.18, boxstyle="round,pad=0.018,rounding_size=0.02", linewidth=1.1, edgecolor=edge, facecolor=face)
			ax_cards.add_patch(box)
			value_text = f"{value:+.3f}" if label == "Delta Sharpe" else f"{value:+.2f} p.p."
			ax_cards.text(0.13, y + 0.118, label, ha="left", va="center", fontsize=10, weight="bold")
			ax_cards.text(0.87, y + 0.118, value_text, ha="right", va="center", fontsize=12, weight="bold")
			ax_cards.text(0.13, y + 0.045, note, ha="left", va="center", fontsize=8.7, color="#555555")
	ax_cards.set_xlim(0, 1)
	ax_cards.set_ylim(0, 1)

	fig.suptitle("Figura 6. CVaR y riesgo de cola", fontsize=15)
	plt.show()

	if not cvar_row.empty and not minvar_row.empty:
		obs_text = (
			f"CVaR entrega Sharpe neto {vis_metric(cvar_row, 'sharpe_net'):.3f} frente a {vis_metric(minvar_row, 'sharpe_net'):.3f} en MinVar, "
			f"con ES95 de {vis_metric(cvar_row, 'expected_shortfall_net') * 100.0:.2f}% frente a {vis_metric(minvar_row, 'expected_shortfall_net') * 100.0:.2f}%."
		)
	else:
		obs_text = "La comparación directa CVaR-MinVar no está completa."
	vis_show_reading(
		"Retorno frente a ES95, tabla compacta de Sharpe/ES95/MaxDD y deltas separados para evitar mezclar unidades en un único eje.",
		obs_text,
		"CVaR es una prueba de riesgo de cola. Si no mejora claramente ES95 o MaxDD frente a MinVar, no rescata la tesis estructural crypto.",
		"ES95 se muestra como pérdida positiva: menor es mejor. MaxDD se lee como magnitud de pérdida: menor magnitud es mejor.",
	)
''',
)

set_source(
	nb,
	38,
	r'''
regime_features = safe_read_csv(REGIME_DIR / "regime_features.csv", parse_dates=["date"])
regime_palette = {"Riesgo contenido": COLOR["risk_contained"], "Estrés alto": COLOR["high_stress"]}
regime_light = {"Riesgo contenido": COLOR["risk_contained_light"], "Estrés alto": COLOR["high_stress_light"]}


def _episode_stats(timeline: pd.DataFrame) -> pd.DataFrame:
	if timeline.empty:
		return pd.DataFrame(columns=["Régimen", "% observaciones", "Episodios", "Duración media", "Duración máxima"])
	ordered = timeline.sort_values("date").copy()
	ordered["episode"] = ordered["Régimen"].ne(ordered["Régimen"].shift()).cumsum()
	spans = ordered.groupby(["Régimen", "episode"]).agg(start=("date", "min"), end=("date", "max"), days=("date", "size")).reset_index()
	counts = ordered["Régimen"].value_counts(normalize=True)
	rows = []
	for regime_name in ["Riesgo contenido", "Estrés alto"]:
		sub = spans.loc[spans["Régimen"] == regime_name]
		rows.append({
			"Régimen": regime_name,
			"% observaciones": f"{counts.get(regime_name, 0.0) * 100:.1f}%",
			"Episodios": int(len(sub)),
			"Duración media": f"{sub['days'].mean():.0f} días" if not sub.empty else "0 días",
			"Duración máxima": f"{sub['days'].max():.0f} días" if not sub.empty else "0 días",
		})
	return pd.DataFrame(rows)


if (VIS_REGIME_LABELS is None or VIS_REGIME_LABELS.empty) and (VIS_REGIME_PERF is None or VIS_REGIME_PERF.empty):
	vis_note("Faltan `regime_labels.csv` y `regime_conditional_performance_net.csv`.")
	vis_show_reading(
		"La cronología de regímenes y su relación con performance y exposición crypto.",
		"La figura no puede renderizarse porque faltan los outputs básicos del análisis de regímenes.",
		"Sin esta capa solo queda la comparación agregada, sin contexto de entorno.",
		"Los regímenes se tratan aquí como diagnóstico histórico, no como señal operativa.",
	)
else:
	fig = plt.figure(figsize=(16.5, 9.0), constrained_layout=True)
	grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
	ax_indicator = fig.add_subplot(grid[0, :])
	ax_duration = fig.add_subplot(grid[1, 0])
	ax_perf = fig.add_subplot(grid[1, 1])

	regime_view = pd.DataFrame()
	if VIS_REGIME_LABELS is None or VIS_REGIME_LABELS.empty or regime_features is None or regime_features.empty:
		ax_indicator.text(0.5, 0.5, "Sin indicador observable de régimen", ha="center", va="center", fontsize=11)
		ax_indicator.axis("off")
	else:
		timeline = VIS_REGIME_LABELS.copy().sort_values("date")
		timeline["Régimen"] = timeline["regime_name"].map(vis_h_regime)
		feat = regime_features.copy().sort_values("date")
		indicator_col = "realized_vol_spy_63d" if "realized_vol_spy_63d" in feat.columns else "drawdown_spy_126d"
		regime_view = timeline.merge(feat[["date", indicator_col]], on="date", how="left")
		regime_view["Indicador"] = regime_view[indicator_col].abs() * 100.0
		regime_view["episode"] = regime_view["Régimen"].ne(regime_view["Régimen"].shift()).cumsum()

		for _, span in regime_view.groupby("episode"):
			regime_name = span["Régimen"].iloc[0]
			ax_indicator.axvspan(span["date"].min(), span["date"].max(), color=regime_light.get(regime_name, COLOR["neutral"]), alpha=0.65, linewidth=0)
		ax_indicator.plot(regime_view["date"], regime_view["Indicador"], color=COLOR["threshold"], linewidth=1.8, label="Volatilidad SPY 63d" if indicator_col == "realized_vol_spy_63d" else "Drawdown SPY 126d")
		ax_indicator.set_title("Panel A. Indicador observable con bandas de régimen", fontsize=12)
		ax_indicator.set_ylabel("Volatilidad SPY 63d (%)" if indicator_col == "realized_vol_spy_63d" else "Magnitud drawdown SPY 126d (%)")
		ax_indicator.set_xlabel("Fecha")
		ax_indicator.grid(True, alpha=0.25, color=COLOR["grid"])
		for label, date_hint in [("COVID", "2020-03-15"), ("Subidas de tipos", "2022-06-15")]:
			date = pd.Timestamp(date_hint)
			if regime_view["date"].min() <= date <= regime_view["date"].max():
				yval = regime_view.loc[(regime_view["date"] - date).abs().idxmin(), "Indicador"]
				ax_indicator.annotate(label, xy=(date, yval), xytext=(6, 12), textcoords="offset points", fontsize=8.5, color="#555555", arrowprops={"arrowstyle": "-", "color": COLOR["threshold"], "lw": 0.8})
		handles = [plt.Line2D([0], [0], color=regime_palette[name], lw=6, alpha=0.8, label=name) for name in ["Riesgo contenido", "Estrés alto"]]
		handles.append(plt.Line2D([0], [0], color=COLOR["threshold"], lw=1.8, label="Indicador observable"))
		ax_indicator.legend(handles=handles, frameon=False, loc="upper left")

	duration_table = _episode_stats(regime_view)
	if duration_table.empty:
		ax_duration.text(0.5, 0.5, "Sin estadística de episodios", ha="center", va="center", fontsize=11)
		ax_duration.axis("off")
	else:
		ax_duration.axis("off")
		table = ax_duration.table(cellText=duration_table.values, colLabels=duration_table.columns, cellLoc="left", colLoc="left", loc="center")
		table.auto_set_font_size(False)
		table.set_fontsize(9.2)
		table.scale(1.0, 1.45)
		for row_idx in range(len(duration_table) + 1):
			for col_idx in range(len(duration_table.columns)):
				cell = table[row_idx, col_idx]
				cell.set_edgecolor("#dddddd")
				if row_idx == 0:
					cell.set_facecolor(COLOR["neutral"])
					cell.set_text_props(weight="bold")
				elif col_idx == 0:
					regime_name = duration_table.iloc[row_idx - 1, 0]
					cell.set_facecolor(regime_light.get(regime_name, "white"))
		ax_duration.set_title("Panel B. Observaciones y duración de episodios", fontsize=12, pad=8)

	if VIS_REGIME_PERF is None or VIS_REGIME_PERF.empty:
		ax_perf.text(0.5, 0.5, "Sin performance condicional", ha="center", va="center", fontsize=11)
		ax_perf.axis("off")
	else:
		perf = VIS_REGIME_PERF.copy()
		perf["Estrategia"] = perf["strategy"].map(vis_h_strategy)
		perf["Régimen"] = perf["regime_name"].map(vis_h_regime)
		wanted = ["MinVar", "MinVar sin crypto", "CVaR", "Overlay combinado"]
		perf = perf.loc[perf["Estrategia"].isin(wanted)]
		x_positions = {"Riesgo contenido": 0, "Estrés alto": 1}
		for strategy in [name for name in wanted if name in perf["Estrategia"].unique()]:
			vals = []
			for regime_name in ["Riesgo contenido", "Estrés alto"]:
				value = perf.loc[(perf["Estrategia"] == strategy) & (perf["Régimen"] == regime_name), "sharpe"].mean()
				vals.append(value)
			if all(pd.notna(vals)):
				ax_perf.plot([0, 1], vals, marker="o", linewidth=2.0, color=VIS_COLORS.get(strategy), label=strategy)
				ax_perf.text(1.02, vals[-1], strategy, va="center", fontsize=8.5, color=VIS_COLORS.get(strategy))
		ax_perf.set_xticks([0, 1], ["Riesgo contenido", "Estrés alto"])
		ax_perf.set_title("Panel C. Sharpe condicional por régimen", fontsize=12)
		ax_perf.set_ylabel("Sharpe")
		ax_perf.grid(True, axis="y", alpha=0.25, color=COLOR["grid"])
		ax_perf.axhline(0, color=COLOR["zero"], linewidth=0.9)

	fig.suptitle("Figura 7. Regímenes de mercado", fontsize=15)
	plt.show()

	crypto_regime_text = ""
	if VIS_REGIME_CRYPTO is not None and not VIS_REGIME_CRYPTO.empty:
		expo = VIS_REGIME_CRYPTO.copy()
		expo["Estrategia"] = expo["strategy"].map(vis_h_strategy)
		expo["Régimen"] = expo["regime_name"].map(vis_h_regime)
		mv_expo = expo.loc[expo["Estrategia"] == "MinVar"]
		if not mv_expo.empty:
			pieces = [f"{row.Régimen}: {row.mean_crypto_weight * 100:.2f}%" for row in mv_expo.itertuples()]
			crypto_regime_text = " Exposición crypto media MinVar por régimen: " + "; ".join(pieces) + "."

	vis_show_reading(
		"Volatilidad realizada SPY 63d como indicador observable, bandas de riesgo contenido/estrés alto, duración de episodios y Sharpe condicional.",
		"Las bandas rojas ubican episodios de estrés y las verdes periodos de riesgo contenido; el rendimiento condicional cae en estrés." + crypto_regime_text,
		"Los regímenes muestran cuándo el mercado entra en estrés y cómo cambia el rendimiento de las estrategias, pero son diagnóstico histórico, no señal live validada.",
		"No demuestra timing ex ante; una mejora futura razonable sería validar reglas de transición con datos posteriores no usados en la definición de estados.",
	)
''',
)

source40 = cell_source(nb, 40)
source40 = source40.replace("Figura 6.8", "Figura 8")
source40 = source40.replace('cell.set_facecolor("#e9eef3")', 'cell.set_facecolor(COLOR["neutral"])')
source40 = source40.replace('cell.set_facecolor("#d9e6dd" if selection_view.iloc[row_idx - 1, col_idx] == "Sí" else "#f2ead3")', 'cell.set_facecolor(COLOR["risk_contained_light"] if selection_view.iloc[row_idx - 1, col_idx] == "Sí" else "#f2ead3")')
source40 = source40.replace('color="#444444"', 'color=COLOR["threshold"]')
source40 = source40.replace('ax_bucket.plot(sub["bucket"], sub["mean_pred"], marker="o", linewidth=2.0, label=f"Predicho: {label}")', 'ax_bucket.plot(sub["bucket"], sub["mean_pred"], marker="o", linewidth=2.0, color=COLOR["overlay_ml"], label=f"Predicho: {label}")')
source40 = source40.replace('ax_bucket.plot(sub["bucket"], sub["mean_realized"], marker="s", linewidth=1.8, linestyle="--", label=f"Realizado: {label}")', 'ax_bucket.plot(sub["bucket"], sub["mean_realized"], marker="s", linewidth=1.8, linestyle="--", color=COLOR["sixty_forty"], label=f"Realizado: {label}")')
set_source(nb, 40, source40)

source42 = cell_source(nb, 42)
source42 = source42.replace("Figura 6.9", "Figura 9")
source42 = source42.replace('color="#2f4b7c"', 'color=COLOR["minvar"]')
source42 = source42.replace('color="#a0513a"', 'color=COLOR["overlay"]')
source42 = source42.replace('color="#8c6d31"', 'color=COLOR["overlay_stress"]')
source42 = source42.replace('color="#d8c49a"', 'color=COLOR["high_stress_light"]')
source42 = source42.replace('color="#5f7f8f"', 'color=COLOR["overlay"]')
source42 = source42.replace('cell.set_facecolor("#e9eef3")', 'cell.set_facecolor(COLOR["neutral"])')
set_source(nb, 42, source42)

set_source(
	nb,
	46,
	r'''
rows = []

bench_6040 = vis_pick_row(VIS_BENCH, "benchmark", "sixty_forty")
backtest_minvar = vis_pick_row(VIS_BACKTEST, "strategy", "min_variance")
if not bench_6040.empty and not backtest_minvar.empty:
	evidence_minvar = f"Sharpe {vis_metric(backtest_minvar, 'sharpe'):.2f} vs {vis_metric(bench_6040, 'sharpe'):.2f}; retorno {vis_metric(backtest_minvar, 'ann_return') * 100.0:.1f}% vs {vis_metric(bench_6040, 'ann_return') * 100.0:.1f}%."
else:
	evidence_minvar = "Comparación no disponible."
rows.append(["MinVar mejora al 60/40", "Favorable", evidence_minvar, "La mejora pertenece primero a la construcción MinVar, no a crypto por atribución automática."])

crypto_evidence = "Control sin crypto no disponible."
if not VIS_BASELINE_ROW.empty and not VIS_NO_CRYPTO_ROW.empty:
	delta_sharpe = vis_metric(VIS_BASELINE_ROW, "sharpe") - vis_metric(VIS_NO_CRYPTO_ROW, "sharpe")
	crypto_evidence = f"Delta Sharpe {delta_sharpe:+.3f}"
	if VIS_CONFIDENCE is not None and not VIS_CONFIDENCE.empty:
		anchor = VIS_CONFIDENCE.loc[VIS_CONFIDENCE["comparison_id"] == "C1_anchor_pair"]
		if not anchor.empty:
			crypto_evidence = f"Delta Sharpe {delta_sharpe:+.3f}; IC [{float(anchor.iloc[0]['ci_lower']):+.3f}, {float(anchor.iloc[0]['ci_upper']):+.3f}]"
rows.append(["Crypto no explica robustamente la mejora", "Negativa", crypto_evidence, "Un delta pequeño con intervalo que cruza cero no respalda contribución estructural de BTC/ETH."])

crypto_weight_evidence = "Sin panel de pesos."
if not VIS_CRYPTO_PANEL.empty:
	focus = VIS_CRYPTO_PANEL.loc[VIS_CRYPTO_PANEL["Estrategia"] == "MinVar", "crypto_total"]
	if not focus.empty:
		crypto_weight_evidence = f"Peso medio {focus.mean() * 100.0:.2f}%; >2% en {(focus > 0.02).mean() * 100.0:.1f}%"
rows.append(["No hay evidencia estructural", "Negativa", crypto_weight_evidence, "La exposición no aparece con frecuencia y tamaño suficientes para llamarla asignación estructural."])
rows.append(["Compatible con exposición táctica", "Mixta", crypto_weight_evidence, "BTC/ETH pueden aparecer de forma intermitente, sin que eso implique tesis estratégica."])

cvar_evidence = "Comparación no disponible."
if VIS_TAIL_NET is not None and not VIS_TAIL_NET.empty:
	cvar_row = vis_pick_row(VIS_TAIL_NET, "strategy", "cvar_baseline")
	minvar_row = vis_pick_row(VIS_TAIL_NET, "strategy", "minvar_baseline_ch1")
	if not cvar_row.empty and not minvar_row.empty:
		cvar_evidence = f"Sharpe {vis_metric(cvar_row, 'sharpe_net'):.3f} vs {vis_metric(minvar_row, 'sharpe_net'):.3f}; ES95 {vis_metric(cvar_row, 'expected_shortfall_net') * 100.0:.2f}% vs {vis_metric(minvar_row, 'expected_shortfall_net') * 100.0:.2f}%"
rows.append(["CVaR no cambia la conclusión", "Mixta", cvar_evidence, "La prueba de cola no rescata la tesis crypto estructural si ES95/MaxDD no mejoran claramente."])

regime_evidence = "Performance por régimen no disponible."
if VIS_REGIME_PERF is not None and not VIS_REGIME_PERF.empty:
	mv_reg = VIS_REGIME_PERF.loc[VIS_REGIME_PERF["strategy"] == "minvar_baseline_ch1"]
	low = mv_reg.loc[mv_reg["regime_name"] == "Low-stress / Risk-on", "sharpe"]
	high = mv_reg.loc[mv_reg["regime_name"] == "High-stress / Risk-off", "sharpe"]
	if not low.empty and not high.empty:
		regime_evidence = f"Sharpe {float(low.iloc[0]):.2f} riesgo contenido vs {float(high.iloc[0]):.2f} estrés"
rows.append(["Regímenes contextualizan", "Favorable", regime_evidence, "Ayudan a leer cuándo cambia el riesgo, pero no son una señal live validada."])

overlay_evidence = "Comparación no disponible."
if not VIS_BASELINE_ROW.empty and not VIS_OVERLAY_ROW.empty:
	delta_sharpe_overlay = vis_metric(VIS_OVERLAY_ROW, "sharpe") - vis_metric(VIS_BASELINE_ROW, "sharpe")
	delta_es_overlay = vis_metric(VIS_OVERLAY_ROW, "es95") - vis_metric(VIS_BASELINE_ROW, "es95")
	overlay_evidence = f"Delta Sharpe {delta_sharpe_overlay:+.3f}; Delta ES95 {delta_es_overlay * 100.0:+.2f} p.p."
rows.append(["Overlay no monetiza robustamente", "Mixta", overlay_evidence, "La capa es trazable y prudencial, pero debe mejorar métricas netas para tener valor económico claro."])

dashboard = pd.DataFrame(rows, columns=["Afirmación", "Veredicto", "Evidencia numérica", "Lectura"])
verdict_colors = {"Favorable": COLOR["risk_contained_light"], "Negativa": COLOR["high_stress_light"], "Mixta": "#f2ead3"}
verdict_edges = {"Favorable": COLOR["positive"], "Negativa": COLOR["negative"], "Mixta": COLOR["mixed"]}

fig, ax_cards = plt.subplots(figsize=(16.0, 7.2))
ax_cards.axis("off")
card_rows = dashboard.copy()
x_positions = [0.18, 0.50, 0.82]
y_positions = [0.72, 0.42, 0.12]
for idx, row in enumerate(card_rows.itertuples(index=False)):
	x = x_positions[idx % 3]
	y = y_positions[idx // 3]
	face = verdict_colors.get(row.Veredicto, COLOR["neutral"])
	edge = verdict_edges.get(row.Veredicto, COLOR["threshold"])
	box = FancyBboxPatch((x - 0.135, y), 0.27, 0.20, boxstyle="round,pad=0.02,rounding_size=0.025", linewidth=1.2, edgecolor=edge, facecolor=face)
	ax_cards.add_patch(box)
	ax_cards.text(x, y + 0.132, row.Afirmación, ha="center", va="center", fontsize=10.8, weight="bold", wrap=True)
	ax_cards.text(x, y + 0.060, row.Veredicto, ha="center", va="center", fontsize=11.2, weight="bold", color="#333333")
ax_cards.set_xlim(0, 1)
ax_cards.set_ylim(0, 1)
ax_cards.set_title("Figura 10. Veredictos principales", fontsize=15, pad=12)
plt.show()

display(Markdown("### Tabla 6 — Evidencia numérica final"))
display(dashboard[["Afirmación", "Evidencia numérica", "Lectura"]].style.hide(axis="index"))

vis_show_reading(
	"Tarjetas de veredicto separadas de la evidencia numérica para evitar texto cortado y solapamientos.",
	"El cierre visual mantiene la secuencia: MinVar sí; crypto estructural no; CVaR no rescata; regímenes contextualizan; overlay no monetiza de forma robusta.",
	"La evidencia principal es metodológicamente defendible, no universal: está acotada por muestra, universo, costes y especificación.",
	"La tabla final resume números clave; no reemplaza las figuras anteriores ni sus cautelas metodológicas.",
)
''',
)

markdown_replacements = {
	"# Notebook 06 — Nota de investigación final": "# Reporte final de investigación",
	"Notebook 06": "Reporte final de investigación",
	"Capítulo 6": "Síntesis final",
	"Figura 6.10B": "Tabla 6",
	"Figura 6.10A": "Figura 10",
	"Figura 6.10": "Figura 10",
	"Figura 6.9": "Figura 9",
	"Figura 6.8": "Figura 8",
	"Figura 6.7": "Figura 7",
	"Figura 6.6": "Figura 6",
	"Figura 6.5": "Figura 5",
	"Figura 6.4": "Figura 4",
	"Figura 6.3": "Figura 3",
	"Figura 6.2": "Figura 2",
	"Figura 6.1": "Figura 1",
	"Tabla A": "Tabla 1",
	"Tabla B": "Tabla 2",
	"Tabla E": "Tabla 3",
	"Tabla C1": "Tabla 4",
	"Tabla C2": "Tabla 5",
	"Hallazgos por capítulo": "Hallazgos por bloque",
	"capítulos 1 a 5": "bloques empíricos previos",
	"notebooks de capítulos 1 a 5": "notebooks empíricos previos",
	"Chapter-by-Chapter Findings": "hallazgos por bloque",
}

for old, new in markdown_replacements.items():
	replace_in_source(nb, old, new)

nb = recursive_replace(nb, {
	"Figura 6.10B": "Tabla 6",
	"Figura 6.10A": "Figura 10",
	"Figura 6.10": "Figura 10",
	"Figura 6.9": "Figura 9",
	"Figura 6.8": "Figura 8",
	"Figura 6.7": "Figura 7",
	"Figura 6.6": "Figura 6",
	"Figura 6.5": "Figura 5",
	"Figura 6.4": "Figura 4",
	"Figura 6.3": "Figura 3",
	"Figura 6.2": "Figura 2",
	"Figura 6.1": "Figura 1",
	"Notebook 06": "Reporte final de investigación",
	"Capítulo 6": "Síntesis final",
})

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Updated {NOTEBOOK}")
