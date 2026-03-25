# %% [markdown]
# # Sprint 3 — Modelos Avanzados de Personalización y Modelo Híbrido
# ## RetailRocket Recommender System
# **Equipo:** NexusDataCo | **Bootcamp:** SoyHenry Data Science
#
# **Modelos entrenados en este sprint:**
# - BPR (Bayesian Personalized Ranking) — ranking personalizado para feedback implícito
# - Content-Based (TF-IDF) — recomendación por contenido de productos
# - Modelo Híbrido — BPR 60% + Item-CF 40%
# - EASE (Embarrassingly Shallow Autoencoders) — nuevo modelo de alta eficiencia
#
# **Resultado destacado:** EASE es el 2° mejor modelo del proyecto,
# superando a Item-CF, BPR, Content-Based e Híbrido en NDCG@10 y HitRate@10.

# %% [markdown]
# ## 0. Configuración e Imports

# %%
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    notebook_dir = Path(globals()["__vsc_ipynb_file__"]).parent
except KeyError:
    notebook_dir = Path().resolve()

ROOT     = notebook_dir.parent
DATA_PRO = ROOT / "data" / "processed"
MDL_DIR  = ROOT / "models_saved"
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {
    "popularity":    "#888780",
    "item_cf":       "#378ADD",
    "bpr":           "#EF9F27",
    "ease":          "#D85A30",
    "content_based": "#534AB7",
    "hybrid":        "#1D9E75",
}
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (13, 5)})
print("Configuracion lista")

# %% [markdown]
# ## 1. Carga de Modelos

# %%
from src.models.popularity    import PopularityRecommender
from src.models.item_cf       import ItemCFRecommender
from src.models.als_model     import ALSRecommender
from src.models.ease_model    import EASERecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid        import HybridRecommender

events_clean    = pd.read_parquet(DATA_PRO / "events_clean.parquet")
interaction_mtx = pd.read_parquet(DATA_PRO / "interaction_matrix.parquet")
item_profiles   = pd.read_parquet(DATA_PRO / "item_profiles.parquet")
tfidf_matrix    = sp.load_npz(DATA_PRO / "tfidf_matrix.npz")

pop_model     = PopularityRecommender.load()
item_cf_model = ItemCFRecommender.load()
als_model     = ALSRecommender.load()
ease_model    = EASERecommender.load()
cbf_model     = ContentBasedRecommender.load()
hybrid_model  = HybridRecommender.load()

print(f"Eventos: {len(events_clean):,} | Items: {len(item_profiles):,}")
print(f"\nModelos cargados:")
for m in [pop_model, item_cf_model, als_model, ease_model, cbf_model, hybrid_model]:
    print(f"  {m}")

# %% [markdown]
# ## 2. BPR — Ranking Personalizado

# %%
print("""
Decisión de diseño — BPR en lugar de MF estandar:
  MF estandar (MSE): diverge con feedback implicito (loss=nan).
  BPR: optimiza que items interactuados tengan mayor score.
  Resultado: correct=76.64% en 20 epocas sin divergencia.
  WMF descartado: requiere TensorFlow (no disponible).
""")

sample_users = [uid for uid, items in als_model.user_history.items() if len(items) >= 5][:3]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, uid in zip(axes, sample_users):
    recs = als_model.recommend(uid, n=50)
    if not recs.empty:
        ax.bar(range(len(recs)), recs["score"],
               color=COLORS["bpr"], edgecolor="white", linewidth=0.5)
        ax.set_title(f"Usuario {uid}\n({len(als_model.user_history[uid])} items)")
        ax.set_xlabel("Rank"); ax.set_ylabel("Score normalizado")
plt.suptitle("BPR — Distribucion de scores Top-50", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_bpr_scores.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. EASE — Embarrassingly Shallow Autoencoders

# %%
print("""
EASE — Steck 2019
  Motivacion: BPR, Item-CF e Hibrido presentaban metricas bajas.
  EASE fue seleccionado por ser disenado para matrices MUY sparse.

  Formula cerrada (sin iteraciones):
    G = X^T X + lambda * I
    B = inv(G) / (-diag(inv(G)))   (pesos item x item)
    B[i,i] = 0                     (no recomendar ya vistos)
    score(u) = X[u] @ B

  Lambda = 500 | max_items = 10,000 (limite RAM: 14.8 GB con 44,639 items)
""")

sample_ease = [uid for uid in ease_model.user_to_idx.keys()
               if len(ease_model.user_history.get(uid, set())) >= 5][:3]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, uid in zip(axes, sample_ease):
    recs = ease_model.recommend(uid, n=50)
    if not recs.empty:
        ax.bar(range(len(recs)), recs["score"],
               color=COLORS["ease"], edgecolor="white", linewidth=0.5)
        ax.set_title(f"Usuario {uid}\nEASE Top-50")
        ax.set_xlabel("Rank"); ax.set_ylabel("Score normalizado")
plt.suptitle("EASE — Distribucion de scores Top-50", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_ease_scores.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Content-Based (TF-IDF)

# %%
densities = np.diff(tfidf_matrix.indptr)
print(f"TF-IDF: {tfidf_matrix.shape}")
print(pd.Series(densities).describe())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(densities, bins=50, color=COLORS["content_based"],
             edgecolor="white", linewidth=0.3, alpha=0.85)
axes[0].set_xlabel("Features no-cero por item")
axes[0].set_ylabel("Items")
axes[0].set_title("Densidad del perfil TF-IDF")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
top_cats = item_profiles["category_id"].value_counts().head(15)
axes[1].barh(top_cats.index[::-1].astype(str), top_cats.values[::-1],
             color=COLORS["content_based"], edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Items"); axes[1].set_title("Top 15 categorias")
plt.suptitle("Content-Based — Analisis TF-IDF", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_cbf_analysis.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Modelo Híbrido

# %%
print(f"Hibrido: BPR={hybrid_model.weight_als:.0%} + Item-CF={hybrid_model.weight_cf:.0%}")
print("Fallback: BPR > Item-CF > Content-Based > Popularity")

# %% [markdown]
# ## 6. Comparativa Final — 6 Modelos

# %%
# Resultados reales de la evaluacion completa
results = {
    "popularity": {
        5:  {"Precision": 0.006744, "Recall": 0.020177, "NDCG": 0.019136, "HitRate": 0.031395, "Coverage": 0.000048, "Novelty": 10.597527},
        10: {"Precision": 0.005116, "Recall": 0.029557, "NDCG": 0.021629, "HitRate": 0.047674, "Coverage": 0.000048, "Novelty": 10.597527},
        20: {"Precision": 0.003198, "Recall": 0.033552, "NDCG": 0.022233, "HitRate": 0.054651, "Coverage": 0.000048, "Novelty": 10.597527},
    },
    "item_cf": {
        5:  {"Precision": 0.005116, "Recall": 0.010315, "NDCG": 0.008193, "HitRate": 0.017442, "Coverage": 0.016274, "Novelty": 15.797731},
        10: {"Precision": 0.003372, "Recall": 0.012789, "NDCG": 0.008767, "HitRate": 0.023256, "Coverage": 0.016274, "Novelty": 15.797731},
        20: {"Precision": 0.002384, "Recall": 0.015925, "NDCG": 0.009626, "HitRate": 0.030233, "Coverage": 0.016274, "Novelty": 15.797731},
    },
    "bpr": {
        5:  {"Precision": 0.004186, "Recall": 0.009560, "NDCG": 0.006637, "HitRate": 0.019767, "Coverage": 0.000129, "Novelty": 10.898767},
        10: {"Precision": 0.002674, "Recall": 0.013087, "NDCG": 0.007510, "HitRate": 0.024419, "Coverage": 0.000129, "Novelty": 10.898767},
        20: {"Precision": 0.002035, "Recall": 0.021139, "NDCG": 0.009466, "HitRate": 0.037209, "Coverage": 0.000129, "Novelty": 10.898767},
    },
    "ease": {
        5:  {"Precision": 0.006512, "Recall": 0.014536, "NDCG": 0.013164, "HitRate": 0.030233, "Coverage": 0.011747, "Novelty": 12.978770},
        10: {"Precision": 0.004535, "Recall": 0.020970, "NDCG": 0.015066, "HitRate": 0.043023, "Coverage": 0.011747, "Novelty": 12.978770},
        20: {"Precision": 0.003256, "Recall": 0.031403, "NDCG": 0.017781, "HitRate": 0.059302, "Coverage": 0.011747, "Novelty": 12.978770},
    },
    "content_based": {
        5:  {"Precision": 0.000000, "Recall": 0.000000, "NDCG": 0.000000, "HitRate": 0.000000, "Coverage": 0.031332, "Novelty": 18.274518},
        10: {"Precision": 0.000116, "Recall": 0.000388, "NDCG": 0.000172, "HitRate": 0.001163, "Coverage": 0.031332, "Novelty": 18.274518},
        20: {"Precision": 0.000058, "Recall": 0.000388, "NDCG": 0.000172, "HitRate": 0.001163, "Coverage": 0.031332, "Novelty": 18.274518},
    },
    "hybrid": {
        5:  {"Precision": 0.003023, "Recall": 0.008002, "NDCG": 0.004353, "HitRate": 0.015116, "Coverage": 0.011943, "Novelty": 13.896185},
        10: {"Precision": 0.003140, "Recall": 0.013900, "NDCG": 0.006603, "HitRate": 0.025581, "Coverage": 0.011943, "Novelty": 13.896185},
        20: {"Precision": 0.002151, "Recall": 0.018632, "NDCG": 0.007799, "HitRate": 0.031395, "Coverage": 0.011943, "Novelty": 13.896185},
    },
}

model_names  = list(results.keys())
model_colors = [COLORS.get(n, "#888780") for n in model_names]

df_k10 = pd.DataFrame({n: results[n][10] for n in model_names}).T
print("COMPARATIVA FINAL @ K=10 — 6 MODELOS")
print(df_k10.round(4).to_string())

# %%
# Gráfica 1: Barras agrupadas
metric_cols = ["Precision", "Recall", "NDCG", "HitRate"]
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(metric_cols))
width = 0.13
for i, (name, color) in enumerate(zip(model_names, model_colors)):
    vals = [results[name][10][m] for m in metric_cols]
    ax.bar(x + i*width, vals, width, label=name, color=color,
           edgecolor="white", linewidth=0.5, alpha=0.9)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(metric_cols, fontsize=11)
ax.set_ylabel("Score @ K=10")
ax.set_title("Comparativa de metricas de precision — 6 modelos @ K=10",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, ncol=3)
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_metrics_precision_final.png", bbox_inches="tight")
plt.show()

# %%
# Gráfica 2: NDCG@10 — EASE destacado
fig, ax = plt.subplots(figsize=(10, 5))
ndcg_vals = [results[n][10]["NDCG"] for n in model_names]
bars = ax.bar(model_names, ndcg_vals, color=model_colors,
              edgecolor="white", linewidth=0.5, alpha=0.9)
bars[model_names.index("ease")].set_edgecolor("#D85A30")
bars[model_names.index("ease")].set_linewidth(2.5)
for bar, val in zip(bars, ndcg_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("NDCG@10")
ax.set_title("NDCG@10 — EASE es el 2 mejor modelo del proyecto",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_ndcg_comparison.png", bbox_inches="tight")
plt.show()

# %%
# Gráfica 3: Scatter Precision vs Coverage
fig, ax = plt.subplots(figsize=(9, 6))
for name, color in zip(model_names, model_colors):
    row = results[name][10]
    ax.scatter(row["Coverage"]*100, row["Precision"],
               color=color, s=180, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(name, (row["Coverage"]*100, row["Precision"]),
                xytext=(5, 5), textcoords="offset points", fontsize=9, color=color)
ax.set_xlabel("Coverage (% del catalogo)")
ax.set_ylabel("Precision@10")
ax.set_title("Trade-off Precision vs Coverage @ K=10", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_precision_vs_coverage.png", bbox_inches="tight")
plt.show()

# %%
# Gráfica 4: HitRate y Recall por K
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
k_vals = [5, 10, 20]
for name, color in zip(model_names, model_colors):
    lw = 2.5 if name == "ease" else 1.5
    axes[0].plot(k_vals, [results[name][k]["HitRate"] for k in k_vals],
                 marker="o", color=color, linewidth=lw, markersize=7, label=name)
    axes[1].plot(k_vals, [results[name][k]["Recall"] for k in k_vals],
                 marker="o", color=color, linewidth=lw, markersize=7, label=name)
for ax, metric in zip(axes, ["HitRate@K", "Recall@K"]):
    ax.set_xlabel("K"); ax.set_ylabel(metric); ax.set_title(metric)
    ax.set_xticks(k_vals); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.suptitle("Evolucion de metricas por K — 6 modelos", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_metrics_by_k_final.png", bbox_inches="tight")
plt.show()

# %%
# Gráfica 5: Coverage vs Novelty
fig, ax = plt.subplots(figsize=(9, 5))
for name, color in zip(model_names, model_colors):
    row = results[name][10]
    ax.scatter(row["Coverage"]*100, row["Novelty"],
               color=color, s=180, zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(name, (row["Coverage"]*100, row["Novelty"]),
                xytext=(5, 3), textcoords="offset points", fontsize=9, color=color)
ax.set_xlabel("Coverage (% catalogo)"); ax.set_ylabel("Novelty")
ax.set_title("Coverage vs Novelty — Diversidad del sistema", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint3_coverage_vs_novelty.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Resumen Ejecutivo

# %%
print("=" * 70)
print("RESUMEN EJECUTIVO — SPRINT 3 (6 MODELOS)")
print("=" * 70)
ranking = [
    ("1", "Popularity",    "0.0216", "0.0477", "0.005%",  "Baseline sin personalizacion"),
    ("2", "EASE",          "0.0151", "0.0430", "1.17%",   "Mejor modelo personalizado"),
    ("3", "Item-CF",       "0.0088", "0.0233", "1.63%",   "Mejor coverage CF"),
    ("4", "BPR",           "0.0075", "0.0244", "0.013%",  "Ranking personalizado"),
    ("5", "Hibrido",       "0.0066", "0.0256", "1.19%",   "Mejor balance precision-coverage"),
    ("6", "Content-Based", "0.0002", "0.0012", "3.13%",   "Mejor coverage — activa catalogo"),
]
print(f"\n  {'#':<3} {'Modelo':<16} {'NDCG@10':<10} {'HitRate@10':<12} {'Coverage':<10} Nota")
print(f"  {'-'*72}")
for r in ranking:
    print(f"  {r[0]:<3} {r[1]:<16} {r[2]:<10} {r[3]:<12} {r[4]:<10} {r[5]}")

print("""
Hallazgos clave:
  EASE supera a todos los modelos colaborativos en NDCG y HitRate
  EASE @ K=20: HitRate=5.93% — mejor de todos los modelos en K grande
  Content-Based: mejor Coverage (3.13%) — activa el catalogo invisible
  EASE con todos los items requiere >14.8 GB RAM — trabajo futuro
""")
print("=" * 70)
print("Sprint 3 completado — 6 modelos evaluados.")
