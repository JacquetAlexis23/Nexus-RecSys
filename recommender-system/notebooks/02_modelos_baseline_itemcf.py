# %% [markdown]
# # Sprint 2 — Modelos Baseline y Collaborative Filtering
# ## RetailRocket Recommender System
# **Equipo:** NexusDataCo | **Bootcamp:** SoyHenry Data Science
#
# **Objetivo:** Entrenar y evaluar el modelo de popularidad (baseline) y el modelo
# Item-CF, comparar sus métricas y extraer conclusiones para el Sprint 3.

# %% [markdown]
# ## 0. Configuración e Imports

# %%
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Directorio de trabajo ──────────────────────────────────────────────────
try:
    notebook_dir = Path(globals()["__vsc_ipynb_file__"]).parent
except KeyError:
    notebook_dir = Path().resolve()

ROOT     = notebook_dir.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PRO = ROOT / "data" / "processed"
MDL_DIR  = ROOT / "models_saved"
os.chdir(ROOT)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Estilo ─────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {
    "popularity": "#378ADD",
    "item_cf":    "#1D9E75",
    "hybrid":     "#EF9F27",
    "neutral":    "#888780",
}
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (12, 5)})

print("✅ Configuración lista")
print(f"   ROOT:    {ROOT}")
print(f"   DATA_PRO: {DATA_PRO}")
print(f"   MDL_DIR:  {MDL_DIR}")

# %% [markdown]
# ## 1. Carga de Artefactos del Pipeline ETL

# %%
print("Cargando artefactos del Sprint 1...")

events_clean    = pd.read_parquet(DATA_PRO / "events_clean.parquet")
interaction_mtx = pd.read_parquet(DATA_PRO / "interaction_matrix.parquet")
item_profiles   = pd.read_parquet(DATA_PRO / "item_profiles.parquet")

print(f"\nEventos limpios:       {len(events_clean):,} filas")
print(f"Matriz interacciones:  {len(interaction_mtx):,} pares usuario-ítem")
print(f"  Usuarios únicos:     {interaction_mtx['visitorid'].nunique():,}")
print(f"  Ítems únicos:        {interaction_mtx['itemid'].nunique():,}")
print(f"Perfiles de ítems:     {len(item_profiles):,}")
print(f"\nColumnas interaction_matrix:\n{interaction_mtx.dtypes}")
print(f"\nMuestra:\n{interaction_mtx.head()}")

# %% [markdown]
# ## 2. Split Temporal Train / Test

# %%
from src.evaluation.metrics import temporal_train_test_split

train_events, test_events = temporal_train_test_split(events_clean, test_days=30)

print(f"\nTrain: {len(train_events):,} eventos")
print(f"  Período: {train_events['datetime'].min().date()} → {train_events['datetime'].max().date()}")
print(f"  Usuarios únicos: {train_events['visitorid'].nunique():,}")
print(f"\nTest:  {len(test_events):,} eventos")
print(f"  Período: {test_events['datetime'].min().date()} → {test_events['datetime'].max().date()}")
print(f"  Usuarios únicos: {test_events['visitorid'].nunique():,}")

# Ground truth: addtocart + transaction en el período de test
test_purchases = test_events[
    test_events["event"].isin(["addtocart", "transaction"])
][["visitorid", "itemid"]].drop_duplicates()

print(f"\nGround truth (test):")
print(f"  Usuarios con compras/carritos: {test_purchases['visitorid'].nunique():,}")
print(f"  Pares usuario-ítem relevantes: {len(test_purchases):,}")

# %% [markdown]
# ### Visualización del split temporal

# %%
events_clean["date"] = events_clean["datetime"].dt.date
daily = events_clean.groupby("date")["event"].count().reset_index()
daily["date"] = pd.to_datetime(daily["date"])

cutoff = train_events["datetime"].max()

fig, ax = plt.subplots(figsize=(12, 4))
train_mask = daily["date"] <= cutoff
ax.fill_between(daily["date"], daily["event"],
                where=train_mask, alpha=0.6,
                color=COLORS["popularity"], label="Train (80%)")
ax.fill_between(daily["date"], daily["event"],
                where=~train_mask, alpha=0.6,
                color=COLORS["item_cf"], label="Test — últimos 30 días")
ax.axvline(cutoff, color="red", linestyle="--", linewidth=1.5, label=f"Corte: {cutoff.date()}")
ax.set_xlabel("Fecha")
ax.set_ylabel("Eventos por día")
ax.set_title("Split temporal Train / Test", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.legend()
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint2_train_test_split.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Modelo Baseline — Popularidad Global

# %% [markdown]
# ### 3.1 Entrenamiento

# %%
from src.models.popularity import PopularityRecommender

popularity_model = PopularityRecommender(top_n=10)
popularity_model.fit(events_clean=events_clean, item_profiles=item_profiles)
popularity_model.save()

print("Modelo de popularidad entrenado y guardado.")

# %% [markdown]
# ### 3.2 Análisis del modelo de popularidad

# %%
top_items = popularity_model.global_popular.head(20)

print("Top 20 ítems más populares:")
print(top_items[["itemid", "popularity_score", "n_transactions", "category_id"]].to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Top 20 ítems por score
top20 = popularity_model.global_popular.head(20).copy()
top20["itemid_str"] = top20["itemid"].astype(str)

axes[0].barh(top20["itemid_str"][::-1], top20["popularity_score"][::-1],
             color=COLORS["popularity"], edgecolor="white", linewidth=0.5)
axes[0].set_xlabel("Score de popularidad (ponderado)")
axes[0].set_title("Top 20 ítems más populares")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))))

# Distribución de popularidad (long tail)
all_scores = popularity_model.global_popular["popularity_score"].values
axes[1].hist(np.log1p(all_scores), bins=60,
             color=COLORS["popularity"], edgecolor="white", linewidth=0.3, alpha=0.8)
axes[1].set_xlabel("log(1 + popularity_score)")
axes[1].set_ylabel("Número de ítems")
axes[1].set_title("Distribución de popularidad (escala log — long tail)")

plt.suptitle("Modelo Baseline — Popularidad Global", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint2_popularity_analysis.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.3 Ejemplo de recomendaciones del baseline

# %%
recs_global = popularity_model.recommend(n=10)
print("Top-10 recomendaciones globales (sin filtro de categoría):")
print(recs_global.to_string(index=False))

# Recomendación por categoría
sample_category = item_profiles["category_id"].value_counts().index[0]
recs_cat = popularity_model.recommend(category_id=sample_category, n=5)
print(f"\nTop-5 recomendaciones para categoría '{sample_category}':")
print(recs_cat.to_string(index=False))

# %% [markdown]
# ## 4. Modelo Item-CF — Collaborative Filtering Ítem-Ítem

# %% [markdown]
# ### 4.1 Entrenamiento

# %%
from src.models.item_cf import ItemCFRecommender
from config.config import EVENT_WEIGHTS, MIN_ITEM_INTERACTIONS

# Reconstruir matriz de entrenamiento (solo con datos de train)
train_matrix = (
    train_events[train_events["user_type"] == "warm"]
    .groupby(["visitorid", "itemid"])["event_weight"]
    .sum()
    .reset_index()
    .rename(columns={"event_weight": "score"})
)

# Filtrar ítems con pocas interacciones
item_counts = train_matrix.groupby("itemid")["score"].sum()
valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
train_matrix = train_matrix[train_matrix["itemid"].isin(valid_items)]

print(f"Matriz de entrenamiento Item-CF:")
print(f"  Usuarios: {train_matrix['visitorid'].nunique():,}")
print(f"  Ítems:    {train_matrix['itemid'].nunique():,}")
print(f"  Pares:    {len(train_matrix):,}")

item_cf_model = ItemCFRecommender(top_n=10, n_neighbors=50)
item_cf_model.fit(train_matrix)
item_cf_model.save()

print("\nModelo Item-CF entrenado y guardado.")

# %% [markdown]
# ### 4.2 Análisis de similitudes ítem-ítem

# %%
# Distribución del número de vecinos por ítem
n_neighbors_per_item = [
    len(v) for v in item_cf_model.item_similarity.values()
]

print(f"Estadísticas de vecinos por ítem:")
print(pd.Series(n_neighbors_per_item).describe())
print(f"\nÍtems sin vecinos: {sum(1 for n in n_neighbors_per_item if n == 0):,}")
print(f"Ítems con ≥10 vecinos: {sum(1 for n in n_neighbors_per_item if n >= 10):,}")
print(f"Ítems con 50 vecinos (máximo): {sum(1 for n in n_neighbors_per_item if n == 50):,}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histograma de vecinos por ítem
axes[0].hist(n_neighbors_per_item, bins=51,
             color=COLORS["item_cf"], edgecolor="white", linewidth=0.3, alpha=0.85)
axes[0].set_xlabel("Número de vecinos por ítem")
axes[0].set_ylabel("Número de ítems")
axes[0].set_title("Distribución de vecinos por ítem")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1000 else str(int(x))))

# Distribución de scores de similitud (muestra de 10K)
sample_sims = []
for neighbors in list(item_cf_model.item_similarity.values())[:500]:
    sample_sims.extend([s for _, s in neighbors])

axes[1].hist(sample_sims, bins=50,
             color=COLORS["item_cf"], edgecolor="white", linewidth=0.3, alpha=0.85)
axes[1].set_xlabel("Score de similitud coseno")
axes[1].set_ylabel("Frecuencia")
axes[1].set_title("Distribución de scores de similitud (muestra)")

plt.suptitle("Modelo Item-CF — Análisis de similitudes", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint2_itemcf_analysis.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4.3 Ejemplo de recomendaciones Item-CF

# %%
# Buscar un usuario warm con suficiente historial
warm_users = [
    uid for uid, items in item_cf_model.user_item_scores.items()
    if len(items) >= 5
]
sample_user = warm_users[0] if warm_users else None

if sample_user:
    user_history = item_cf_model.user_item_scores[sample_user]
    print(f"Usuario de ejemplo: {sample_user}")
    print(f"Historial ({len(user_history)} ítems):")
    for iid, score in sorted(user_history.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  ítem {iid}: score {score}")

    recs_cf = item_cf_model.recommend(sample_user, n=10)
    print(f"\nTop-10 recomendaciones Item-CF:")
    print(recs_cf.to_string(index=False))

    # Ítems similares a uno del historial
    seed_item = list(user_history.keys())[0]
    similar = item_cf_model.recommend_similar(seed_item, n=5)
    print(f"\nÍtems similares al ítem {seed_item}:")
    print(similar.to_string(index=False))

# %% [markdown]
# ## 5. Evaluación Comparativa

# %% [markdown]
# ### 5.1 Cálculo de métricas

# %%
from src.evaluation.metrics import evaluate_model, compare_models

catalog_size = item_profiles["itemid"].nunique()
popularity_scores = events_clean.groupby("itemid")["event_weight"].sum()

print(f"Catálogo total: {catalog_size:,} ítems")
print(f"Evaluando sobre {test_purchases['visitorid'].nunique():,} usuarios...")

# Función de recomendación para cada modelo
def rec_popularity(uid, k):
    recs = popularity_model.recommend(n=k)
    return recs["itemid"].tolist()

def rec_item_cf(uid, k):
    recs = item_cf_model.recommend(uid, n=k)
    if recs.empty:
        recs = popularity_model.recommend(n=k)
    return recs["itemid"].tolist() if not recs.empty else []

# Evaluar
print("\nEvaluando Popularity...")
metrics_pop = evaluate_model(
    recommender_fn=rec_popularity,
    test_interactions=test_purchases,
    catalog_size=catalog_size,
    popularity_scores=popularity_scores,
)

print("\nEvaluando Item-CF...")
metrics_cf = evaluate_model(
    recommender_fn=rec_item_cf,
    test_interactions=test_purchases,
    catalog_size=catalog_size,
    popularity_scores=popularity_scores,
)

# %% [markdown]
# ### 5.2 Tabla comparativa

# %%
comparison = compare_models(
    {"popularity": metrics_pop, "item_cf": metrics_cf},
    k=10
)
print("\nComparativa @ K=10:")
print(comparison.to_string(index=False))

# Guardar métricas
metrics_pop["model"] = "popularity"
metrics_cf["model"]  = "item_cf"
all_metrics = pd.concat([metrics_pop, metrics_cf], ignore_index=True)
all_metrics.to_parquet(DATA_PRO / "metrics_sprint2.parquet", index=False)
print(f"\nMétricas guardadas en: {DATA_PRO / 'metrics_sprint2.parquet'}")

# %% [markdown]
# ### 5.3 Visualizaciones comparativas

# %%
# Gráfica 1: Métricas de precisión (barras agrupadas)
metrics_k10 = all_metrics[all_metrics["K"] == 10].set_index("model")
metric_cols  = ["Precision@K", "Recall@K", "NDCG@K", "HitRate@K"]
model_colors = [COLORS["popularity"], COLORS["item_cf"]]

fig, ax = plt.subplots(figsize=(11, 5))
x      = np.arange(len(metric_cols))
width  = 0.35

for i, (model, color) in enumerate(zip(["popularity", "item_cf"], model_colors)):
    vals = [metrics_k10.loc[model, m] for m in metric_cols]
    bars = ax.bar(x + i * width, vals, width,
                  label=model, color=color,
                  edgecolor="white", linewidth=0.8, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, rotation=45)

ax.set_xticks(x + width / 2)
ax.set_xticklabels(metric_cols)
ax.set_ylabel("Score")
ax.set_title("Métricas de precisión @ K=10 — Popularity vs Item-CF",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint2_metrics_precision.png", bbox_inches="tight")
plt.show()

# %%
# Gráfica 2: Coverage y Novelty
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

models   = ["popularity", "item_cf"]
coverage = [metrics_k10.loc[m, "Coverage"] * 100 for m in models]
novelty  = [metrics_k10.loc[m, "Novelty"] for m in models]

bars1 = axes[0].bar(models, coverage, color=model_colors,
                    edgecolor="white", linewidth=0.8, alpha=0.9)
axes[0].set_ylabel("% del catálogo recomendado")
axes[0].set_title("Coverage @ K=10")
for bar, val in zip(bars1, coverage):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.3f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

bars2 = axes[1].bar(models, novelty, color=model_colors,
                    edgecolor="white", linewidth=0.8, alpha=0.9)
axes[1].set_ylabel("Novelty score")
axes[1].set_title("Novelty @ K=10\n(mayor = menos sesgo de popularidad)")
for bar, val in zip(bars2, novelty):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.suptitle("Coverage y Novelty — KPI Secundario del Proyecto",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint2_coverage_novelty.png", bbox_inches="tight")
plt.show()

# %%
# Gráfica 3: Métricas por K (curvas)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
k_vals = [5, 10, 20]

for ax, metric in zip(axes, ["NDCG@K", "Recall@K", "HitRate@K"]):
    for model, color in zip(["popularity", "item_cf"], model_colors):
        model_data = all_metrics[all_metrics["model"] == model]
        vals = [model_data[model_data["K"] == k][metric].values[0] for k in k_vals]
        ax.plot(k_vals, vals, marker="o", color=color,
                linewidth=2, markersize=7, label=model)
        for k, v in zip(k_vals, vals):
            ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("K")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} por K")
    ax.set_xticks(k_vals)
    ax.legend(fontsize=9)

plt.suptitle("Evolución de métricas por K — Popularity vs Item-CF",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(DATA_PRO / "sprint2_metrics_by_k.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Análisis de Resultados

# %%
print("=" * 65)
print("ANÁLISIS DE RESULTADOS — SPRINT 2")
print("=" * 65)

pop_ndcg  = metrics_k10.loc["popularity", "NDCG@K"]
cf_ndcg   = metrics_k10.loc["item_cf", "NDCG@K"]
pop_cov   = metrics_k10.loc["popularity", "Coverage"]
cf_cov    = metrics_k10.loc["item_cf", "Coverage"]
pop_nov   = metrics_k10.loc["popularity", "Novelty"]
cf_nov    = metrics_k10.loc["item_cf", "Novelty"]

print(f"""
Comparativa @ K=10:
  NDCG@10:
    Popularity: {pop_ndcg:.4f}  ← gana en precisión de ranking
    Item-CF:    {cf_ndcg:.4f}

  Coverage:
    Popularity: {pop_cov*100:.4f}%   ← recomienda siempre los mismos ítems
    Item-CF:    {cf_cov*100:.2f}%    ← {cf_cov/pop_cov:.0f}x más cobertura de catálogo

  Novelty:
    Popularity: {pop_nov:.2f}
    Item-CF:    {cf_nov:.2f}   ← {((cf_nov-pop_nov)/pop_nov*100):.0f}% menos sesgo de popularidad
""")

hallazgos = [
    ("ESPERADO",  "Popularity supera a Item-CF en precisión. Recomienda ítems"
                  " que cualquier usuario tiene alta probabilidad de ver."),
    ("CRÍTICO",   f"Coverage de Popularity es {pop_cov*100:.4f}% — prácticamente"
                  " cero. El sistema actual NO activa el catálogo invisible."),
    ("POSITIVO",  f"Item-CF tiene {cf_cov/pop_cov:.0f}x más Coverage que Popularity."
                  " Directamente ataca el KPI secundario del proyecto."),
    ("CAUSA",     "NDCG bajo en Item-CF se debe a sparsity extrema (99.99%)."
                  " ALS (Sprint 3) maneja mejor matrices sparse con feedback implícito."),
    ("SPRINT 3",  "Objetivo: ALS supere a Popularity en NDCG Y mantenga Coverage"
                  " de Item-CF. El híbrido combinará ambas fortalezas."),
]

for tag, texto in hallazgos:
    emoji = {"ESPERADO": "🔵", "CRÍTICO": "🔴", "POSITIVO": "🟢",
             "CAUSA": "🟡", "SPRINT 3": "⚪"}[tag]
    print(f"{emoji} [{tag}] {texto}\n")

print("=" * 65)
print("Artefactos generados en data/processed/:")
for f in ["sprint2_train_test_split.png", "sprint2_popularity_analysis.png",
          "sprint2_itemcf_analysis.png", "sprint2_metrics_precision.png",
          "sprint2_coverage_novelty.png", "sprint2_metrics_by_k.png",
          "metrics_sprint2.parquet"]:
    print(f"  {f}")
print("=" * 65)
print("✅ Sprint 2 completado.")
