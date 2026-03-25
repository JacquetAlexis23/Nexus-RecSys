"""
src/evaluation/metrics.py
Métricas estándar para evaluar sistemas de recomendación:
  - Precision@K
  - Recall@K
  - NDCG@K
  - Coverage (KPI secundario del proyecto)
  - Novelty (anti-popularidad)
  - Hit Rate@K

Todas las funciones aceptan listas de recomendaciones y ground truth.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Callable
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import EVAL_K


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS POR USUARIO
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fracción de ítems recomendados en top-K que son relevantes.
    
    Parameters
    ----------
    recommended : lista ordenada de itemid recomendados
    relevant    : conjunto de itemid relevantes (ground truth)
    k           : cutoff
    
    Returns
    -------
    float en [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Fracción de ítems relevantes capturados en top-K.
    
    Returns
    -------
    float en [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @K.
    Penaliza recomendaciones relevantes en posiciones bajas.
    
    Returns
    -------
    float en [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    
    top_k = recommended[:k]
    
    # DCG: relevancia binaria, descuento logarítmico por posición
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    
    # IDCG: caso ideal (todos los relevantes en posiciones top)
    n_relevant_in_k = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_k))
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    """
    1 si al menos 1 ítem relevante aparece en top-K, 0 si no.
    """
    if not recommended or not relevant:
        return 0.0
    return float(any(item in relevant for item in recommended[:k]))


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS AGREGADAS (sobre todos los usuarios de test)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    recommender_fn: Callable[[int, int], list],
    test_interactions: pd.DataFrame,
    catalog_size: int,
    k_values: list[int] = EVAL_K,
    popularity_scores: pd.Series | None = None,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """
    Evalúa un modelo de recomendación sobre el conjunto de test.

    Parameters
    ----------
    recommender_fn    : función (visitor_id, k) → lista de itemid recomendados
    test_interactions : DataFrame con columnas [visitorid, itemid] (ground truth)
    catalog_size      : número total de ítems en el catálogo
    k_values          : lista de K para evaluar
    popularity_scores : Serie (itemid → score) para calcular novelty
    sample_size       : si se especifica, evalúa sobre N usuarios aleatorios

    Returns
    -------
    pd.DataFrame con métricas por K
    """
    logger.info(f"Evaluando modelo con K={k_values}...")

    # Ground truth: usuario → set de ítems relevantes
    ground_truth = (
        test_interactions
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    users = list(ground_truth.keys())
    if sample_size and sample_size < len(users):
        rng = np.random.default_rng(42)
        users = rng.choice(users, size=sample_size, replace=False).tolist()
        logger.info(f"  Evaluando sobre muestra de {sample_size:,} usuarios")
    else:
        logger.info(f"  Evaluando sobre {len(users):,} usuarios")

    results = {k: {"precision": [], "recall": [], "ndcg": [], "hit_rate": []}
               for k in k_values}
    
    recommended_items_all = set()   # para coverage
    novelty_scores_all = []

    for visitor_id in users:
        relevant = ground_truth[visitor_id]
        max_k = max(k_values)
        
        try:
            recs = recommender_fn(visitor_id, max_k)
        except Exception as e:
            logger.debug(f"Error generando recomendaciones para {visitor_id}: {e}")
            continue

        recommended_items_all.update(recs)

        # Novelty: inverso del log de popularidad
        if popularity_scores is not None:
            item_popularities = [
                popularity_scores.get(item, 1) for item in recs[:max_k]
            ]
            novelty = np.mean([-np.log2(p / popularity_scores.sum() + 1e-10)
                               for p in item_popularities])
            novelty_scores_all.append(novelty)

        for k in k_values:
            results[k]["precision"].append(precision_at_k(recs, relevant, k))
            results[k]["recall"].append(recall_at_k(recs, relevant, k))
            results[k]["ndcg"].append(ndcg_at_k(recs, relevant, k))
            results[k]["hit_rate"].append(hit_rate_at_k(recs, relevant, k))

    # Agregar resultados
    rows = []
    for k in k_values:
        row = {
            "K":           k,
            "Precision@K": np.mean(results[k]["precision"]),
            "Recall@K":    np.mean(results[k]["recall"]),
            "NDCG@K":      np.mean(results[k]["ndcg"]),
            "HitRate@K":   np.mean(results[k]["hit_rate"]),
            "Coverage":    len(recommended_items_all) / catalog_size,
            "Novelty":     np.mean(novelty_scores_all) if novelty_scores_all else None,
            "N_users":     len(users),
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    
    logger.info("Resultados de evaluación:")
    logger.info(f"\n{metrics_df.to_string(index=False)}")

    return metrics_df


def compare_models(
    models: dict[str, pd.DataFrame],
    k: int = 10,
) -> pd.DataFrame:
    """
    Compara múltiples modelos en una tabla de resumen para K dado.

    Parameters
    ----------
    models : dict nombre_modelo → DataFrame de métricas (salida de evaluate_model)
    k      : valor de K para comparar

    Returns
    -------
    pd.DataFrame comparativo ordenado por NDCG@K
    """
    rows = []
    for model_name, metrics_df in models.items():
        row_k = metrics_df[metrics_df["K"] == k].iloc[0].to_dict()
        row_k["Model"] = model_name
        rows.append(row_k)

    comparison = pd.DataFrame(rows).sort_values("NDCG@K", ascending=False)
    comparison = comparison[
        ["Model", "Precision@K", "Recall@K", "NDCG@K", "HitRate@K", "Coverage", "Novelty"]
    ]
    
    logger.info(f"\nComparativa de modelos @ K={k}:")
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    return comparison


def temporal_train_test_split(
    events_clean: pd.DataFrame,
    test_days: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporal: últimos N días como test.
    Evita data leakage (no split aleatorio).

    Returns
    -------
    (train, test) DataFrames
    """
    cutoff = events_clean["datetime"].max() - pd.Timedelta(days=test_days)
    train = events_clean[events_clean["datetime"] <= cutoff]
    test  = events_clean[events_clean["datetime"] > cutoff]

    logger.info(
        f"Split temporal (últimos {test_days} días como test):\n"
        f"  Train: {len(train):,} eventos ({train['datetime'].max().date()})\n"
        f"  Test:  {len(test):,} eventos ({test['datetime'].max().date()})"
    )

    # Solo evaluar usuarios que aparecen en ambos conjuntos
    train_users = set(train["visitorid"].unique())
    test_warm   = test[test["visitorid"].isin(train_users)]
    logger.info(
        f"  Usuarios en test con historial en train: "
        f"{test_warm['visitorid'].nunique():,}"
    )

    return train, test_warm
