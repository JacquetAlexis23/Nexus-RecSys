"""
src/data/feature_engineering.py
Construcción de features para los modelos de recomendación:
  1. Perfil de ítem: vector TF-IDF sobre tokens de item_properties
  2. Perfil de usuario: vector de preferencias por categoría
  3. Features temporales: hora del día, día de semana, recencia
  4. Integración con datos sintéticos demográficos
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from loguru import logger
import scipy.sparse as sp
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    PROC_DIR,
    CBF_MAX_FEATURES,
    RANDOM_SEED,
)

PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PERFIL DE ÍTEM (Content-Based)
# ─────────────────────────────────────────────────────────────────────────────

def build_item_profiles(item_props_clean: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Construye perfiles de ítems usando TF-IDF sobre los tokens
    property_value de item_properties (valores hasheados incluidos).

    Estrategia:
      - Cada ítem = documento = concatenación de sus tokens "prop_value"
      - TF-IDF captura qué tokens son distintivos de cada ítem
      - Los hashes funcionan como features categóricas: si dos ítems
        comparten el hash de "categoryid", son similares en categoría

    Returns
    -------
    item_profiles : pd.DataFrame (itemid, category_id)
    tfidf_matrix  : scipy.sparse matrix (n_items × n_features)
    vectorizer    : TfidfVectorizer entrenado (para nuevos ítems)
    """
    logger.info("Construyendo perfiles de ítems con TF-IDF...")

    # Agrupar tokens por ítem: cada ítem es un "documento"
    item_docs = (
        item_props_clean
        .groupby("itemid")["prop_value_token"]
        .apply(lambda tokens: " ".join(tokens))
        .reset_index()
        .rename(columns={"prop_value_token": "document"})
    )

    # Extraer categoryid si existe como propiedad legible
    category_props = item_props_clean[
        item_props_clean["property"] == "categoryid"
    ][["itemid", "value"]].rename(columns={"value": "category_id"})

    # Un ítem puede tener múltiples categoryid en el tiempo → tomar el más reciente
    if "timestamp" in item_props_clean.columns:
        category_props = (
            item_props_clean[item_props_clean["property"] == "categoryid"]
            .sort_values("timestamp")
            .groupby("itemid")["value"]
            .last()
            .reset_index()
            .rename(columns={"value": "category_id"})
        )

    item_docs = item_docs.merge(category_props, on="itemid", how="left")
    item_docs["category_id"] = item_docs["category_id"].fillna("unknown")

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=CBF_MAX_FEATURES,
        min_df=2,         # token debe aparecer en al menos 2 ítems
        sublinear_tf=True,
        analyzer="word",
    )
    tfidf_matrix = vectorizer.fit_transform(item_docs["document"])
    tfidf_matrix = normalize(tfidf_matrix, norm="l2")  # cosine similarity ready

    logger.info(
        f"  Perfiles construidos: {tfidf_matrix.shape[0]:,} ítems × "
        f"{tfidf_matrix.shape[1]:,} features"
    )
    logger.info(f"  Categorías únicas: {item_docs['category_id'].nunique():,}")

    # Guardar artefactos
    item_docs.to_parquet(PROC_DIR / "item_profiles.parquet", index=False)
    sp.save_npz(PROC_DIR / "tfidf_matrix.npz", tfidf_matrix)
    joblib.dump(vectorizer, PROC_DIR / "tfidf_vectorizer.pkl")

    logger.info(f"  Artefactos guardados en {PROC_DIR}")
    return item_docs, tfidf_matrix, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES TEMPORALES
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(events_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features temporales al DataFrame de eventos:
      - hour_of_day (0-23)
      - day_of_week (0=lunes, 6=domingo)
      - is_weekend
      - days_since_first_event (recencia relativa al inicio del dataset)
      - recency_score: peso decreciente por antigüedad (para ALS con recencia)
    """
    logger.info("Agregando features temporales...")

    df = events_clean.copy()

    df["hour_of_day"]  = df["datetime"].dt.hour.astype("int8")
    df["day_of_week"]  = df["datetime"].dt.dayofweek.astype("int8")
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype("int8")

    # Recencia: días desde el primer evento del dataset
    t_min = df["datetime"].min()
    df["days_since_start"] = (df["datetime"] - t_min).dt.days.astype("int16")

    # Score de recencia: eventos más recientes tienen mayor peso
    # Fórmula: exp(-lambda * days_ago), lambda=0.01 → decaimiento suave
    t_max = df["datetime"].max()
    days_ago = (t_max - df["datetime"]).dt.days
    df["recency_score"] = np.exp(-0.01 * days_ago).round(4)

    logger.info(
        f"  Período de datos: {t_min.date()} → {t_max.date()} "
        f"({(t_max - t_min).days} días)"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERFIL DE USUARIO BASADO EN CATEGORÍAS
# ─────────────────────────────────────────────────────────────────────────────

def build_user_category_profiles(
    events_clean: pd.DataFrame,
    item_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye un perfil de usuario como distribución de preferencias
    por categoría de ítem, ponderado por event_weight.

    Útil para:
      - Cold-start: recomendar ítems de categorías que el usuario ha visto
      - Ensamble híbrido: señal de contenido para refinar CF

    Returns
    -------
    pd.DataFrame: (visitorid, category_id, preference_score)
    """
    logger.info("Construyendo perfiles de usuario por categoría...")

    # Unir eventos con categoría del ítem
    events_with_cat = events_clean.merge(
        item_profiles[["itemid", "category_id"]],
        on="itemid",
        how="left",
    )

    user_cat = (
        events_with_cat
        .groupby(["visitorid", "category_id"])["event_weight"]
        .sum()
        .reset_index()
        .rename(columns={"event_weight": "preference_score"})
    )

    # Normalizar por usuario (distribución de probabilidad)
    user_totals = user_cat.groupby("visitorid")["preference_score"].transform("sum")
    user_cat["preference_pct"] = (user_cat["preference_score"] / user_totals).round(4)

    logger.info(
        f"  Perfiles de usuario × categoría: {len(user_cat):,} filas | "
        f"Usuarios: {user_cat['visitorid'].nunique():,}"
    )

    user_cat.to_parquet(PROC_DIR / "user_category_profiles.parquet", index=False)
    return user_cat


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_engineering(
    events_clean: pd.DataFrame,
    item_props_clean: pd.DataFrame,
) -> dict:
    """
    Pipeline completo de feature engineering.

    Returns
    -------
    dict con:
      'events_featured'     : eventos con features temporales
      'item_profiles'       : perfiles de ítems (DataFrame)
      'tfidf_matrix'        : matriz TF-IDF (scipy sparse)
      'user_cat_profiles'   : perfiles usuario × categoría
    """
    logger.info("━" * 60)
    logger.info("FEATURE ENGINEERING — RetailRocket")
    logger.info("━" * 60)

    events_featured = add_temporal_features(events_clean)

    item_profiles, tfidf_matrix, vectorizer = build_item_profiles(item_props_clean)

    user_cat_profiles = build_user_category_profiles(events_featured, item_profiles)

    logger.info("━" * 60)
    logger.info("Feature engineering completado.")
    logger.info("━" * 60)

    return {
        "events_featured":   events_featured,
        "item_profiles":     item_profiles,
        "tfidf_matrix":      tfidf_matrix,
        "user_cat_profiles": user_cat_profiles,
    }


if __name__ == "__main__":
    events_clean     = pd.read_parquet(PROC_DIR / "events_clean.parquet")
    item_props_clean = pd.read_parquet(PROC_DIR / "item_props_clean.parquet")
    artifacts = run_feature_engineering(events_clean, item_props_clean)
