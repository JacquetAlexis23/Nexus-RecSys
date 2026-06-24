"""
src/data/cleaner.py
Pipeline de limpieza del dataset RetailRocket.
Trata todos los problemas de calidad identificados:
  - Duplicados
  - transactionid inválidos
  - Sesiones incompletas
  - Usuarios y ítems con muy pocas interacciones (cold-start boundary)
  - Valores hasheados en item_properties (tratamiento como categorías)
"""

import pandas as pd
import numpy as np
from loguru import logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    EVENT_WEIGHTS,
    MIN_USER_INTERACTIONS,
    MIN_ITEM_INTERACTIONS,
    SESSION_GAP_MINUTES,
    EVENTS_CLEAN,
    PROC_DIR,
)

PROC_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LIMPIEZA DE EVENTOS
# ─────────────────────────────────────────────────────────────────────────────

def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza completa del DataFrame de eventos.
    
    Pasos:
      1. Eliminar duplicados exactos
      2. Validar transactionid en eventos de compra
      3. Asignar peso por tipo de evento
      4. Segmentar usuarios en 'warm' (historial suficiente) y 'cold'
      5. Agregar identificador de sesión

    Parameters
    ----------
    df : pd.DataFrame — salida de loader.load_events()

    Returns
    -------
    pd.DataFrame limpio con columnas adicionales:
      event_weight, session_id, user_type
    """
    original_len = len(df)
    logger.info(f"Iniciando limpieza de eventos. Filas iniciales: {original_len:,}")

    # ── Paso 1: Duplicados exactos ─────────────────────────────────────────
    df = df.drop_duplicates(
        subset=["timestamp", "visitorid", "event", "itemid"]
    )
    logger.info(f"  Duplicados eliminados: {original_len - len(df):,}")

    # ── Paso 2: Validar transacciones ─────────────────────────────────────
    # transactionid debe existir y no ser vacío para eventos 'transaction'
    mask_transaction = df["event"] == "transaction"
    invalid_transactions = mask_transaction & (
        df["transactionid"].isna() | (df["transactionid"].str.strip() == "")
    )
    n_invalid = invalid_transactions.sum()
    if n_invalid > 0:
        logger.warning(
            f"  Transacciones con transactionid inválido: {n_invalid:,} → eliminadas"
        )
        df = df[~invalid_transactions]

    # ── Paso 3: Peso por evento ────────────────────────────────────────────
    df["event_weight"] = df["event"].map(EVENT_WEIGHTS).fillna(1).astype("int8")

    # ── Paso 4: Identificación de sesiones ────────────────────────────────
    # Ordenar por usuario y tiempo para calcular gaps
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    
    gap_ms = SESSION_GAP_MINUTES * 60 * 1000  # convertir a milisegundos
    time_diff = df.groupby("visitorid")["timestamp"].diff()
    new_session = (time_diff > gap_ms) | (time_diff.isna())
    df["session_id"] = new_session.cumsum().astype("int32")
    
    total_sessions = df["session_id"].nunique()
    logger.info(f"  Sesiones identificadas: {total_sessions:,}")

    # ── Paso 5: Clasificación warm / cold ─────────────────────────────────
    interaction_counts = df.groupby("visitorid")["event"].count()
    warm_users = interaction_counts[
        interaction_counts >= MIN_USER_INTERACTIONS
    ].index
    
    df["user_type"] = np.where(
        df["visitorid"].isin(warm_users), "warm", "cold"
    )
    
    n_warm = df[df["user_type"] == "warm"]["visitorid"].nunique()
    n_cold = df[df["user_type"] == "cold"]["visitorid"].nunique()
    pct_warm = n_warm / (n_warm + n_cold) * 100
    
    logger.info(
        f"  Usuarios warm (≥{MIN_USER_INTERACTIONS} interacciones): "
        f"{n_warm:,} ({pct_warm:.1f}%)"
    )
    logger.info(f"  Usuarios cold-start: {n_cold:,} ({100-pct_warm:.1f}%)")

    # ── Reporte final ──────────────────────────────────────────────────────
    logger.info(
        f"Limpieza completada. Filas finales: {len(df):,} "
        f"(eliminadas: {original_len - len(df):,})"
    )
    _log_funnel(df)

    return df


def _log_funnel(df: pd.DataFrame) -> None:
    """Imprime el funnel de conversión post-limpieza."""
    counts = df["event"].value_counts()
    total = len(df)
    logger.info("  Funnel de conversión (post-limpieza):")
    for event, count in counts.items():
        logger.info(f"    {event:<15}: {count:>10,}  ({count/total*100:.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LIMPIEZA DE ITEM PROPERTIES
# ─────────────────────────────────────────────────────────────────────────────

def clean_item_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento de item_properties con valores hasheados.
    
    Estrategia para valores hasheados:
      - Los hashes son estables y consistentes → útiles como features categóricas
      - NO intentamos decodificar: los tratamos como IDs opacos
      - Construimos un perfil de ítem como bolsa de (property, value) pairs
      - El resultado es compatible con TF-IDF y embeddings

    Returns
    -------
    pd.DataFrame con columna adicional 'prop_value_token' para NLP
    """
    logger.info(f"Limpiando propiedades de ítems. Filas: {len(df):,}")

    # Eliminar duplicados (mismo ítem, misma propiedad, mismo valor, mismo tiempo)
    df = df.drop_duplicates(subset=["itemid", "property", "value"])
    logger.info(f"  Post-dedup: {len(df):,} filas")

    # Normalizar valores: strip, lowercase
    df["property"] = df["property"].str.strip().str.lower()
    df["value"]    = df["value"].fillna("unknown").str.strip()

    # Crear token combinado property_value para uso en TF-IDF
    # Ejemplo: "categoryid_123456" o "available_n"
    df["prop_value_token"] = df["property"] + "_" + df["value"]

    # Identificar propiedades con valores legibles vs hasheados
    # Heurística: si el valor tiene >20 chars y es alfanumérico → hash
    df["is_hashed"] = df["value"].str.len() > 20

    n_hashed = df["is_hashed"].sum()
    pct_hashed = n_hashed / len(df) * 100
    logger.info(
        f"  Valores hasheados detectados: {n_hashed:,} ({pct_hashed:.1f}%) "
        f"→ se usarán como features categóricas opacas"
    )

    logger.info(
        f"  Propiedades únicas: {df['property'].nunique():,} | "
        f"Ítems con propiedades: {df['itemid'].nunique():,}"
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUCCIÓN DE LA MATRIZ DE INTERACCIONES
# ─────────────────────────────────────────────────────────────────────────────

def build_interaction_matrix(events_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la matriz usuario-ítem ponderada agregando los pesos
    de todos los eventos por par (visitorid, itemid).

    Solo incluye usuarios 'warm' para el entrenamiento de modelos CF.
    Los usuarios cold-start tienen su propio pipeline de recomendación.

    Returns
    -------
    pd.DataFrame con columnas: visitorid, itemid, score
      donde score = suma ponderada de interacciones
    """
    logger.info("Construyendo matriz de interacciones ponderada...")

    # Filtrar solo usuarios warm para modelos colaborativos
    warm_events = events_clean[events_clean["user_type"] == "warm"].copy()

    # Agregar pesos por par (usuario, ítem)
    matrix = (
        warm_events
        .groupby(["visitorid", "itemid"])["event_weight"]
        .sum()
        .reset_index()
        .rename(columns={"event_weight": "score"})
    )

    # Filtrar ítems con pocas interacciones globales
    item_counts = matrix.groupby("itemid")["score"].sum()
    valid_items = item_counts[item_counts >= MIN_ITEM_INTERACTIONS].index
    matrix = matrix[matrix["itemid"].isin(valid_items)]

    sparsity = 1 - len(matrix) / (
        matrix["visitorid"].nunique() * matrix["itemid"].nunique()
    )

    logger.info(
        f"  Matriz construida: "
        f"{matrix['visitorid'].nunique():,} usuarios × "
        f"{matrix['itemid'].nunique():,} ítems"
    )
    logger.info(f"  Sparsity de la matriz: {sparsity:.4%}")
    logger.info(f"  Score promedio por interacción: {matrix['score'].mean():.2f}")

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# 4. PIPELINE COMPLETO
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning_pipeline(
    events_raw: pd.DataFrame,
    item_props_raw: pd.DataFrame,
    save: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de limpieza y retorna los artefactos procesados.

    Parameters
    ----------
    events_raw     : salida de loader.load_events()
    item_props_raw : salida de loader.load_item_properties()
    save           : si True, guarda los artefactos en data/processed/

    Returns
    -------
    dict con claves:
      'events_clean', 'item_props_clean', 'interaction_matrix'
    """
    logger.info("━" * 60)
    logger.info("PIPELINE DE LIMPIEZA — RetailRocket")
    logger.info("━" * 60)

    events_clean    = clean_events(events_raw)
    item_props_clean = clean_item_properties(item_props_raw)
    interaction_mtx  = build_interaction_matrix(events_clean)

    if save:
        PROC_DIR.mkdir(parents=True, exist_ok=True)
        events_clean.to_parquet(EVENTS_CLEAN, index=False)
        item_props_clean.to_parquet(PROC_DIR / "item_props_clean.parquet", index=False)
        interaction_mtx.to_parquet(PROC_DIR / "interaction_matrix.parquet", index=False)
        logger.info(f"Artefactos guardados en: {PROC_DIR}")

    logger.info("━" * 60)
    logger.info("Pipeline de limpieza completado.")
    logger.info("━" * 60)

    return {
        "events_clean":       events_clean,
        "item_props_clean":   item_props_clean,
        "interaction_matrix": interaction_mtx,
    }


if __name__ == "__main__":
    from src.data.loader import load_events, load_item_properties
    events_raw    = load_events()
    item_props_raw = load_item_properties()
    artifacts     = run_cleaning_pipeline(events_raw, item_props_raw, save=True)
