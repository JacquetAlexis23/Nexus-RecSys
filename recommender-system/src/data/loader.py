"""
src/data/loader.py
Carga de los archivos fuente del dataset RetailRocket.
Incluye validación de esquema y logging detallado.
"""

import pandas as pd
from loguru import logger
from pathlib import Path
import sys

# Agregar raíz del proyecto al path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    EVENTS_FILE, ITEM_PROPS_1, ITEM_PROPS_2,
    CATEGORY_TREE, PROC_DIR
)

# ── Esquemas esperados ────────────────────────────────────────────────────────
EVENTS_DTYPES = {
    "timestamp":     "int64",
    "visitorid":     "int64",
    "event":         "category",
    "itemid":        "int64",
    "transactionid": "object",   # puede ser NaN
}

ITEM_PROPS_DTYPES = {
    "timestamp": "int64",
    "itemid":    "int64",
    "property":  "object",
    "value":     "object",
}

CATEGORY_DTYPES = {
    "categoryid":       "int64",
    "parentid":         "object",   # raíz tiene NaN
}


def load_events(filepath: Path = EVENTS_FILE) -> pd.DataFrame:
    """
    Carga events.csv con tipos optimizados.
    
    Returns
    -------
    pd.DataFrame con columnas: timestamp, visitorid, event, itemid, transactionid
    """
    logger.info(f"Cargando eventos desde: {filepath}")
    
    df = pd.read_csv(
        filepath,
        dtype={
            "visitorid": "int64",
            "event": "category",
            "itemid": "int64",
            "transactionid": "object",
        },
    )
    
    # Convertir timestamp de milisegundos a datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    logger.info(
        f"Eventos cargados: {len(df):,} filas | "
        f"Visitantes únicos: {df['visitorid'].nunique():,} | "
        f"Ítems únicos: {df['itemid'].nunique():,}"
    )
    logger.info(f"Distribución de eventos:\n{df['event'].value_counts(normalize=True).mul(100).round(2)}")
    
    return df


def load_item_properties() -> pd.DataFrame:
    """
    Carga y concatena item_properties_part1 y part2.
    
    Returns
    -------
    pd.DataFrame con columnas: timestamp, itemid, property, value
    """
    logger.info("Cargando propiedades de ítems (parte 1 y 2)...")
    
    dfs = []
    for filepath in [ITEM_PROPS_1, ITEM_PROPS_2]:
        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            continue
        chunk = pd.read_csv(filepath, dtype=ITEM_PROPS_DTYPES)
        dfs.append(chunk)
        logger.info(f"  {filepath.name}: {len(chunk):,} filas")
    
    if not dfs:
        raise FileNotFoundError(
            "No se encontraron archivos de item_properties. "
            "Coloca item_properties_part1.csv y item_properties_part2.csv en data/raw/"
        )
    
    df = pd.concat(dfs, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    logger.info(
        f"Propiedades cargadas: {len(df):,} filas | "
        f"Ítems únicos: {df['itemid'].nunique():,} | "
        f"Propiedades únicas: {df['property'].nunique():,}"
    )
    return df


def load_category_tree(filepath: Path = CATEGORY_TREE) -> pd.DataFrame:
    """
    Carga la jerarquía de categorías.
    
    Returns
    -------
    pd.DataFrame con columnas: categoryid, parentid
    """
    logger.info(f"Cargando árbol de categorías desde: {filepath}")
    
    df = pd.read_csv(filepath, dtype=CATEGORY_DTYPES)
    df["parentid"] = pd.to_numeric(df["parentid"], errors="coerce")
    
    n_roots = df["parentid"].isna().sum()
    logger.info(
        f"Categorías: {len(df):,} | "
        f"Categorías raíz (sin padre): {n_roots}"
    )
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """
    Carga todos los datasets y retorna un diccionario.
    Útil para notebooks y scripts de análisis.
    
    Returns
    -------
    dict con claves: 'events', 'item_properties', 'category_tree'
    """
    logger.info("=" * 60)
    logger.info("CARGA COMPLETA DEL DATASET RetailRocket")
    logger.info("=" * 60)
    
    data = {}
    
    # Verificar existencia de archivos antes de cargar
    missing = []
    for name, path in [
        ("events.csv", EVENTS_FILE),
        ("item_properties_part1.csv", ITEM_PROPS_1),
        ("item_properties_part2.csv", ITEM_PROPS_2),
        ("category_tree.csv", CATEGORY_TREE),
    ]:
        if not path.exists():
            missing.append(name)
    
    if missing:
        logger.error(
            f"Archivos faltantes en data/raw/:\n  " + "\n  ".join(missing)
        )
        logger.info(
            "Descarga el dataset desde: "
            "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset"
        )
        raise FileNotFoundError(f"Faltan {len(missing)} archivo(s) del dataset.")
    
    data["events"]          = load_events()
    data["item_properties"] = load_item_properties()
    data["category_tree"]   = load_category_tree()
    
    logger.info("=" * 60)
    logger.info("Carga completada exitosamente.")
    logger.info("=" * 60)
    
    return data


if __name__ == "__main__":
    # Test rápido de carga
    data = load_all()
    for name, df in data.items():
        print(f"\n{name}: {df.shape}")
        print(df.head(3))
