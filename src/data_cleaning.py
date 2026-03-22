"""
data_cleaning.py
================
Módulo de preprocesamiento y limpieza de datos para el sistema de recomendación NexusDataCo.
"""

import pandas as pd
import numpy as np
import os
import sys


# ──────────────────────────────────────────────────────────
# FUNCIONES DE DIAGNÓSTICO
# ──────────────────────────────────────────────────────────

def summarize_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Imprime un resumen completo del DataFrame."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  RESUMEN: {name}")
    print(f"{sep}")
    print(f"  Filas    : {df.shape[0]:>12,}")
    print(f"  Columnas : {df.shape[1]:>12}")
    print(f"\n  TIPOS DE DATOS Y VALORES NULOS:")
    info_df = pd.DataFrame({
        "Tipo"    : df.dtypes,
        "Nulos"   : df.isnull().sum(),
        "% Nulos" : (df.isnull().sum() / max(len(df), 1) * 100).round(2),
    })
    print(info_df.to_string())
    print(f"\n  Filas duplicadas: {df.duplicated().sum():,} "
          f"({df.duplicated().sum() / max(len(df), 1) * 100:.2f}%)")
    print(f"{sep}\n")


def memory_usage(df: pd.DataFrame) -> str:
    """Devuelve el uso de memoria del DataFrame en MB."""
    mem = df.memory_usage(deep=True).sum() / 1_048_576
    return f"{mem:.2f} MB"


# ──────────────────────────────────────────────────────────
# CARGA Y LIMPIEZA: EVENTS
# ──────────────────────────────────────────────────────────

def load_and_clean_events(path: str) -> pd.DataFrame:
    """
    Carga events.csv, convierte timestamps y elimina duplicados.

    Returns
    -------
    pd.DataFrame  con columnas: timestamp, visitorid, event,
                                itemid, transactionid, datetime
    """
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.drop_duplicates(inplace=True)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────────────────
# CARGA Y LIMPIEZA: ITEM PROPERTIES
# ──────────────────────────────────────────────────────────

def load_item_properties(path1: str, path2: str = None,
                         sample_n: int = None) -> pd.DataFrame:
    """
    Carga item_properties_part1.csv y opcionalmente part2.
    Permite muestrear para pruebas rápidas.
    """
    df1 = pd.read_csv(path1, nrows=sample_n)
    if path2 and os.path.exists(path2):
        df2 = pd.read_csv(path2, nrows=sample_n)
        df = pd.concat([df1, df2], ignore_index=True)
    else:
        df = df1
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms",
                                     errors="coerce")
    df.drop_duplicates(inplace=True)
    return df


# ──────────────────────────────────────────────────────────
# CARGA Y LIMPIEZA: CATEGORY TREE
# ──────────────────────────────────────────────────────────

def load_category_tree(path: str) -> pd.DataFrame:
    """Carga y limpia el árbol de categorías."""
    df = pd.read_csv(path)
    df.drop_duplicates(inplace=True)
    return df


# ──────────────────────────────────────────────────────────
# MÉTRICAS CLAVE
# ──────────────────────────────────────────────────────────

def calculate_sparsity(df: pd.DataFrame,
                       user_col: str = "visitorid",
                       item_col: str = "itemid") -> dict:
    """
    Calcula la dispersión de la matriz de interacciones usuario-ítem.

    Returns
    -------
    dict  con claves: n_users, n_items, n_interactions, sparsity
    """
    n_users        = df[user_col].nunique()
    n_items        = df[item_col].nunique()
    n_interactions = len(df)
    sparsity       = 1.0 - n_interactions / (n_users * n_items)
    return {
        "n_users"       : n_users,
        "n_items"       : n_items,
        "n_interactions": n_interactions,
        "sparsity"      : sparsity,
    }


def cold_start_analysis(df: pd.DataFrame,
                        user_col: str = "visitorid") -> dict:
    """
    Segmenta usuarios por número de interacciones para cuantificar el Cold-Start.

    Returns
    -------
    dict  con recuentos por segmento
    """
    counts = df[user_col].value_counts()
    return {
        "cold_1"       : int((counts == 1).sum()),
        "cold_2"       : int((counts == 2).sum()),
        "low_3_5"      : int(((counts >= 3) & (counts <= 5)).sum()),
        "active_gt5"   : int((counts > 5).sum()),
        "total_users"  : int(len(counts)),
    }


def conversion_funnel(df: pd.DataFrame,
                      event_col: str = "event") -> pd.DataFrame:
    """
    Construye un DataFrame con el embudo de conversión.

    Returns
    -------
    pd.DataFrame  con columnas: event, count, pct_total, drop_off_pct
    """
    counts = df[event_col].value_counts()
    funnel = pd.DataFrame({
        "event": counts.index,
        "count": counts.values,
    })
    funnel["pct_total"]    = (funnel["count"] / funnel["count"].iloc[0] * 100).round(2)
    funnel["drop_off_pct"] = funnel["count"].pct_change().fillna(0).mul(100).round(2)
    return funnel.reset_index(drop=True)
