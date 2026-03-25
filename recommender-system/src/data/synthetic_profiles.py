"""
src/data/synthetic_profiles.py
Generación de perfiles demográficos sintéticos de usuarios con Faker.
Simula variables de contexto ausentes por privacidad en el dataset original:
  - Edad estimada
  - Región geográfica (Colombia)
  - Segmento de cliente (occasional / regular / vip)
  - Device type

Los segmentos se condicionan por el comportamiento real del usuario
(transacciones reales → mayor probabilidad de ser VIP).
"""

import pandas as pd
import numpy as np
from faker import Faker
from loguru import logger
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    SYNTH_DIR,
    SYNTH_N_USERS,
    SYNTH_REGIONS,
    SYNTH_SEGMENTS,
    RANDOM_SEED,
)

SYNTH_DIR.mkdir(parents=True, exist_ok=True)
fake = Faker("es_CO")
np.random.seed(RANDOM_SEED)


def _assign_segment_from_behavior(events_clean: pd.DataFrame) -> pd.Series:
    """
    Asigna segmento condicionado por comportamiento real:
      - VIP     : ≥ 1 transacción
      - Regular : ≥ 2 addtocart sin transacción
      - Occasional: resto
    """
    transactions = (
        events_clean[events_clean["event"] == "transaction"]
        .groupby("visitorid")["event"].count()
    )
    carts = (
        events_clean[events_clean["event"] == "addtocart"]
        .groupby("visitorid")["event"].count()
    )

    all_visitors = events_clean["visitorid"].unique()
    segments = {}
    for v in all_visitors:
        if transactions.get(v, 0) >= 1:
            segments[v] = "vip"
        elif carts.get(v, 0) >= 2:
            segments[v] = "regular"
        else:
            segments[v] = "occasional"

    return pd.Series(segments, name="segment")


def generate_user_profiles(
    events_clean: pd.DataFrame,
    n_synthetic: int = SYNTH_N_USERS,
) -> pd.DataFrame:
    """
    Genera perfiles demográficos sintéticos para usuarios del dataset.
    
    Para usuarios con historial real → segmento condicionado por comportamiento.
    Para usuarios sintéticos adicionales → distribución aleatoria calibrada.

    Returns
    -------
    pd.DataFrame con columnas:
      visitorid, age, region, segment, device, is_synthetic
    """
    logger.info("Generando perfiles demográficos sintéticos...")

    # Segmentos basados en comportamiento real
    real_segments = _assign_segment_from_behavior(events_clean)
    real_visitors = real_segments.index.tolist()

    # Distribución de segmentos observada
    seg_dist = real_segments.value_counts(normalize=True)
    logger.info(f"  Distribución de segmentos (usuarios reales):\n{seg_dist}")

    # Parámetros de edad por segmento (realista para e-commerce Colombia)
    age_params = {
        "occasional": (28, 12),   # mu, sigma
        "regular":    (32, 10),
        "vip":        (38, 8),
    }

    device_probs = {
        "mobile":  0.62,
        "desktop": 0.30,
        "tablet":  0.08,
    }

    def _generate_rows(visitor_ids, segments_map, is_synthetic: bool):
        rows = []
        for vid in visitor_ids:
            seg = segments_map.get(vid, np.random.choice(
                SYNTH_SEGMENTS,
                p=[seg_dist.get(s, 1/3) for s in SYNTH_SEGMENTS]
            ))
            mu, sigma = age_params[seg]
            age = int(np.clip(np.random.normal(mu, sigma), 18, 70))
            rows.append({
                "visitorid":    vid,
                "age":          age,
                "age_group":    _age_group(age),
                "region":       np.random.choice(SYNTH_REGIONS),
                "segment":      seg,
                "device":       np.random.choice(
                    list(device_probs.keys()),
                    p=list(device_probs.values())
                ),
                "is_synthetic": is_synthetic,
            })
        return rows

    # Perfiles para usuarios reales
    real_rows = _generate_rows(
        real_visitors,
        real_segments.to_dict(),
        is_synthetic=False
    )

    # Perfiles para usuarios puramente sintéticos (simulan tráfico futuro)
    fake_visitors = [-(i + 1) for i in range(n_synthetic)]  # IDs negativos
    seg_array = np.random.choice(
        SYNTH_SEGMENTS,
        size=n_synthetic,
        p=[seg_dist.get(s, 1/3) for s in SYNTH_SEGMENTS]
    )
    synth_rows = _generate_rows(
        fake_visitors,
        dict(zip(fake_visitors, seg_array)),
        is_synthetic=True
    )

    df = pd.DataFrame(real_rows + synth_rows)
    df["visitorid"] = df["visitorid"].astype("int64")

    # Estadísticas
    logger.info(f"  Perfiles generados: {len(df):,}")
    logger.info(f"    Reales: {len(real_rows):,} | Sintéticos: {len(synth_rows):,}")
    logger.info(f"  Edad promedio: {df['age'].mean():.1f} años")
    logger.info(f"  Top región: {df['region'].value_counts().index[0]}")

    # Guardar
    out_path = SYNTH_DIR / "user_profiles.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"  Perfiles guardados en: {out_path}")

    return df


def _age_group(age: int) -> str:
    if age < 25:    return "18-24"
    elif age < 35:  return "25-34"
    elif age < 45:  return "35-44"
    elif age < 55:  return "45-54"
    else:           return "55+"


if __name__ == "__main__":
    events_clean = pd.read_parquet(
        Path(__file__).resolve().parents[2] / "data/processed/events_clean.parquet"
    )
    profiles = generate_user_profiles(events_clean)
    print(profiles.head())
    print(profiles["segment"].value_counts())
