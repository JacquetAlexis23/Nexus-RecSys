"""
src/models/popularity.py
Modelo Baseline — Popularidad Global y por Categoría.

Sirve como:
  1. Benchmark de referencia para comparar todos los modelos
  2. Fallback para usuarios cold-start
  3. Componente del sistema de recomendación para nuevas sesiones

Lógica:
  - Score de popularidad = suma ponderada de interacciones de todos los usuarios
  - Versión global: top-N ítems más populares del catálogo
  - Versión por categoría: top-N ítems de la categoría del ítem en foco
"""

import pandas as pd
import numpy as np
from loguru import logger
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import MODELS_DIR, TOP_N, EVENT_WEIGHTS


class PopularityRecommender:
    """
    Recomendador basado en popularidad global y por categoría.
    
    Parameters
    ----------
    top_n : int — número de recomendaciones a retornar
    """

    def __init__(self, top_n: int = TOP_N):
        self.top_n = top_n
        self.global_popular: pd.DataFrame | None = None
        self.category_popular: dict[str, pd.DataFrame] = {}
        self.item_to_category: dict[int, str] = {}
        self._is_fitted = False

    def fit(
        self,
        events_clean: pd.DataFrame,
        item_profiles: pd.DataFrame,
    ) -> "PopularityRecommender":
        """
        Ajusta el modelo calculando scores de popularidad.

        Parameters
        ----------
        events_clean  : DataFrame de eventos limpios con event_weight
        item_profiles : DataFrame con columnas itemid, category_id
        """
        logger.info("Ajustando modelo de popularidad...")

        # Popularidad global ponderada por tipo de evento
        global_scores = (
            events_clean
            .groupby("itemid")["event_weight"]
            .sum()
            .reset_index()
            .rename(columns={"event_weight": "popularity_score"})
            .sort_values("popularity_score", ascending=False)
        )

        # Contar transacciones reales (métrica de calidad)
        transaction_counts = (
            events_clean[events_clean["event"] == "transaction"]
            .groupby("itemid")["event"].count()
            .reset_index()
            .rename(columns={"event": "n_transactions"})
        )

        self.global_popular = global_scores.merge(
            item_profiles[["itemid", "category_id"]],
            on="itemid", how="left"
        ).merge(transaction_counts, on="itemid", how="left")
        self.global_popular["n_transactions"] = (
            self.global_popular["n_transactions"].fillna(0).astype("int32")
        )

        # Mapeo ítem → categoría
        self.item_to_category = (
            item_profiles.set_index("itemid")["category_id"].to_dict()
        )

        # Popularidad por categoría
        for cat_id, group in self.global_popular.groupby("category_id"):
            self.category_popular[str(cat_id)] = (
                group.sort_values("popularity_score", ascending=False)
            )

        logger.info(
            f"  Modelo ajustado: {len(self.global_popular):,} ítems | "
            f"{len(self.category_popular):,} categorías"
        )
        logger.info(
            f"  Top-3 ítems globales:\n"
            f"{self.global_popular.head(3)[['itemid','popularity_score','n_transactions']]}"
        )

        self._is_fitted = True
        return self

    def recommend(
        self,
        visitor_id: int | None = None,
        category_id: str | None = None,
        exclude_items: list[int] | None = None,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Retorna top-N recomendaciones.

        Parameters
        ----------
        visitor_id    : ignorado (baseline no personaliza), incluido por interfaz común
        category_id   : si se especifica, recomienda dentro de esa categoría
        exclude_items : ítems a excluir (ej. los que el usuario ya vio)
        n             : número de recomendaciones (default: self.top_n)

        Returns
        -------
        pd.DataFrame con columnas: itemid, score, category_id, rank
        """
        assert self._is_fitted, "Modelo no ajustado. Llama a .fit() primero."
        n = n or self.top_n
        exclude = set(exclude_items or [])

        if category_id and str(category_id) in self.category_popular:
            pool = self.category_popular[str(category_id)]
        else:
            pool = self.global_popular

        recs = (
            pool[~pool["itemid"].isin(exclude)]
            .head(n)
            [["itemid", "popularity_score", "category_id"]]
            .rename(columns={"popularity_score": "score"})
            .reset_index(drop=True)
        )
        recs["rank"] = recs.index + 1
        recs["model"] = "popularity"

        return recs

    def get_category_for_item(self, item_id: int) -> str | None:
        return self.item_to_category.get(item_id)

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "popularity_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Modelo guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "PopularityRecommender":
        path = path or MODELS_DIR / "popularity_model.pkl"
        model = joblib.load(path)
        logger.info(f"Modelo cargado: {path}")
        return model


if __name__ == "__main__":
    from pathlib import Path
    PROC_DIR = Path("data/processed")
    
    events_clean  = pd.read_parquet(PROC_DIR / "events_clean.parquet")
    item_profiles = pd.read_parquet(PROC_DIR / "item_profiles.parquet")

    model = PopularityRecommender(top_n=10)
    model.fit(events_clean, item_profiles)
    
    recs = model.recommend(n=10)
    print("\nTop-10 recomendaciones globales:")
    print(recs)
    
    model.save()
