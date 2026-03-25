"""
src/models/hybrid.py
Modelo Híbrido — Ensamble de ALS + Item-CF + Content-Based.

Estrategia:
  - Warm users: ALS (60%) + Item-CF (40%) — mejor balance precisión/cobertura
  - Si ALS falla: Item-CF (100%)
  - Si Item-CF falla: CBF (100%)
  - Cold-start total: Popularity por categoría
"""

import numpy as np
import pandas as pd
from loguru import logger
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import MODELS_DIR, TOP_N, HYBRID_WEIGHT_CF, HYBRID_WEIGHT_CBF
from src.models.popularity    import PopularityRecommender
from src.models.als_model     import ALSRecommender
from src.models.item_cf       import ItemCFRecommender
from src.models.content_based import ContentBasedRecommender


class HybridRecommender:
    """
    Recomendador híbrido: ALS + Item-CF con fallback en cascada.

    Parámetros
    ----------
    weight_als  : peso del modelo ALS (default 0.6)
    weight_cf   : peso del Item-CF   (default 0.4)
    top_n       : recomendaciones a retornar
    """

    def __init__(self, weight_als: float = 0.6, weight_cf: float = 0.4,
                 top_n: int = TOP_N):
        assert abs(weight_als + weight_cf - 1.0) < 1e-6, "Pesos deben sumar 1.0"
        self.weight_als = weight_als
        self.weight_cf  = weight_cf
        self.top_n      = top_n

        self.als_model: ALSRecommender | None = None
        self.cf_model:  ItemCFRecommender | None = None
        self.cbf_model: ContentBasedRecommender | None = None
        self.pop_model: PopularityRecommender | None = None
        self._is_fitted = False

    def fit(self, als_model, cf_model, cbf_model, pop_model) -> "HybridRecommender":
        self.als_model = als_model
        self.cf_model  = cf_model
        self.cbf_model = cbf_model
        self.pop_model = pop_model
        self._is_fitted = True
        logger.info(
            f"Modelo Híbrido configurado: "
            f"ALS={self.weight_als:.0%} + Item-CF={self.weight_cf:.0%}"
        )
        return self

    def recommend(self, visitor_id: int, n: int | None = None,
                  exclude_seen: bool = True,
                  category_id: str | None = None) -> pd.DataFrame:
        assert self._is_fitted, "Modelo no configurado."
        n = n or self.top_n
        fetch_n = n * 3

        recs_als = self.als_model.recommend(visitor_id, n=fetch_n, exclude_seen=exclude_seen)
        recs_cf  = self.cf_model.recommend(visitor_id, n=fetch_n, exclude_seen=exclude_seen)

        als_ok = not recs_als.empty
        cf_ok  = not recs_cf.empty

        if als_ok and cf_ok:
            strategy = "hybrid_als_cf"
            result   = self._blend(recs_als, recs_cf, n)
        elif cf_ok:
            strategy = "item_cf_only"
            result   = recs_cf.head(n).copy()
        elif als_ok:
            strategy = "als_only"
            result   = recs_als.head(n).copy()
        else:
            # Intentar CBF
            recs_cbf = self.cbf_model.recommend(visitor_id, n=n, exclude_seen=exclude_seen)
            if not recs_cbf.empty:
                strategy = "content_based_fallback"
                result   = recs_cbf.head(n).copy()
            else:
                strategy = "popularity_fallback"
                result   = self.pop_model.recommend(category_id=category_id, n=n)

        result["model"]    = "hybrid"
        result["strategy"] = strategy
        result["rank"]     = range(1, len(result) + 1)
        return result.head(n)

    def _blend(self, recs_als: pd.DataFrame, recs_cf: pd.DataFrame,
               n: int) -> pd.DataFrame:
        als_scores = dict(zip(recs_als["itemid"], recs_als["score"]))
        cf_scores  = dict(zip(recs_cf["itemid"],  recs_cf["score"]))
        all_items  = set(als_scores) | set(cf_scores)

        blended = {
            iid: als_scores.get(iid, 0.0) * self.weight_als
               + cf_scores.get(iid, 0.0)  * self.weight_cf
            for iid in all_items
        }

        top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:n]
        df  = pd.DataFrame(top, columns=["itemid", "score"])
        df["score"] = df["score"].round(4)
        return df

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "hybrid_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Modelo Híbrido guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "HybridRecommender":
        path = path or MODELS_DIR / "hybrid_model.pkl"
        model = joblib.load(path)
        logger.info(f"Modelo Híbrido cargado: {path}")
        return model

    def __repr__(self):
        s = "fitted" if self._is_fitted else "not fitted"
        return f"HybridRecommender(ALS={self.weight_als:.0%}, CF={self.weight_cf:.0%}, status={s})"
