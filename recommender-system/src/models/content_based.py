"""
src/models/content_based.py
Content-Based Filtering usando perfiles TF-IDF de item_properties.

Lógica central:
  - Cada ítem tiene un vector TF-IDF construido desde sus tokens property_value
  - Para un usuario: promedia los vectores de sus ítems más interactuados
    (ponderado por score de interacción) → "perfil de usuario"
  - Recomienda ítems cuyo vector TF-IDF es más cercano (coseno) al perfil

Ventajas:
  - Funciona para ítems nuevos sin interacciones (cold-start de ítems)
  - No necesita historial de otros usuarios
  - Explota las propiedades del catálogo que CF ignora

Limitaciones:
  - Solo recomienda ítems similares a lo que ya vio (burbuja de filtro)
  - Por eso se combina con CF en el modelo híbrido
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    PROC_DIR,
    MODELS_DIR,
    TOP_N,
    CBF_TOP_SIMILAR,
    RANDOM_SEED,
)


class ContentBasedRecommender:
    """
    Recomendador Content-Based usando TF-IDF sobre propiedades de ítems.

    Parámetros
    ----------
    top_n       : recomendaciones a retornar
    top_similar : ítems similares a pre-computar por ítem semilla
    """

    def __init__(self, top_n: int = TOP_N, top_similar: int = CBF_TOP_SIMILAR):
        self.top_n       = top_n
        self.top_similar = top_similar

        self.tfidf_matrix: sp.csr_matrix | None = None
        self.item_ids: np.ndarray | None = None
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.item_categories: dict[int, str] = {}
        self.user_history: dict[int, dict[int, float]] = {}
        self._is_fitted = False

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        tfidf_matrix: sp.csr_matrix,
        item_profiles: pd.DataFrame,
        interaction_matrix: pd.DataFrame,
    ) -> "ContentBasedRecommender":
        """
        Ajusta el modelo cargando la matriz TF-IDF y el historial de usuarios.

        Parámetros
        ----------
        tfidf_matrix      : matriz sparse (n_items × n_features) de feature_engineering
        item_profiles     : DataFrame con columnas [itemid, category_id]
        interaction_matrix: DataFrame con columnas [visitorid, itemid, score]
        """
        logger.info("Ajustando modelo Content-Based...")

        self.tfidf_matrix = tfidf_matrix
        self.item_ids     = item_profiles["itemid"].values

        self.item_to_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self.idx_to_item = {i: iid for iid, i in self.item_to_idx.items()}
        self.item_categories = dict(zip(
            item_profiles["itemid"],
            item_profiles["category_id"]
        ))

        logger.info(
            f"  TF-IDF: {tfidf_matrix.shape[0]:,} ítems × "
            f"{tfidf_matrix.shape[1]:,} features"
        )

        # Historial de usuarios ponderado
        self.user_history = (
            interaction_matrix
            .groupby("visitorid")
            .apply(lambda g: dict(zip(g["itemid"], g["score"])))
            .to_dict()
        )

        logger.info(f"  Usuarios indexados: {len(self.user_history):,}")
        self._is_fitted = True
        return self

    # ─────────────────────────────────────────────────────────────────────────
    # PERFIL DE USUARIO
    # ─────────────────────────────────────────────────────────────────────────

    def _build_user_profile(self, visitor_id: int) -> np.ndarray | None:
        """
        Construye el perfil del usuario promediando vectores TF-IDF
        de sus ítems más interactuados, ponderados por score.
        """
        history = self.user_history.get(visitor_id, {})
        if not history:
            return None

        # Tomar top-10 ítems del historial por score
        top_items = sorted(history.items(), key=lambda x: x[1], reverse=True)[:10]

        vectors = []
        weights = []
        for iid, score in top_items:
            idx = self.item_to_idx.get(iid)
            if idx is not None:
                vectors.append(self.tfidf_matrix[idx].toarray().flatten())
                weights.append(score)

        if not vectors:
            return None

        weights = np.array(weights)
        weights = weights / weights.sum()  # normalizar

        profile = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm

        return profile

    # ─────────────────────────────────────────────────────────────────────────
    # RECOMMEND
    # ─────────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        visitor_id: int,
        n: int | None = None,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Retorna Top-N recomendaciones basadas en contenido para un usuario.

        Parámetros
        ----------
        visitor_id   : ID del visitante
        n            : número de recomendaciones
        exclude_seen : excluir ítems ya vistos

        Retorna
        -------
        pd.DataFrame con columnas: rank, itemid, score, model
        DataFrame vacío si el usuario no tiene historial (cold-start)
        """
        assert self._is_fitted, "Modelo no ajustado."
        n = n or self.top_n

        user_profile = self._build_user_profile(visitor_id)
        if user_profile is None:
            return pd.DataFrame()

        # Similitud coseno entre el perfil del usuario y todos los ítems
        profile_vec = user_profile.reshape(1, -1)
        scores = cosine_similarity(profile_vec, self.tfidf_matrix).flatten()

        # Excluir ítems ya vistos
        if exclude_seen:
            seen = set(self.user_history.get(visitor_id, {}).keys())
            for iid in seen:
                idx = self.item_to_idx.get(iid)
                if idx is not None:
                    scores[idx] = -1

        # Top-N
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recs = pd.DataFrame({
            "itemid": [self.idx_to_item[i] for i in top_idx],
            "score":  scores[top_idx].round(4),
        })
        recs["rank"]  = recs.index + 1
        recs["model"] = "content_based"

        return recs

    def recommend_similar_items(
        self,
        item_id: int,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Retorna ítems similares a un ítem dado por similitud de contenido.
        Útil para páginas de detalle de producto y cold-start de ítems.
        """
        assert self._is_fitted
        n = n or self.top_n

        idx = self.item_to_idx.get(item_id)
        if idx is None:
            return pd.DataFrame()

        item_vec = self.tfidf_matrix[idx]
        scores   = cosine_similarity(item_vec, self.tfidf_matrix).flatten()
        scores[idx] = -1  # excluir el ítem mismo

        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recs = pd.DataFrame({
            "itemid":     [self.idx_to_item[i] for i in top_idx],
            "score":      scores[top_idx].round(4),
            "category_id": [self.item_categories.get(self.idx_to_item[i], "?")
                            for i in top_idx],
        })
        recs["rank"]  = recs.index + 1
        recs["model"] = "content_based_similar"
        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCIA
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "content_based_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Modelo Content-Based guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ContentBasedRecommender":
        path = path or MODELS_DIR / "content_based_model.pkl"
        model = joblib.load(path)
        logger.info(f"Modelo Content-Based cargado: {path}")
        return model

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return f"ContentBasedRecommender(top_n={self.top_n}, status={status})"


if __name__ == "__main__":
    tfidf_matrix  = sp.load_npz(PROC_DIR / "tfidf_matrix.npz")
    item_profiles = pd.read_parquet(PROC_DIR / "item_profiles.parquet")
    interaction   = pd.read_parquet(PROC_DIR / "interaction_matrix.parquet")

    model = ContentBasedRecommender(top_n=10)
    model.fit(tfidf_matrix, item_profiles, interaction)
    model.save()

    sample_user = interaction["visitorid"].iloc[0]
    recs = model.recommend(sample_user, n=10)
    logger.info(f"\nRecomendaciones CBF para usuario {sample_user}:\n{recs}")
