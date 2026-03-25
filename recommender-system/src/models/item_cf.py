"""
src/models/item_cf.py
Collaborative Filtering basado en ítems (Item-CF).

Lógica central:
  - Construye una matriz usuario-ítem sparse desde interaction_matrix.parquet
  - Calcula similitud coseno entre ítems usando sus vectores de interacción
  - Para un usuario dado: toma sus ítems con mayor score → busca ítems similares
    → agrega scores ponderados → retorna Top-N

Ventajas sobre User-CF en este dataset:
  - Más estable (ítems cambian menos que usuarios)
  - Escala mejor con 1.4M usuarios
  - Mejor para feedback implícito sparse

Manejo de cold-start:
  - Usuarios sin historial en la matriz → fallback a PopularityRecommender
  - Ítems nuevos sin vector → fallback a similitud de contenido (Sprint 3)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import joblib
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    PROC_DIR,
    MODELS_DIR,
    TOP_N,
    ITEM_CF_NEIGHBORS,
    RANDOM_SEED,
)


class ItemCFRecommender:
    """
    Recomendador Collaborative Filtering ítem-ítem.

    Parámetros
    ----------
    top_n        : número de recomendaciones a retornar
    n_neighbors  : vecinos más cercanos por ítem a pre-computar
    """

    def __init__(self, top_n: int = TOP_N, n_neighbors: int = ITEM_CF_NEIGHBORS):
        self.top_n = top_n
        self.n_neighbors = n_neighbors

        # Artefactos que se construyen en fit()
        self.item_similarity: dict[int, list[tuple[int, float]]] = {}
        self.user_item_scores: dict[int, dict[int, float]] = {}
        self.item_ids: np.ndarray | None = None
        self.user_ids: np.ndarray | None = None
        self.interaction_matrix: sp.csr_matrix | None = None
        self._is_fitted = False

    # ─────────────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, interaction_matrix: pd.DataFrame) -> "ItemCFRecommender":
        """
        Ajusta el modelo calculando similitudes ítem-ítem.

        Parámetros
        ----------
        interaction_matrix : DataFrame con columnas [visitorid, itemid, score]
                             salida de cleaner.build_interaction_matrix()
        """
        logger.info("Ajustando Item-CF...")
        logger.info(
            f"  Usuarios: {interaction_matrix['visitorid'].nunique():,} | "
            f"Ítems: {interaction_matrix['itemid'].nunique():,}"
        )

        # ── Codificar IDs a índices enteros ───────────────────────────────
        self.user_ids = interaction_matrix["visitorid"].unique()
        self.item_ids = interaction_matrix["itemid"].unique()

        user_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        item_idx = {iid: i for i, iid in enumerate(self.item_ids)}
        self._item_idx = item_idx
        self._user_idx = user_idx
        self._idx_to_item = {i: iid for iid, i in item_idx.items()}

        rows = interaction_matrix["visitorid"].map(user_idx).values
        cols = interaction_matrix["itemid"].map(item_idx).values
        data = interaction_matrix["score"].values.astype(np.float32)

        # ── Matriz sparse usuario × ítem ─────────────────────────────────
        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        self.interaction_matrix = sp.csr_matrix(
            (data, (rows, cols)), shape=(n_users, n_items)
        )
        logger.info(
            f"  Matriz sparse: {n_users:,} × {n_items:,} | "
            f"nnz: {self.interaction_matrix.nnz:,}"
        )

        # ── Similitud coseno ítem-ítem (en bloques para eficiencia) ───────
        logger.info(f"  Calculando similitudes ítem-ítem (vecinos: {self.n_neighbors})...")
        item_matrix = self.interaction_matrix.T  # ítems × usuarios

        # Normalizar para coseno
        norms = np.array(item_matrix.power(2).sum(axis=1)).flatten()
        norms = np.where(norms == 0, 1, np.sqrt(norms))
        item_matrix_norm = item_matrix.multiply(1.0 / norms[:, np.newaxis]).tocsr()

        # Calcular en bloques de 1000 ítems para no reventar memoria
        BLOCK_SIZE = 1000
        self.item_similarity = {}

        for start in tqdm(range(0, n_items, BLOCK_SIZE), desc="  Similitud ítems"):
            end = min(start + BLOCK_SIZE, n_items)
            block = item_matrix_norm[start:end]
            sim_block = (block @ item_matrix_norm.T).toarray()

            for local_i, global_i in enumerate(range(start, end)):
                sim_row = sim_block[local_i]
                sim_row[global_i] = 0  # excluir el ítem mismo

                # Top-N vecinos más similares
                top_idx = np.argpartition(sim_row, -self.n_neighbors)[-self.n_neighbors:]
                top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]
                top_scores = sim_row[top_idx]

                # Guardar solo los que tienen similitud > 0
                neighbors = [
                    (self._idx_to_item[j], float(s))
                    for j, s in zip(top_idx, top_scores)
                    if s > 0
                ]
                self.item_similarity[self._idx_to_item[global_i]] = neighbors

        # ── Índice usuario → {ítem: score} para recomendación rápida ─────
        logger.info("  Indexando historial de usuarios...")
        for _, row in interaction_matrix.iterrows():
            uid = row["visitorid"]
            iid = row["itemid"]
            score = row["score"]
            if uid not in self.user_item_scores:
                self.user_item_scores[uid] = {}
            self.user_item_scores[uid][iid] = score

        n_with_neighbors = sum(
            1 for v in self.item_similarity.values() if len(v) > 0
        )
        logger.info(
            f"  Ítems con al menos 1 vecino: {n_with_neighbors:,} / {n_items:,}"
        )
        logger.info("Item-CF ajustado correctamente.")
        self._is_fitted = True
        return self

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
        Retorna Top-N recomendaciones para un usuario.

        Algoritmo:
          1. Recuperar ítems del usuario ordenados por score descendente
          2. Para cada ítem del historial (ponderado por score de interacción):
             agregar scores de sus vecinos similares
          3. Excluir ítems ya vistos (opcional)
          4. Retornar top-N por score agregado

        Parámetros
        ----------
        visitor_id   : ID del visitante
        n            : número de recomendaciones (default: self.top_n)
        exclude_seen : si True, excluye ítems que el usuario ya vio

        Retorna
        -------
        pd.DataFrame con columnas: rank, itemid, score, model
        Retorna DataFrame vacío si el usuario no tiene historial (cold-start)
        """
        assert self._is_fitted, "Modelo no ajustado. Llama a .fit() primero."
        n = n or self.top_n

        user_history = self.user_item_scores.get(visitor_id, {})
        if not user_history:
            return pd.DataFrame()  # cold-start → el caller usa fallback

        seen_items = set(user_history.keys()) if exclude_seen else set()

        # Agregar scores de vecinos ponderados por score de interacción del usuario
        candidate_scores: dict[int, float] = {}

        # Ordenar historial por score descendente, tomar top-20 ítems semilla
        sorted_history = sorted(
            user_history.items(), key=lambda x: x[1], reverse=True
        )[:20]

        for seed_item, interaction_score in sorted_history:
            neighbors = self.item_similarity.get(seed_item, [])
            for neighbor_item, sim_score in neighbors:
                if neighbor_item in seen_items:
                    continue
                weighted = sim_score * interaction_score
                if neighbor_item in candidate_scores:
                    candidate_scores[neighbor_item] += weighted
                else:
                    candidate_scores[neighbor_item] = weighted

        if not candidate_scores:
            return pd.DataFrame()

        # Top-N candidatos
        top_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:n]

        recs = pd.DataFrame(top_items, columns=["itemid", "score"])
        recs["rank"] = recs.index + 1
        recs["model"] = "item_cf"

        # Normalizar scores a [0, 1]
        max_score = recs["score"].max()
        if max_score > 0:
            recs["score"] = (recs["score"] / max_score).round(4)

        return recs

    def recommend_similar(
        self,
        item_id: int,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        Retorna ítems similares a un ítem dado (recomendación ítem-a-ítem).
        Útil para páginas de detalle de producto.

        Retorna
        -------
        pd.DataFrame con columnas: rank, itemid, similarity_score, model
        """
        assert self._is_fitted, "Modelo no ajustado."
        n = n or self.top_n

        neighbors = self.item_similarity.get(item_id, [])
        if not neighbors:
            return pd.DataFrame()

        recs = pd.DataFrame(neighbors[:n], columns=["itemid", "similarity_score"])
        recs["rank"] = recs.index + 1
        recs["model"] = "item_cf_similar"
        return recs

    # ─────────────────────────────────────────────────────────────────────────
    # PERSISTENCIA
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "item_cf_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Modelo guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ItemCFRecommender":
        path = path or MODELS_DIR / "item_cf_model.pkl"
        model = joblib.load(path)
        logger.info(f"Modelo cargado: {path}")
        return model

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ItemCFRecommender("
            f"top_n={self.top_n}, "
            f"n_neighbors={self.n_neighbors}, "
            f"status={status})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    logger.info("Cargando matriz de interacciones...")
    interaction_matrix = pd.read_parquet(PROC_DIR / "interaction_matrix.parquet")
    logger.info(f"  Shape: {interaction_matrix.shape}")

    model = ItemCFRecommender(top_n=10, n_neighbors=50)
    model.fit(interaction_matrix)
    model.save()

    # Test rápido
    sample_user = interaction_matrix["visitorid"].iloc[0]
    recs = model.recommend(sample_user, n=10)
    logger.info(f"\nRecomendaciones para usuario {sample_user}:")
    logger.info(f"\n{recs}")

    sample_item = interaction_matrix["itemid"].iloc[0]
    similar = model.recommend_similar(sample_item, n=5)
    logger.info(f"\nÍtems similares a {sample_item}:")
    logger.info(f"\n{similar}")
