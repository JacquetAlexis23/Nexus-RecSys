"""
src/models/ease_model.py
EASE — Embarrassingly Shallow Autoencoders for Sparse Data
(Steck, 2019)

Optimizado para memoria: trabaja solo con los top_items mas interactuados
para evitar el problema de memoria con matrices de 44K x 44K items.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from loguru import logger
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import PROC_DIR, MODELS_DIR, TOP_N, RANDOM_SEED


class EASERecommender:
    """
    EASE con limitacion de items para control de memoria.

    Parametros
    ----------
    lambda_reg : regularizacion L2 (default 500)
    top_n      : recomendaciones a retornar
    max_items  : maximo de items a usar (controla memoria)
                 44639 items → 14.8 GB RAM
                 10000 items → 0.75 GB RAM
                 5000  items → 0.19 GB RAM
    """

    def __init__(self, lambda_reg: float = 500.0, top_n: int = TOP_N,
                 max_items: int = 10000):
        self.lambda_reg = lambda_reg
        self.top_n      = top_n
        self.max_items  = max_items

        self.B: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_ids: np.ndarray | None = None
        self.item_to_idx: dict[int, int] = {}
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.user_history: dict[int, set] = {}
        self.interaction_matrix: sp.csr_matrix | None = None
        self._is_fitted = False

    def fit(self, interaction_matrix: pd.DataFrame) -> "EASERecommender":
        logger.info("Ajustando modelo EASE...")

        # Filtrar a los top_items mas populares para controlar memoria
        item_popularity = interaction_matrix.groupby("itemid")["score"].sum()
        top_items = item_popularity.nlargest(self.max_items).index
        df = interaction_matrix[interaction_matrix["itemid"].isin(top_items)].copy()

        n_items_orig = interaction_matrix["itemid"].nunique()
        logger.info(
            f"  Items originales: {n_items_orig:,} | "
            f"Items usados (top {self.max_items:,}): {df['itemid'].nunique():,} | "
            f"Usuarios: {df['visitorid'].nunique():,} | Lambda: {self.lambda_reg}"
        )

        # Mapeos
        self.user_ids    = df["visitorid"].unique()
        self.item_ids    = df["itemid"].unique()
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {it: i for i, it in enumerate(self.item_ids)}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)

        # Historial
        self.user_history = (
            interaction_matrix
            .groupby("visitorid")["itemid"]
            .apply(set).to_dict()
        )

        # Matriz sparse
        rows = df["visitorid"].map(self.user_to_idx).values
        cols = df["itemid"].map(self.item_to_idx).values
        data = df["score"].values.astype(np.float32)
        X = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.interaction_matrix = X

        ram_gb = (n_items ** 2 * 8) / 1e9
        logger.info(
            f"  Matriz sparse: {n_users:,} x {n_items:,} | "
            f"nnz: {X.nnz:,} | RAM estimada para G: {ram_gb:.2f} GB"
        )

        # EASE — formula cerrada
        logger.info("  Calculando G = X^T X...")
        G = (X.T @ X).toarray().astype(np.float64)

        diag_idx = np.arange(n_items)
        G[diag_idx, diag_idx] += self.lambda_reg

        logger.info("  Invirtiendo G...")
        P = np.linalg.inv(G)

        B = P / (-np.diag(P))
        B[diag_idx, diag_idx] = 0.0
        self.B = B.astype(np.float32)

        logger.info(f"  EASE ajustado. Matriz B: {self.B.shape}")
        self._is_fitted = True
        return self

    def recommend(self, visitor_id: int, n: int | None = None,
                  exclude_seen: bool = True) -> pd.DataFrame:
        assert self._is_fitted
        n = n or self.top_n

        user_idx = self.user_to_idx.get(visitor_id)
        if user_idx is None:
            return pd.DataFrame()

        user_vec = self.interaction_matrix[user_idx].toarray().flatten()
        scores   = user_vec @ self.B

        if exclude_seen:
            for iid in self.user_history.get(visitor_id, set()):
                idx = self.item_to_idx.get(iid)
                if idx is not None:
                    scores[idx] = -np.inf

        valid = np.isfinite(scores)
        n_actual = min(n, valid.sum())
        if n_actual == 0:
            return pd.DataFrame()

        top_idx = np.argpartition(scores, -n_actual)[-n_actual:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recs = pd.DataFrame({
            "rank":   range(1, len(top_idx) + 1),
            "itemid": [self.idx_to_item[int(i)] for i in top_idx],
            "score":  scores[top_idx],
            "model":  "ease",
        })

        mx, mn = recs["score"].max(), recs["score"].min()
        if mx - mn > 0:
            recs["score"] = ((recs["score"] - mn) / (mx - mn)).round(4)
        return recs

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "ease_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Modelo EASE guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "EASERecommender":
        path = path or MODELS_DIR / "ease_model.pkl"
        model = joblib.load(path)
        logger.info(f"Modelo EASE cargado: {path}")
        return model

    def __repr__(self):
        s = "fitted" if self._is_fitted else "not fitted"
        return f"EASERecommender(lambda={self.lambda_reg}, max_items={self.max_items}, status={s})"
