"""
src/models/als_model.py
Matrix Factorization con feedback implícito usando cornac.BPR o cornac.WRMF.

Intenta WRMF primero (ALS verdadero para feedback implícito).
Si no está disponible, usa BPR (Bayesian Personalized Ranking).
Ambos están diseñados para datos de interacción implícita.
"""

import numpy as np
import pandas as pd
import cornac
from cornac.data import Dataset
from loguru import logger
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config.config import (
    PROC_DIR, MODELS_DIR, TOP_N,
    ALS_FACTORS, ALS_ITERATIONS, ALS_REGULARIZATION, RANDOM_SEED,
)

# Detectar qué modelo de feedback implícito está disponible
def _get_implicit_model(n_factors, n_epochs, reg, seed):
    """
    Prioridad: WRMF → WMF → BPR → MF
    WRMF y WMF son ALS verdadero para feedback implícito.
    BPR es ranking personalizado bayesiano, también para feedback implícito.
    """
    try:
        from cornac.models import WRMF
        logger.info("  Usando modelo: WRMF (ALS para feedback implícito)")
        return WRMF(
            k=n_factors,
            max_iter=n_epochs,
            lambda_reg=reg,
            seed=seed,
            verbose=True,
        ), "WRMF"
    except (ImportError, AttributeError):
        pass

    # WMF requiere TensorFlow — omitir

    try:
        from cornac.models import BPR
        logger.info("  Usando modelo: BPR (Bayesian Personalized Ranking)")
        return BPR(
            k=n_factors,
            max_iter=n_epochs,
            lambda_reg=reg,
            seed=seed,
            verbose=True,
        ), "BPR"
    except (ImportError, AttributeError):
        pass

    from cornac.models import MF
    logger.warning("  Fallback a MF estándar (puede no converger con feedback implícito)")
    return MF(
        k=n_factors,
        max_iter=n_epochs,
        lambda_reg=reg,
        use_bias=True,
        seed=seed,
        verbose=True,
    ), "MF"


class ALSRecommender:
    """
    Recomendador con Matrix Factorization para feedback implícito.
    Usa WRMF/WMF/BPR según disponibilidad en la versión instalada de cornac.
    """

    def __init__(self, n_factors=ALS_FACTORS, n_epochs=ALS_ITERATIONS,
                 reg=ALS_REGULARIZATION, top_n=TOP_N):
        self.n_factors  = n_factors
        self.n_epochs   = n_epochs
        self.reg        = reg
        self.top_n      = top_n
        self.model_type = None

        self.model      = None
        self.train_set  = None
        self.user_history: dict[int, set] = {}
        self._uid_to_idx: dict[str, int] = {}
        self._iid_to_idx: dict[str, int] = {}
        self._idx_to_iid: dict[int, int] = {}
        self._is_fitted = False

    def fit(self, interaction_matrix: pd.DataFrame) -> "ALSRecommender":
        logger.info("Ajustando modelo ALS con feedback implícito...")
        logger.info(
            f"  Usuarios: {interaction_matrix['visitorid'].nunique():,} | "
            f"Ítems: {interaction_matrix['itemid'].nunique():,} | "
            f"Factores: {self.n_factors} | Épocas: {self.n_epochs}"
        )

        # Historial de usuarios para exclusión de vistos
        self.user_history = (
            interaction_matrix
            .groupby("visitorid")["itemid"]
            .apply(set).to_dict()
        )

        # Datos UIR (user, item, rating) — cornac requiere strings
        uir_data = [
            (str(r["visitorid"]), str(r["itemid"]), float(r["score"]))
            for _, r in interaction_matrix.iterrows()
        ]

        logger.info("  Construyendo Dataset cornac...")
        self.train_set = Dataset.from_uir(uir_data, seed=RANDOM_SEED)

        # Mapas inversos construidos manualmente
        self._iid_to_idx = dict(self.train_set.iid_map)
        self._idx_to_iid = {v: int(k) for k, v in self._iid_to_idx.items()}
        self._uid_to_idx = dict(self.train_set.uid_map)

        logger.info(
            f"  Mapas: {len(self._iid_to_idx):,} ítems | "
            f"{len(self._uid_to_idx):,} usuarios"
        )

        # Seleccionar modelo disponible
        self.model, self.model_type = _get_implicit_model(
            self.n_factors, self.n_epochs, self.reg, RANDOM_SEED
        )

        logger.info(f"  Entrenando {self.model_type}...")
        self.model.fit(self.train_set)

        logger.info(
            f"{self.model_type} ajustado. "
            f"Usuarios: {self.train_set.num_users:,} | "
            f"Ítems: {self.train_set.num_items:,}"
        )
        self._is_fitted = True
        return self

    def recommend(self, visitor_id: int, n: int | None = None,
                  exclude_seen: bool = True) -> pd.DataFrame:
        assert self._is_fitted, "Modelo no ajustado."
        n = n or self.top_n

        user_str = str(visitor_id)
        if user_str not in self._uid_to_idx:
            return pd.DataFrame()

        user_idx = self._uid_to_idx[user_str]

        try:
            scores = self.model.score(user_idx).copy()
        except Exception as e:
            logger.debug(f"Error en score para usuario {visitor_id}: {e}")
            return pd.DataFrame()

        # Excluir ítems ya vistos
        if exclude_seen:
            for iid in self.user_history.get(visitor_id, set()):
                idx = self._iid_to_idx.get(str(iid))
                if idx is not None:
                    scores[idx] = -np.inf

        # Top-N
        valid = np.isfinite(scores)
        if valid.sum() < n:
            n = int(valid.sum())
        if n == 0:
            return pd.DataFrame()

        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        recs = []
        for rank, idx in enumerate(top_idx, 1):
            orig_iid = self._idx_to_iid.get(int(idx))
            if orig_iid is None:
                continue
            recs.append({
                "rank":   rank,
                "itemid": orig_iid,
                "score":  float(scores[idx]),
                "model":  self.model_type.lower(),
            })

        df = pd.DataFrame(recs)
        if not df.empty:
            mx, mn = df["score"].max(), df["score"].min()
            rng = mx - mn
            if rng > 0:
                df["score"] = ((df["score"] - mn) / rng).round(4)
        return df

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "als_model.pkl"
        joblib.dump(self, path)
        logger.info(f"Modelo {self.model_type} guardado: {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ALSRecommender":
        path = path or MODELS_DIR / "als_model.pkl"
        model = joblib.load(path)
        logger.info(f"Modelo cargado: {path}")
        return model

    def __repr__(self):
        s = "fitted" if self._is_fitted else "not fitted"
        return f"ALSRecommender(type={self.model_type}, n_factors={self.n_factors}, status={s})"


if __name__ == "__main__":
    interaction_matrix = pd.read_parquet(PROC_DIR / "interaction_matrix.parquet")
    model = ALSRecommender(n_factors=64, n_epochs=20)
    model.fit(interaction_matrix)
    model.save()
    sample = interaction_matrix["visitorid"].iloc[0]
    recs = model.recommend(sample, n=5)
    logger.info(f"\nRecomendaciones para {sample}:\n{recs}")
