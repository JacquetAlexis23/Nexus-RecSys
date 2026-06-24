"""
src/api/main.py
API REST — RetailRocket Recommender System
Sprint 4 — NexusDataCo

Endpoints:
  GET /                          → info del sistema
  GET /v1/health                 → estado de los modelos
  GET /v1/recommend/{visitor_id} → recomendaciones para un usuario
  GET /v1/recommend/popular      → top productos globales
  GET /v1/metrics                → metricas comparativas de los 6 modelos
  GET /v1/models                 → lista de modelos disponibles
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.popularity    import PopularityRecommender
from src.models.item_cf       import ItemCFRecommender
from src.models.als_model     import ALSRecommender
from src.models.ease_model    import EASERecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid        import HybridRecommender

# ── Estado global ─────────────────────────────────────────────────────────────
MODELS     = {}
METRICS_DF = None
ITEM_CATS  = {}

# ── Lifespan (carga modelos al arrancar, limpia al apagar) ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS, METRICS_DF, ITEM_CATS
    print("Cargando modelos...")

    loaders = {
        "popularity":    PopularityRecommender,
        "item_cf":       ItemCFRecommender,
        "als":           ALSRecommender,
        "ease":          EASERecommender,
        "content_based": ContentBasedRecommender,
        "hybrid":        HybridRecommender,
    }
    for name, cls in loaders.items():
        try:
            MODELS[name] = cls.load()
        except Exception as e:
            print(f"  [{name}] ERROR: {e}")

    try:
        mpath = ROOT / "data" / "processed" / "metrics_all_models.parquet"
        METRICS_DF = pd.read_parquet(mpath)
    except Exception as e:
        print(f"  [metricas] {e}")

    try:
        ip = pd.read_parquet(ROOT / "data" / "processed" / "item_profiles.parquet")
        ITEM_CATS = ip.set_index("itemid")["category_id"].to_dict()
    except Exception as e:
        print(f"  [item_profiles] {e}")

    print(f"Modelos listos: {list(MODELS.keys())}")
    yield
    MODELS.clear()
    print("API apagada.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RetailRocket Recommender API",
    description="Sistema de recomendacion e-commerce — NexusDataCo | SoyHenry",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendedItem(BaseModel):
    rank:        int
    itemid:      int
    score:       float
    category_id: Optional[int] = None

class RecommendResponse(BaseModel):
    visitor_id:  int
    model_used:  str
    strategy:    str
    n:           int
    response_ms: float
    items:       list[RecommendedItem]

class HealthResponse(BaseModel):
    status:        str
    models_loaded: list[str]
    models_failed: list[str]

# ── Helpers ───────────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    "popularity":    "Popularidad",
    "item_cf":       "Item-CF",
    "als":           "BPR",
    "ease":          "EASE",
    "content_based": "Content-Based",
    "hybrid":        "Hibrido",
}

def get_model(name: str):
    m = MODELS.get(name)
    if m is None:
        raise HTTPException(status_code=503, detail=f"Modelo '{name}' no disponible.")
    return m

def recs_to_items(df: pd.DataFrame) -> list[RecommendedItem]:
    items = []
    for _, row in df.iterrows():
        items.append(RecommendedItem(
            rank=int(row["rank"]),
            itemid=int(row["itemid"]),
            score=round(float(row["score"]), 4),
            category_id=int(ITEM_CATS[row["itemid"]]) if row["itemid"] in ITEM_CATS else None,
        ))
    return items

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "name":      "RetailRocket Recommender API",
        "version":   "1.0.0",
        "team":      "NexusDataCo — SoyHenry Data Science",
        "models":    list(MODELS.keys()),
        "docs":      "/docs",
        "health":    "/v1/health",
        "recommend": "/v1/recommend/{visitor_id}",
        "metrics":   "/v1/metrics",
    }


@app.get("/v1/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    all_models = ["popularity", "item_cf", "als", "ease", "content_based", "hybrid"]
    loaded = [m for m in all_models if m in MODELS]
    failed = [m for m in all_models if m not in MODELS]
    return HealthResponse(
        status="ok" if len(loaded) >= 4 else "degraded",
        models_loaded=loaded,
        models_failed=failed,
    )


@app.get("/v1/models", tags=["Sistema"])
def list_models():
    return {
        "available": [
            {"id": name, "display": MODEL_DISPLAY.get(name, name), "loaded": name in MODELS}
            for name in ["popularity", "item_cf", "als", "ease", "content_based", "hybrid"]
        ]
    }


@app.get("/v1/recommend/popular", tags=["Recomendaciones"])
def recommend_popular(
    n: int = Query(default=10, ge=1, le=50),
    category_id: Optional[int] = Query(default=None),
):
    t0    = time.time()
    model = get_model("popularity")
    recs  = model.recommend(category_id=str(category_id) if category_id else None, n=n)
    if recs.empty:
        raise HTTPException(status_code=404, detail="Sin recomendaciones disponibles.")
    return {
        "model_used":  "popularity",
        "n":           len(recs),
        "response_ms": round((time.time() - t0) * 1000, 2),
        "items":       recs_to_items(recs),
    }


@app.get("/v1/recommend/{visitor_id}", response_model=RecommendResponse, tags=["Recomendaciones"])
def recommend(
    visitor_id: int,
    n:           int          = Query(default=10, ge=1, le=50),
    model:       str          = Query(default="hybrid"),
    category_id: Optional[int] = Query(default=None),
):
    """
    Genera Top-N recomendaciones personalizadas para un usuario.
    - **model**: popularity | item_cf | als | ease | content_based | hybrid
    """
    t0 = time.time()

    valid_models = ["popularity", "item_cf", "als", "ease", "content_based", "hybrid"]
    if model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Modelo '{model}' no valido.")

    m        = get_model(model)
    strategy = model
    recs     = pd.DataFrame()

    try:
        if model == "popularity":
            recs = m.recommend(
                category_id=str(category_id) if category_id else None, n=n
            )
            strategy = "popularity_by_category" if category_id else "popularity_global"

        elif model == "hybrid":
            recs = m.recommend(
                visitor_id=visitor_id, n=n,
                category_id=str(category_id) if category_id else None,
            )
            if not recs.empty and "strategy" in recs.columns:
                strategy = recs["strategy"].iloc[0]

        else:
            recs = m.recommend(visitor_id=visitor_id, n=n)
            strategy = model

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en modelo: {str(e)}")

    # Fallback a popularidad
    if recs.empty:
        pop = MODELS.get("popularity")
        if pop:
            recs     = pop.recommend(
                category_id=str(category_id) if category_id else None, n=n
            )
            strategy = "popularity_fallback"

    if recs.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Sin recomendaciones para visitor_id={visitor_id}."
        )

    return RecommendResponse(
        visitor_id=visitor_id,
        model_used=model,
        strategy=strategy,
        n=len(recs),
        response_ms=round((time.time() - t0) * 1000, 2),
        items=recs_to_items(recs),
    )


@app.get("/v1/metrics", tags=["Evaluacion"])
def get_metrics(k: int = Query(default=10, description="K: 5, 10 o 20")):
    if METRICS_DF is None:
        raise HTTPException(status_code=503, detail="Metricas no disponibles.")
    df = METRICS_DF[METRICS_DF["K"] == k].copy()
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No hay metricas para K={k}.")
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "model":     row["model"],
            "K":         int(row["K"]),
            "precision": round(float(row["Precision@K"]), 6),
            "recall":    round(float(row["Recall@K"]), 6),
            "ndcg":      round(float(row["NDCG@K"]), 6),
            "hitrate":   round(float(row["HitRate@K"]), 6),
            "coverage":  round(float(row["Coverage"]), 6),
            "novelty":   None if pd.isna(row.get("Novelty", float("nan")))
                         else round(float(row["Novelty"]), 4),
        })
    return {"k": k, "models": rows}
