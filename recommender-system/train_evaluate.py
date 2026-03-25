"""
train_evaluate.py — Sprint 2, 3 y EASE
"""
import sys, argparse
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import PROC_DIR, MODELS_DIR, EVAL_K, TEST_DAYS, TOP_N, MIN_ITEM_INTERACTIONS
from src.evaluation.metrics import evaluate_model, compare_models, temporal_train_test_split
from src.models.popularity    import PopularityRecommender
from src.models.item_cf       import ItemCFRecommender
from src.models.als_model     import ALSRecommender
from src.models.ease_model    import EASERecommender
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid        import HybridRecommender

def load_artifacts():
    logger.info("Cargando artefactos...")
    ev = pd.read_parquet(PROC_DIR / "events_clean.parquet")
    im = pd.read_parquet(PROC_DIR / "interaction_matrix.parquet")
    ip = pd.read_parquet(PROC_DIR / "item_profiles.parquet")
    tf = sp.load_npz(PROC_DIR / "tfidf_matrix.npz")
    logger.info(f"  Eventos: {len(ev):,} | Matriz: {len(im):,} | Items: {len(ip):,} | TF-IDF: {tf.shape}")
    return {"events_clean": ev, "interaction_mtx": im, "item_profiles": ip, "tfidf_matrix": tf}

def build_train_matrix(events_clean):
    train_events, _ = temporal_train_test_split(events_clean, test_days=TEST_DAYS)
    tm = (train_events[train_events["user_type"] == "warm"]
          .groupby(["visitorid","itemid"])["event_weight"].sum()
          .reset_index().rename(columns={"event_weight":"score"}))
    ic = tm.groupby("itemid")["score"].sum()
    tm = tm[tm["itemid"].isin(ic[ic >= MIN_ITEM_INTERACTIONS].index)]
    logger.info(f"  Train matrix: {tm['visitorid'].nunique():,} usuarios x {tm['itemid'].nunique():,} items")
    return tm

def train_popularity(a):
    logger.info("\n" + "━"*60 + "\nENTRENANDO: Popularity Baseline\n" + "━"*60)
    m = PopularityRecommender(top_n=TOP_N)
    m.fit(a["events_clean"], a["item_profiles"]); m.save(); return m

def train_item_cf(a):
    logger.info("\n" + "━"*60 + "\nENTRENANDO: Item-CF\n" + "━"*60)
    m = ItemCFRecommender(top_n=TOP_N, n_neighbors=50)
    m.fit(build_train_matrix(a["events_clean"])); m.save(); return m

def train_als(a):
    logger.info("\n" + "━"*60 + "\nENTRENANDO: ALS/BPR\n" + "━"*60)
    m = ALSRecommender(n_factors=64, n_epochs=20)
    m.fit(build_train_matrix(a["events_clean"])); m.save(); return m

def train_content_based(a):
    logger.info("\n" + "━"*60 + "\nENTRENANDO: Content-Based\n" + "━"*60)
    m = ContentBasedRecommender(top_n=TOP_N)
    m.fit(a["tfidf_matrix"], a["item_profiles"], a["interaction_mtx"]); m.save(); return m

def train_ease(a):
    logger.info("\n" + "━"*60 + "\nENTRENANDO: EASE\n" + "━"*60)
    m = EASERecommender(lambda_reg=500, top_n=TOP_N)
    m.fit(build_train_matrix(a["events_clean"])); m.save(); return m

def train_hybrid(als_m, cf_m, cbf_m, pop_m):
    logger.info("\n" + "━"*60 + "\nCONFIGURANDO: Hibrido\n" + "━"*60)
    m = HybridRecommender(weight_als=0.6, weight_cf=0.4, top_n=TOP_N)
    m.fit(als_m, cf_m, cbf_m, pop_m); m.save(); return m

def evaluate_all(models, artifacts, sample_size=5000):
    logger.info("\n" + "━"*60 + "\nEVALUACION DE MODELOS\n" + "━"*60)
    _, test_events = temporal_train_test_split(artifacts["events_clean"], test_days=TEST_DAYS)
    test_purchases = test_events[test_events["event"].isin(["addtocart","transaction"])][["visitorid","itemid"]].drop_duplicates()
    catalog_size   = artifacts["item_profiles"]["itemid"].nunique()
    pop_scores     = artifacts["events_clean"].groupby("itemid")["event_weight"].sum()
    pop_model      = models.get("popularity")
    logger.info(f"  Test: {test_purchases['visitorid'].nunique():,} usuarios | {len(test_purchases):,} pares")

    def make_fn(model):
        def fn(uid, k):
            r = model.recommend(uid, n=k)
            if r.empty and pop_model: r = pop_model.recommend(n=k)
            return r["itemid"].tolist() if not r.empty else []
        return fn

    pop_fn = lambda uid, k: pop_model.recommend(n=k)["itemid"].tolist() if pop_model else []

    all_metrics = {}
    for name, model in models.items():
        logger.info(f"\n  Evaluando: {name}...")
        fn = pop_fn if name == "popularity" else make_fn(model)
        all_metrics[name] = evaluate_model(fn, test_purchases, catalog_size, EVAL_K, pop_scores, sample_size)

    comparison = compare_models(all_metrics, k=10)
    dfs = [df.assign(model=n) for n, df in all_metrics.items()]
    pd.concat(dfs, ignore_index=True).to_parquet(PROC_DIR / "metrics_all_models.parquet", index=False)
    logger.info(f"  Metricas guardadas en: {PROC_DIR / 'metrics_all_models.parquet'}")
    return comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["popularity","item_cf","als","ease","content_based","hybrid","sprint2","sprint3","all"],
        default="all"
    )
    parser.add_argument("--eval-sample", type=int, default=5000)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    logger.info("="*60 + f"\nRETAILROCKET — {args.model.upper()}\n" + "="*60)
    a = load_artifacts()
    models = {}

    if args.model in ("popularity","sprint2","all"):    models["popularity"]    = train_popularity(a)
    if args.model in ("item_cf","sprint2","all"):        models["item_cf"]       = train_item_cf(a)
    if args.model in ("als","sprint3","all"):            models["als"]           = train_als(a)
    if args.model in ("ease","sprint3","all"):           models["ease"]          = train_ease(a)
    if args.model in ("content_based","sprint3","all"):  models["content_based"] = train_content_based(a)
    if args.model in ("hybrid","sprint3","all"):
        als_m = models.get("als")           or ALSRecommender.load()
        cf_m  = models.get("item_cf")       or ItemCFRecommender.load()
        cbf_m = models.get("content_based") or ContentBasedRecommender.load()
        pop_m = models.get("popularity")    or PopularityRecommender.load()
        models["hybrid"] = train_hybrid(als_m, cf_m, cbf_m, pop_m)

    if not args.skip_eval and models:
        comp = evaluate_all(models, a, args.eval_sample)
        logger.info("\n" + "="*60 + "\nCOMPARATIVA FINAL @ K=10\n" + "="*60)
        logger.info(f"\n{comp.to_string(index=False)}")

    logger.info("\n✅ Completado.")

if __name__ == "__main__":
    main()
