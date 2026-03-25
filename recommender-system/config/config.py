"""
config.py
Configuración central del proyecto — RetailRocket Recommender System
Todos los parámetros ajustables en un solo lugar.
"""

from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
# Siempre apunta a la raíz del proyecto, sin importar desde dónde se ejecute
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
SYNTH_DIR  = DATA_DIR / "synthetic"
MODELS_DIR = ROOT_DIR / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)

# Archivos fuente (colocar los CSV de Kaggle en data/raw/)
EVENTS_FILE      = RAW_DIR / "events.csv"
ITEM_PROPS_1     = RAW_DIR / "item_properties_part1.csv"
ITEM_PROPS_2     = RAW_DIR / "item_properties_part2.csv"
CATEGORY_TREE    = RAW_DIR / "category_tree.csv"

# Archivos procesados
EVENTS_CLEAN     = PROC_DIR / "events_clean.parquet"
ITEM_PROFILES    = PROC_DIR / "item_profiles.parquet"
INTERACTION_MTX  = PROC_DIR / "interaction_matrix.parquet"
USER_PROFILES    = SYNTH_DIR / "user_profiles.parquet"

# ── Pesos de Interacción (feedback implícito) ─────────────────────────────────
EVENT_WEIGHTS = {
    "view":        1,
    "addtocart":   4,
    "transaction": 10,
}

# ── Filtros de Calidad ────────────────────────────────────────────────────────
MIN_USER_INTERACTIONS = 3    # usuarios con menos interacciones → cold-start
MIN_ITEM_INTERACTIONS = 5    # ítems con menos interacciones → excluir de CF
SESSION_GAP_MINUTES   = 30   # inactividad > 30 min → nueva sesión

# ── Modelos ───────────────────────────────────────────────────────────────────
TOP_N = 10   # número de recomendaciones por defecto

# ALS (implicit)
ALS_FACTORS      = 64
ALS_ITERATIONS   = 20
ALS_REGULARIZATION = 0.1
ALS_ALPHA        = 40    # escala de confianza para feedback implícito

# Item-CF
ITEM_CF_NEIGHBORS = 50   # vecinos más cercanos por ítem

# Content-Based
CBF_MAX_FEATURES  = 500  # vocabulario TF-IDF sobre hashes de propiedades
CBF_TOP_SIMILAR   = 50   # ítems similares a pre-computar por ítem

# Híbrido (pesos del ensamble)
HYBRID_WEIGHT_CF  = 0.6
HYBRID_WEIGHT_CBF = 0.4

# ── Evaluación ────────────────────────────────────────────────────────────────
EVAL_K          = [5, 10, 20]      # valores de K para Precision/Recall/NDCG
TEST_DAYS       = 30               # últimos N días como conjunto de test
RANDOM_SEED     = 42

# ── Datos Sintéticos ──────────────────────────────────────────────────────────
SYNTH_N_USERS   = 50_000          # usuarios sintéticos a generar
SYNTH_REGIONS   = [               # regiones de Colombia (contexto del proyecto)
    "Bogotá", "Medellín", "Cali", "Barranquilla",
    "Cartagena", "Bucaramanga", "Pereira", "Manizales",
]
SYNTH_SEGMENTS  = ["occasional", "regular", "vip"]

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_VERSION = "v1"
