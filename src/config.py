"""
config.py
=========
Configuración centralizada del proyecto NexusDataCo.
Modifica este archivo para ajustar rutas y parámetros globales.
"""

import os

# ──────────────────────────────────────────────────────────
# RUTAS
# ──────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
NOTEBOOKS_DIR  = os.path.join(BASE_DIR, "notebooks")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")
MODELS_DIR     = os.path.join(BASE_DIR, "models")

# ──────────────────────────────────────────────────────────
# ARCHIVOS RAW
# ──────────────────────────────────────────────────────────
EVENTS_RAW         = os.path.join(DATA_RAW, "events.csv")
ITEM_PROPS_PART1   = os.path.join(DATA_RAW, "item_properties_part1.csv")
ITEM_PROPS_PART2   = os.path.join(DATA_RAW, "item_properties_part2.csv")
ITEM_FEATURES_RAW  = os.path.join(DATA_RAW, "item_features.csv")
USER_FEATURES_RAW  = os.path.join(DATA_RAW, "user_features.csv")
CATEGORY_TREE_RAW  = os.path.join(DATA_RAW, "category_tree.csv")

# ──────────────────────────────────────────────────────────
# ARCHIVOS PROCESADOS
# ──────────────────────────────────────────────────────────
EVENTS_CLEAN        = os.path.join(DATA_PROCESSED, "events_clean.csv")
CATEGORY_TREE_CLEAN = os.path.join(DATA_PROCESSED, "category_tree_clean.csv")
ITEM_CAT_CLEAN      = os.path.join(DATA_PROCESSED, "item_categories_clean.csv")

# ──────────────────────────────────────────────────────────
# PARÁMETROS DEL MODELO
# ──────────────────────────────────────────────────────────
COLD_START_THRESHOLD = 5       # Interacciones mínimas para considerar usuario "activo"
LONG_TAIL_THRESHOLD  = 5       # Interacciones mínimas para ítem "popular"
RANDOM_SEED          = 42
TEST_SIZE            = 0.2

# ──────────────────────────────────────────────────────────
# VISUALIZACIÓN
# ──────────────────────────────────────────────────────────
PALETTE_PRIMARY   = "viridis"
PALETTE_SECONDARY = "magma"
FIGURE_DPI        = 100
FIGURE_SIZE_WIDE  = (14, 6)
FIGURE_SIZE_SQUARE= (8, 8)
