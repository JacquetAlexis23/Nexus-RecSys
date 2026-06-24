"""
run_pipeline.py
Script principal del proyecto — ejecutar siempre desde la raíz:

    cd D:\\Work\\Cursos\\Data Science\\Proyecto Final
    python run_pipeline.py

Opciones:
    python run_pipeline.py --step loader          # solo carga y valida CSVs
    python run_pipeline.py --step cleaner         # limpieza y matriz de interacciones
    python run_pipeline.py --step features        # TF-IDF, features temporales
    python run_pipeline.py --step synthetic       # perfiles demográficos sintéticos
    python run_pipeline.py --step all             # pipeline completo (default)
    python run_pipeline.py --step check           # verifica instalación sin procesar datos
"""

import sys
import argparse
from pathlib import Path

# ── Garantizar que la raíz del proyecto esté en el path ──────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger


def check_installation():
    """Verifica que todas las dependencias y archivos estén en orden."""
    logger.info("Verificando instalación...")

    # Dependencias críticas
    missing_deps = []
    for pkg, import_name in [
        ("pandas",      "pandas"),
        ("numpy",       "numpy"),
        ("loguru",      "loguru"),
        ("sklearn",     "sklearn"),
        ("scipy",       "scipy"),
        ("pyarrow",     "pyarrow"),
        ("faker",       "faker"),
        ("tqdm",        "tqdm"),
        ("joblib",      "joblib"),
    ]:
        try:
            __import__(import_name)
            logger.info(f"  ✓ {pkg}")
        except ImportError:
            logger.error(f"  ✗ {pkg} — instalar con: pip install {pkg}")
            missing_deps.append(pkg)

    if missing_deps:
        logger.error(f"\nFaltan {len(missing_deps)} dependencias. Ejecuta:")
        logger.error(f"  pip install -r requirements.txt")
        return False

    # Archivos de datos
    from config.config import EVENTS_FILE, ITEM_PROPS_1, ITEM_PROPS_2, CATEGORY_TREE
    missing_files = []
    for name, path in [
        ("events.csv",                 EVENTS_FILE),
        ("item_properties_part1.csv",  ITEM_PROPS_1),
        ("item_properties_part2.csv",  ITEM_PROPS_2),
        ("category_tree.csv",          CATEGORY_TREE),
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1_048_576
            logger.info(f"  ✓ {name} ({size_mb:.1f} MB)")
        else:
            logger.error(f"  ✗ {name} — no encontrado en {path}")
            missing_files.append(name)

    if missing_files:
        logger.error(
            f"\nFaltan {len(missing_files)} archivo(s) de datos. "
            f"Colócalos en: {EVENTS_FILE.parent}"
        )
        return False

    logger.info("\n✅ Todo listo para ejecutar el pipeline.")
    return True


def run_loader():
    from src.data.loader import load_all
    data = load_all()
    logger.info(f"\nResumen de carga:")
    for name, df in data.items():
        logger.info(f"  {name}: {df.shape}")
    return data


def run_cleaner(data=None):
    if data is None:
        from src.data.loader import load_events, load_item_properties
        logger.info("Cargando datos para limpieza...")
        events_raw     = load_events()
        item_props_raw = load_item_properties()
    else:
        events_raw     = data["events"]
        item_props_raw = data["item_properties"]

    from src.data.cleaner import run_cleaning_pipeline
    artifacts = run_cleaning_pipeline(events_raw, item_props_raw, save=True)
    return artifacts


def run_features(artifacts=None):
    from config.config import PROC_DIR
    import pandas as pd

    if artifacts is None:
        logger.info("Cargando artefactos limpios desde disco...")
        events_clean     = pd.read_parquet(PROC_DIR / "events_clean.parquet")
        item_props_clean = pd.read_parquet(PROC_DIR / "item_props_clean.parquet")
    else:
        events_clean     = artifacts["events_clean"]
        item_props_clean = artifacts["item_props_clean"]

    from src.data.feature_engineering import run_feature_engineering
    features = run_feature_engineering(events_clean, item_props_clean)
    return features


def run_synthetic(artifacts=None):
    from config.config import PROC_DIR
    import pandas as pd

    if artifacts is None:
        logger.info("Cargando eventos limpios para perfiles sintéticos...")
        events_clean = pd.read_parquet(PROC_DIR / "events_clean.parquet")
    else:
        events_clean = artifacts.get("events_clean")

    from src.data.synthetic_profiles import generate_user_profiles
    profiles = generate_user_profiles(events_clean)
    logger.info(f"\nPerfil de segmentos generados:\n{profiles['segment'].value_counts()}")
    return profiles


def main():
    parser = argparse.ArgumentParser(description="RetailRocket Recommender — Pipeline ETL")
    parser.add_argument(
        "--step",
        choices=["check", "loader", "cleaner", "features", "synthetic", "all"],
        default="all",
        help="Paso del pipeline a ejecutar (default: all)"
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RETAILROCKET RECOMMENDER SYSTEM — Pipeline ETL")
    logger.info(f"Paso: {args.step.upper()}")
    logger.info("=" * 60)

    if args.step == "check":
        check_installation()
        return

    if args.step == "loader":
        run_loader()

    elif args.step == "cleaner":
        run_cleaner()

    elif args.step == "features":
        run_features()

    elif args.step == "synthetic":
        run_synthetic()

    elif args.step == "all":
        # Pipeline completo encadenado (más eficiente: no relee del disco)
        logger.info("\n[1/4] Cargando datos...")
        data = run_loader()

        logger.info("\n[2/4] Limpiando datos...")
        artifacts = run_cleaner(data)

        logger.info("\n[3/4] Construyendo features...")
        features = run_features(artifacts)

        logger.info("\n[4/4] Generando perfiles sintéticos...")
        run_synthetic(artifacts)

        logger.info("\n" + "=" * 60)
        logger.info("✅ Pipeline ETL completado.")
        logger.info("Artefactos generados en data/processed/ y data/synthetic/")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
