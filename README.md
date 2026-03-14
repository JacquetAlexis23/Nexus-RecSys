# Nexus RecSys

**Sistema de Recomendación de E-Commerce sobre el dataset público Retailrocket**

> Estado del proyecto: **En desarrollo** · Fase actual: Modelado completado · Próximo paso: Deployment/API

---

## Descripción

Nexus RecSys es un pipeline end-to-end de ciencia de datos que construye un **sistema de recomendación** a partir del dataset público de comportamiento de usuarios de [Retailrocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). El proyecto implementa desde la exploración inicial hasta la generación del feature set listo para modelado, siguiendo buenas prácticas de reproducibilidad, trazabilidad y separación de responsabilidades entre etapas.

### Objetivo general

Producir un sistema capaz de recomendar productos a usuarios, combinando señales de comportamiento implícito (vistas, carritos, compras) con features de item y contexto de usuario, resolviendo además el problema de **cold-start** que afecta a la mayoría de los visitantes del catálogo.

### Dataset fuente

| Archivo | Descripción | Registros |
|---|---|---|
| `events.csv` | Log de interacciones usuario-ítem (`view`, `addtocart`, `transaction`) | ~2.75 M |
| `item_properties_part1/2.csv` | Snapshot-log de atributos de ítems (precio, categoría, disponibilidad) | ~20 M |
| `category_tree.csv` | Jerarquía padre-hijo de categorías del catálogo | ~1.6 K |

---

## Estructura del repositorio

```
nexus-recsys/
├── data/
│   ├── raw/                    ← CSVs originales — INMUTABLES, no modificar
│   ├── interim/                ← Checkpoints .parquet entre notebooks
│   └── processed/              ← Outputs finales listos para modelado
├── notebooks/                  ← Pipeline de análisis y feature engineering
│   ├── 01_eda_events.ipynb
│   ├── 02_eda_items_categories.ipynb
│   ├── 03_funnel_analysis.ipynb
│   ├── 04_merge_pipeline.ipynb
│   ├── 05_synthetic_demographics.ipynb
│   ├── 06_feature_engineering.ipynb
│   └── 07_modeling.ipynb           ← ★ NUEVO: Modelado completo
├── docs/                           ← Documentación técnica del modelado
│   ├── model_justification.md          ← Justificación del modelo final (versionado)
│   ├── model_comparison_final.csv      ← Generado por NB07 (no versionado)
│   ├── fig_dataset_stats.png           ← Generado por NB07 (no versionado)
│   └── fig_model_comparison.png        ← Generado por NB07 (no versionado)
├── scripts/                        ← Scripts de generación de notebooks
│   └── generate_modeling_notebook.py   ← Fuente de verdad de NB07
├── encoders/                   ← Scalers, encoders y modelo final (.pkl)
├── requirements.txt
└── README.md
```

---

## Pipeline de notebooks

El pipeline está organizado en **7 notebooks numerados** que deben ejecutarse en orden secuencial. Cada notebook lee el checkpoint del anterior y produce el suyo propio.

### 01 · EDA de Eventos (`events.csv`)

**Entrada:** `data/raw/events.csv`  
**Salida:** `data/interim/cp01_events_clean.parquet`

Exploración y limpieza del log central de comportamiento. Cubre validación de schema, análisis de distribuciones de eventos, segmentación de usuarios por nivel de actividad, análisis de popularidad de ítems y patrones temporales (heatmap hora×día, series diarias). Produce el artefacto base del pipeline.

**Hallazgos clave:**
- Tasa de conversión global view → compra: ~0.7% (embudo muy estrecho, típico de e-commerce)
- Más del 50% de los visitantes tienen ≤ 2 eventos registrados → problema estructural de cold-start
- Distribución de actividad por ítem sigue una ley de potencias pronunciada → riesgo de sesgo de popularidad
- Pico de actividad entre las 10 h y las 19 h, con mayor volumen de lunes a viernes

---

### 02 · EDA de Ítems & Categorías

**Entrada:** `data/raw/item_properties_part1/2.csv`, `data/raw/category_tree.csv`  
**Salida:** `data/interim/cp02_items_flat.parquet`, `data/interim/cp02_category_enriched.parquet`

El dataset de propiedades viene en formato **snapshot-log vertical** (una fila por atributo×instante). Este notebook lo transforma en una tabla **wide** (una fila por ítem) tomando el valor más reciente de cada propiedad. Paralelamente, resuelve la jerarquía del árbol de categorías de forma iterativa para calcular la profundidad y el ancestro raíz de cada categoría.

---

### 03 · Análisis del Funnel de Conversión

**Entrada:** `data/interim/cp01_events_clean.parquet`  
**Salida:** `data/interim/cp03_funnel_metrics.parquet`

Análisis en profundidad del embudo `view → addtocart → transaction` a dos niveles de granularidad: usuarios únicos (visión macro) y pares (visitorid, itemid) (visión micro, relevante para el modelo). Cuantifica anomalías estructurales (transacciones sin view o carrito previo), calcula tiempos entre etapas, y construye el DataFrame de métricas de comportamiento por visitante que se usará como features en el modelo.

---

### 04 · Merge Pipeline — Dataset Integrado

**Entrada:** `cp01`, `cp02_items_flat`, `cp02_category_enriched`, `cp03_funnel_metrics`  
**Salida:** `data/interim/cp04_merged.parquet`

Integración progresiva de todas las fuentes mediante **left joins validados** (assert de integridad tras cada join). Combina el log de eventos con propiedades de producto, jerarquía de categorías y métricas de comportamiento por usuario en una única tabla analítica que preserva la granularidad original del log.

---

### 05 · Datos Demográficos Sintéticos

**Entrada:** `cp04_merged.parquet`, `cp03_funnel_metrics.parquet`  
**Salida:** `data/interim/cp05_with_demographics.parquet`

El dataset Retailrocket no incluye información personal de los usuarios. Para simular un entorno productivo real, se generan perfiles demográficos sintéticos con distribuciones basadas en benchmarks de e-commerce LATAM: `age` (normal truncada), `gender`, `country`, `region`, `customer_segment` (derivado del comportamiento real), y `registration_days_ago`. Los datos son puramente ficticios y se usan exclusivamente como features de contexto.

---

### 06 · Feature Engineering — Feature Set Final

**Entrada:** `data/interim/cp05_with_demographics.parquet`  
**Salida:** `data/processed/` + `encoders/`

Último notebook pre-modelado. Construye el feature set completo organizado en tres granularidades:

| Artefacto | Granularidad | Descripción |
|---|---|---|
| `user_features.csv` | 1 fila / visitante | Comportamiento + demografía + encoded + scaled |
| `item_features.csv` | 1 fila / ítem | Popularidad + conversión + scaled |
| `interaction_matrix.csv` | 1 fila / par user×item | `interaction_strength`, timestamps de primera/última interacción |
| `train_test_split_info.json` | — | Fecha de corte, conteos y porcentajes de train/test |

Adicionalmente aplica LabelEncoding y One-Hot Encoding, normalización con `StandardScaler` (serializado en `encoders/`) y **split temporal** en el percentil 80 de fechas para evitar data leakage.

---

### 07 · Modelado — Sistema de Recomendación Completo ★

**Entrada:** `data/processed/` (5 artefactos)  
**Salida:** `encoders/final_model.pkl`, `encoders/hybrid_model.pkl`, `encoders/lgb_model_opt.txt`, `docs/model_comparison_final.csv`, `docs/fig_model_comparison.png`

> ⚠️ El notebook `07_modeling.ipynb` es **generado automáticamente** por `scripts/generate_modeling_notebook.py`.
> Si necesitas modificar el pipeline de modelado, edita el script y regénera:
> ```bash
> python scripts/generate_modeling_notebook.py
> ```

Pipeline completo de modelado que implementa y compara **9 modelos** organizados en 4 familias:

| Modelo | Tipo | Librería |
|--------|------|---------|
| Popularity Baseline | Regla heurística | pandas |
| SVD (k=50) | Factorización de Matrices | scipy.sparse |
| NMF (k=50) | Factorización No-Negativa | scikit-learn |
| LightGBM LTR | Learning-to-Rank pointwise | lightgbm |
| Item-CF (SVD emb.) | Item-Based CF por similitud coseno | scipy + sklearn |
| Content-Based Filtering | Perfil de usuario por features de ítem | sklearn |
| SVD Optimizado (Optuna) | SVD + confidence weighting | scipy + optuna |
| LightGBM Optimizado (Optuna) | LTR con búsqueda de hiperparámetros | lightgbm + optuna |
| **Híbrido SVD Opt + CBF ★** | α·SVD + (1-α)·CB, α calibrado en validación | — |

**Métricas implementadas:** `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Coverage`, `Novelty` (K = 5, 10)

**Modelo ganador:** Híbrido (α=0.5) — Coverage 10× superior al SVD puro (0.040 vs 0.004), Novelty más alta (16.69), y el único que mitiga cold-start de usuario mediante el componente CBF.

**Estadísticas clave del dataset:**

| Métrica | Valor |
|---------|-------|
| Usuarios únicos | 1 407 580 |
| Ítems únicos | 235 061 |
| Interacciones totales | 2 145 179 |
| Sparsity | 99.9994 % |
| Split temporal (cutoff) | 2015-08-22 |

Ver justificación detallada en [`docs/model_justification.md`](docs/model_justification.md).

---

## Mapa de checkpoints

```
data/raw/
├── events.csv
├── item_properties_part1.csv
├── item_properties_part2.csv
└── category_tree.csv
        │
        ▼  (NB 01)
data/interim/cp01_events_clean.parquet       ~2.75 M filas
        │
        ▼  (NB 02)
data/interim/cp02_items_flat.parquet         ~417 K ítems
data/interim/cp02_category_enriched.parquet  ~1.6 K categorías
        │
        ▼  (NB 03)
data/interim/cp03_funnel_metrics.parquet     1 fila / visitante
        │
        ▼  (NB 04)
data/interim/cp04_merged.parquet             ~2.75 M filas × N cols
        │
        ▼  (NB 05)
data/interim/cp05_with_demographics.parquet  + perfil demográfico
        │
        ▼  (NB 06)
data/processed/user_features.csv
data/processed/item_features.csv
data/processed/interaction_matrix.csv
data/processed/train_test_split_info.json
encoders/scaler_user.pkl
encoders/scaler_item.pkl
encoders/label_encoders.pkl
        │
        ▼  (NB 07)  ★ Modelado
encoders/final_model.pkl         ← Híbrido SVD Opt + CB (modelo ganador)
encoders/hybrid_model.pkl        ← Artefacto del Híbrido (α, params, metrics)
encoders/lgb_model_opt.txt       ← LightGBM (re-ranker opcional)
docs/model_comparison_final.csv  ← Tabla comparativa de los 9 modelos
docs/fig_model_comparison.png    ← Gráfico comparativo
```

---

## Instalación y ejecución

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd nexus-recsys

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar los notebooks en orden (01 → 07)
# Desde la raíz del repositorio, abrir Jupyter y ejecutar en secuencia
jupyter notebook notebooks/
```

> Los archivos `data/raw/*.csv` deben descargarse del [dataset Retailrocket en Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) y colocarse en `data/raw/` antes de ejecutar el pipeline.

---

## Convenciones del proyecto

| Convención | Aplicación |
|---|---|
| `snake_case` | Variables, funciones, nombres de columnas |
| Idioma | Comentarios y markdown en **español** |
| `random_state=42` | Toda operación estocástica |
| `pathlib.Path` | Todas las rutas de archivo |
| `logging` | Trazabilidad en lugar de `print` |
| `.parquet` | Formato de checkpoints (preserva dtypes, ~5× más compacto que CSV) |
| `assert len(df) == n_before` | Validación de integridad tras cada join |

---

## Estado del proyecto

```
✅ EDA de eventos           (NB 01)
✅ EDA de ítems y categorías (NB 02)
✅ Análisis de funnel        (NB 03)
✅ Merge pipeline            (NB 04)
✅ Datos demográficos        (NB 05)
✅ Feature engineering       (NB 06)
✅ Modelado completo         (NB 07) ← 9 modelos (CF + CBF + Híbrido) + Optuna + comparativa
⬜ API / Deployment          (pendiente)
```

---

## Tecnologías utilizadas

| Librería | Uso |
|---|---|
| `pandas` ≥ 2.0 | Manipulación y análisis de datos |
| `numpy` ≥ 1.26 | Operaciones numéricas |
| `matplotlib` / `seaborn` | Visualizaciones |
| `scikit-learn` ≥ 1.4 | Encoders, scalers, splitting, NMF |
| `scipy` | SVD truncada para factorización de matrices |
| `lightgbm` | Modelo Learning-to-Rank |
| `optuna` | Optimización bayesiana de hiperparámetros |
| `pyarrow` ≥ 15.0 | Serialización Parquet |
| `faker` | Generación de datos demográficos sintéticos |
| `nbformat` / `nbconvert` | Validación y ejecución de notebooks |
