# 🛒 RetailRocket Recommender System

Sistema de recomendación personalizado para e-commerce, desarrollado como Proyecto Final de SoyHenry.

## Problema de Negocio

Un retailer online enfrenta una **tasa de conversión del 0.8%** debido a una experiencia de navegación genérica que no considera el historial ni las preferencias individuales de cada usuario.

**Solución:** Motor de recomendación con feedback implícito que personaliza la experiencia usando señales de comportamiento (views, addtocart, transactions).

---

## KPIs

| KPI | Tipo | Descripción |
|-----|------|-------------|
| Tasa de conversión del sistema | Principal | % usuarios que compran a partir de un ítem recomendado |
| Cobertura del catálogo | Secundario | % productos únicos recomendados al menos 1 vez |

---

## Dataset

**Fuente:** [RetailRocket E-Commerce Dataset — Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

| Archivo | Descripción |
|---------|-------------|
| `events.csv` | 2.75M eventos de comportamiento (view / addtocart / transaction) |
| `item_properties_part1/2.csv` | Propiedades de ítems (valores hasheados) |
| `category_tree.csv` | Jerarquía de categorías del catálogo |

### Configurar dataset
```bash
# Descargar de Kaggle y colocar en:
data/raw/events.csv
data/raw/item_properties_part1.csv
data/raw/item_properties_part2.csv
data/raw/category_tree.csv
```

---

## Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd recommender-system

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Estructura del Proyecto

```
recommender-system/
├── config/
│   └── config.py              # Configuración central (parámetros, rutas)
├── data/
│   ├── raw/                   # Archivos originales de Kaggle
│   ├── processed/             # Datos limpios (parquet)
│   └── synthetic/             # Perfiles demográficos generados
├── src/
│   ├── data/
│   │   ├── loader.py          # Carga de archivos fuente
│   │   ├── cleaner.py         # Pipeline de limpieza
│   │   ├── feature_engineering.py  # Features para modelos
│   │   └── synthetic_profiles.py   # Datos demográficos con Faker
│   ├── models/
│   │   ├── popularity.py      # Baseline
│   │   ├── item_cf.py         # Collaborative Filtering ítem-ítem
│   │   ├── als_model.py       # Matrix Factorization (ALS)
│   │   ├── content_based.py   # Content-Based (TF-IDF)
│   │   └── hybrid.py          # Modelo híbrido
│   ├── evaluation/
│   │   └── metrics.py         # Precision@K, Recall@K, NDCG@K, Coverage
│   └── api/
│       └── main.py            # FastAPI REST endpoint
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_baseline_itemcf.ipynb
│   ├── 03_als_contentbased.ipynb
│   └── 04_hybrid_evaluation.ipynb
├── dashboard/
│   └── app.py                 # Streamlit Dashboard
├── models_saved/              # Modelos serializados (.pkl)
├── requirements.txt
└── README.md
```

---

## Modelos

| Modelo | Sprint | Descripción |
|--------|--------|-------------|
| Popularidad Global | 2 | Baseline — top ítems más interactuados |
| Item-CF | 2 | Collaborative Filtering ítem-ítem |
| ALS | 3 | Matrix Factorization para feedback implícito |
| Content-Based | 3 | TF-IDF sobre propiedades de ítems |
| Híbrido | 3 | Ensamble CF (60%) + CBF (40%) |

### Pesos de feedback implícito

```python
EVENT_WEIGHTS = {
    "view":        1,
    "addtocart":   4,
    "transaction": 10,
}
```

---

## Evaluación

**Split:** Temporal — últimos 30 días como test (evita data leakage).

**Métricas:**
- `Precision@K` — Relevancia de las recomendaciones
- `Recall@K` — Cobertura de preferencias del usuario
- `NDCG@K` — Calidad del ranking
- `Coverage` — KPI secundario del proyecto
- `Novelty` — Anti-sesgo de popularidad

---

## Ejecución del Pipeline

```bash
# Sprint 1: ETL + EDA
python -m src.data.loader
python -m src.data.cleaner
python -m src.data.feature_engineering
python -m src.data.synthetic_profiles

# Sprint 2: Modelos baseline
python -m src.models.popularity
python -m src.models.item_cf

# Sprint 3: Modelos avanzados
python -m src.models.als_model
python -m src.models.content_based
python -m src.models.hybrid

# Sprint 4: API y Dashboard
uvicorn src.api.main:app --reload
streamlit run dashboard/app.py
```

---

## API REST

```bash
# Iniciar servidor
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Endpoint principal
GET /v1/recommend/{visitor_id}?n=10&model=hybrid

# Response
{
  "visitor_id": 1234567,
  "user_type": "warm",
  "model_used": "hybrid",
  "recommendations": [
    {"rank": 1, "itemid": 456789, "score": 0.87, "category_id": "1234"},
    ...
  ]
}
```

---

## Equipo

Proyecto Final — SoyHenry Data Science Bootcamp  
Dataset: RetailRocket (Kaggle) | Período: 4.5 meses de comportamiento real
