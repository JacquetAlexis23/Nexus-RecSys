# 🛒 NexusDataCo: E-commerce Recommendation System

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![EDA](https://img.shields.io/badge/EDA-Senior%20Level-purple)
![Notebooks](https://img.shields.io/badge/Notebooks-3%20Senior%20EDAs-green)

Sistema de recomendación para e-commerce construido sobre el dataset de **Retailrocket**. El proyecto aborda los desafíos reales de **Cold-Start**, **Long-Tail** y **matrix sparsity** con un enfoque híbrido (Colaborativo + Contenido).

---

## 📊 Métricas Clave del Dataset

| Métrica | Valor | Implicación |
|:--------|:-----:|:------------|
| **Interacciones totales** | 2,756,101 | Alta escala de datos |
| **Usuarios únicos** | 1,407,580 | Muy disperso |
| **Ítems únicos** | 235,061 | Catálogo grande |
| **Sparsity de la matriz** | **99.9992%** | CF puro es inviable |
| **Cold-Start (1 interacción)** | **71.15%** usuarios | Necesario modelo de contenido |
| **Long-Tail (<5 interacciones)** | **61.31%** del catálogo | Diversidad > Popularidad |
| **Tasa de conversión** | **0.81%** | Oportunidad de mejora |
| **Rango de fechas** | May–Sep 2015 | ~5 meses de histórico |

---

## 📂 Arquitectura del Proyecto

```text
Proyecto Final/
│
├── 📁 data/
│   ├── raw/                     # ⚠️ Datos originales (inmutables)
│   │   ├── events.csv
│   │   ├── item_properties_part1.csv
│   │   ├── item_properties_part2.csv
│   │   ├── category_tree.csv
│   │   ├── item_features.csv
│   │   ├── user_features.csv
│   │   └── README.md
│   └── processed/               # ✅ Datos limpios (generados por src/)
│       ├── events_clean.csv
│       ├── category_tree_clean.csv
│       ├── item_categories_clean.csv
│       └── README.md
│
├── 📁 notebooks/                # 🔬 Análisis Exploratorio (nivel Senior)
│   ├── 01_EDA_Events_Senior.ipynb          # 11 gráficas
│   ├── 02_EDA_ItemProperties_Senior.ipynb  # 7 gráficas
│   ├── 03_EDA_CategoryTree_Senior.ipynb    # 6 gráficas
│   ├── EDA_Events.ipynb                    # (legacy – mantener referencia)
│   └── EDA_ItemProperties.ipynb            # (legacy – mantener referencia)
│
├── 📁 src/                      # 🐍 Módulos Python reutilizables
│   ├── __init__.py
│   ├── config.py                # ⚙️ Rutas y parámetros centralizados
│   ├── data_cleaning.py         # 🧹 ETL, carga y métricas clave
│   └── viz.py                   # 📈 Funciones de visualización reutilizables
│
├── 📁 models/                   # 🤖 Modelos serializados (pendiente)
├── 📁 reports/                  # 📄 Reportes de hallazgos
│   └── eda_findings.md
├── 📁 docs/                     # 📑 Documentación del negocio
│   ├── Propuesta_PF_NexusDataCo.pdf
│   └── Propuesta_PF_NexusDataCo.docx
│
├── README.md                    # ← Este archivo
├── requirements.txt             # 📦 Dependencias Python
└── .gitignore
```

---

## 🚀 Configuración del Entorno

```bash
# 1. Clonar o abrir el proyecto
cd "Proyecto Final"

# 2. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar Jupyter
jupyter notebook
```

---

## 🔬 Notebooks de Análisis (Senior EDA)

| Notebook | Dataset | Gráficas | Análisis Destacado |
|----------|---------|----------|--------------------|
| [`01_EDA_Events_Senior`](notebooks/01_EDA_Events_Senior.ipynb) | events | **11** | Lorenz Curve, Gini, Heatmap semanal, Cold-Start pie |
| [`02_EDA_ItemProperties_Senior`](notebooks/02_EDA_ItemProperties_Senior.ipynb) | item_properties | **7** | Cardinalidad, cobertura por ítem, actualización temporal |
| [`03_EDA_CategoryTree_Senior`](notebooks/03_EDA_CategoryTree_Senior.ipynb) | category_tree | **6** | BFS depth, distribución de hijos, barras apiladas por nivel |

**Total: 24 visualizaciones de nivel senior**

---

## 🛠️ Módulos Clave (`src/`)

```python
from src.config import EVENTS_CLEAN, EVENTS_RAW
from src.data_cleaning import load_and_clean_events, calculate_sparsity, conversion_funnel
from src.viz import plot_lorenz, plot_funnel, plot_time_heatmap
```

---

## 🗺️ Roadmap

- [x] Estructuración profesional del proyecto
- [x] EDA Senior con 24+ gráficas (3 datasets)
- [x] Módulos `src/` reutilizables y bien documentados
- [ ] Pipeline de limpieza automático
- [ ] Modelo Híbrido (Colaborativo + Basado en Contenido)
- [ ] Dashboard de Business Intelligence

---

**NexusDataCo** · *Data-Driven Intelligence for E-commerce* · 2025-2026
