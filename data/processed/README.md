# 📁 data/processed — Datos Limpios y Preprocesados

> Esta carpeta contiene los archivos resultantes del pipeline de limpieza. 
> Los scripts en `src/data_cleaning.py` son los que generan estos archivos.

## Archivos presentes

| Archivo | Descripción | Generado por |
|---------|-------------|--------------|
| `events_clean.csv` | Events con timestamps convertidos y duplicados removidos | `load_and_clean_events()` |
| `category_tree_clean.csv` | Árbol de categorías limpio | `load_category_tree()` |
| `item_categories_clean.csv` | Categorías por ítem, con nulos removidos | Pipeline de limpieza |

## Cómo regenerar

```python
from src.data_cleaning import load_and_clean_events
from src.config import EVENTS_RAW, EVENTS_CLEAN

df = load_and_clean_events(EVENTS_RAW)
df.to_csv(EVENTS_CLEAN, index=False)
```

---
*NexusDataCo – Proyecto Final Data Science*
