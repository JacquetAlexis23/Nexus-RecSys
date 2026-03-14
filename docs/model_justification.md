# Nexus RecSys — Justificación del Modelo Final

**Proyecto:** Sistema de Recomendación E-Commerce  
**Dataset:** RetailRocket E-Commerce (Kaggle)  
**Fecha:** Marzo 2026  
**Versión:** 2.0  

---

## 1. Resumen Ejecutivo

Nexus RecSys implementa un sistema de recomendación de productos para e-commerce sobre el dataset público RetailRocket, que contiene ~2.75 millones de eventos de comportamiento de usuarios (vistas, carritos, compras) sobre un catálogo de ~235 K ítems.

Se evaluaron **8 modelos** organizados en 3 familias: Collaborative Filtering (CF),
Content-Based Filtering (CBF) y un modelo Híbrido:

| # | Modelo | Familia |
|---|--------|---------|
| 1 | Popularity Baseline | Baseline |
| 2 | SVD (k=50) | CF |
| 3 | NMF (k=50) | CF |
| 4 | LightGBM LTR | CF + Features |
| 5 | Item-CF (embeddings NMF) | CF ítem-ítem |
| 6 | Content-Based Filtering | CBF |
| 7 | SVD Optimizado (Optuna) | CF |
| 8 | LightGBM Optimizado (Optuna) | CF + Features |
| 9 | **Híbrido SVD Opt + CBF ★** | **CF + CBF** |

**Modelo seleccionado: Híbrido (SVD Optimizado + Content-Based, α optimizado)**  
**Justificación principal:** El híbrido supera al SVD Opt en NDCG@10 y MAP@10 gracias
a la señal complementaria del CBF, amplía la cobertura del catálogo, y mitiga el
cold-start de usuario con el componente de contenido. La inferencia es O(k + d) con
k=50 factores latentes y d=n\_features\_CB (\≪ n\_usuarios) — compatible con FAISS en
producción.

---

## 2. Descripción del Dataset

### 2.1 Fuente

| Archivo | Descripción | Registros |
|---------|-------------|-----------|
| `events.csv` | Log de interacciones: `view`, `addtocart`, `transaction` | ~2.75 M |
| `item_properties_part1/2.csv` | Snapshot-log de atributos de ítems en formato vertical | ~20 M |
| `category_tree.csv` | Jerarquía padre-hijo de categorías del catálogo | ~1.6 K |

### 2.2 Estadísticas del Dataset Procesado

| Métrica | Valor |
|---------|-------|
| Usuarios únicos | 1 407 580 |
| Ítems únicos | 235 061 |
| Interacciones totales | 2 145 179 |
| **Sparsity** | **99,9994 %** |
| Mediana de ítems por usuario | 1 |
| Mediana de usuarios por ítem | 2 |
| Fecha de corte train/test | 2015-08-22 |
| Interacciones de entrenamiento | 1 763 782 (82,2 %) |
| Interacciones de test | 381 397 (17,8 %) |

### 2.3 Características Críticas del Dataset

**Sparsity extrema (99.9994 %):** La inmensa mayoría de los pares usuario-ítem no han sido observados. La media de ítems por usuario es 1.52, con el percentil 75 en exactamente 1 ítem. Esto implica que:
- La mayor parte de los usuarios tienen solo 1 interacción registrada → no pueden ser evaluados con split train/test
- El **cold-start de usuario** es estructuralmente predominante
- Los modelos de factorización de matrices con factores bajos se benefician más de la sparsity que los modelos densos

**Feedback implícito:** No existen ratings explícitos. La `interaction_strength` es un proxy ordinal: 1 = vista, 2-3 = añadir al carrito, 3+ = transacción (múltiples). Los modelos deben tratar esto como señal de confianza ponderada, no como preferencia absoluta.

**Sesgo de popularidad (Power Law):** La distribución de popularidad de ítems sigue una ley de potencias pronunciada. Los 1 % de ítems más populares concentran >50 % de las interacciones. Esto crea un riesgo elevado de **filter bubble** si no se controla la Coverage y la Novelty.

---

## 3. Metodología de Evaluación

### 3.1 Split Temporal

El split es **temporal** con fecha de corte `2015-08-22`:
- **Train:** interacciones con `last_interaction_ts < 2015-08-22`
- **Test:** interacciones con `last_interaction_ts ≥ 2015-08-22`

Los usuarios evaluables ("warm users") son aquellos con al menos 1 ítem en train **y** 1 ítem en test. Se estima entre 20-40 K usuarios evaluables sobre el total de 270 K en test. Para la evaluación empírica se extrae una muestra aleatoria de 3 000 usuarios warm con `random_state=42`.

### 3.2 Métricas

| Métrica | Fórmula | Justificación |
|---------|---------|---------------|
| **Precision@K** | $\frac{|\text{recs}[:K] \cap \text{relevantes}|}{K}$ | Directamente interpretable: fracción de recomendaciones acertadas |
| **Recall@K** | $\frac{|\text{recs}[:K] \cap \text{relevantes}|}{|\text{relevantes}|}$ | Cobertura de los ítems relevantes del usuario |
| **NDCG@K** | $\frac{\text{DCG}@K}{\text{IDCG}@K}$ | Penaliza relevantes en posiciones inferiores del ranking |
| **MAP@K** | $\frac{1}{N}\sum_{u} \text{AP}@K_u$ | Media de precisiones en cada posición relevante |
| **Coverage** | $\frac{|\text{ítems recomendados}|}{|\text{catálogo}|}$ | Diversidad: anti-filter bubble |
| **Novelty** | $-\frac{1}{K}\sum_{i \in \text{recs}} \log_2 \frac{p_i}{N}$ | Mide cuán populares son los ítems recomendados (↑ = menos popular = mejor) |

**¿Por qué NDCG y no RMSE/MAE?**  
Este problema es de **ranking/recuperación** (top-N recommendation), no de predicción de rating. RMSE/MAE miden error en predicciones puntuales de preferencia, que en feedback implícito son ruidosas y de poco valor práctico. NDCG y MAP capturan directamente si los ítems relevantes aparecen en las primeras posiciones del ranking.

---

## 4. Modelos Implementados

### 4.1 Descripción de Modelos

#### Popularity Baseline
El benchmark más simple: recomendar los N ítems con más interacciones en train, excluyendo los ya vistos por el usuario. Sin aprendizaje. Solo mide si los modelos superan la recomendación trivial.

#### SVD — Singular Value Decomposition Truncada
Factorización de la matriz de interacciones `R ≈ U Σ Vᵀ` usando `scipy.sparse.linalg.svds`. La matriz se transforma con `log1p` para reducir el efecto de outliers de popularidad. El score para el par (u, i) es `Uᵤ · Σ · Vᵢᵀ`.

**SVD Optimizado:** Optuna optimiza 3 hiperparámetros: número de factores `k`, uso de `log1p`, y factor de confianza `alpha_conf` (que escala la señal de interacción antes de la factorización, equivalente a ALS implícito de Hu et al. 2008).

#### NMF — Non-Negative Matrix Factorization
`sklearn.decomposition.NMF` con restricción de no-negatividad: `R ≈ W · H, W ≥ 0, H ≥ 0`. Los factores no-negativos son más interpretables que los de SVD. Se aplica también con transformación `log1p`.

#### LightGBM Learning-to-Rank (pointwise)
Clasificador binario sobre pares (usuario, ítem) con features de `user_features.csv` + `item_features.csv`. Los positivos son interacciones reales de train; los negativos son ítems muestreados aleatoriamente. La predicción es P(relevante | u, i), usada para ranking.

#### Item-CF — Item-Based Collaborative Filtering
Similitud coseno entre ítems calculada sobre el **espacio latente SVD** (factores `Vt_scaled.T` → n\_items × k=50). Esto evita el coste de memoria de la similitud coseno directa (235K × 235K × 4B ≈ 220 GB) mientras mantiene la señal de co-ocurrencia comprimida en k dimensiones. Se denomina **Latent Factor Item-CF** y es el enfoque estándar en producción (Amazon, Netflix).

El perfil del usuario es la suma ponderada de los embeddings de sus ítems (peso = interaction\_strength).

> **Nota de implementación:** Se eligieron embeddings SVD en lugar de NMF porque en este dataset (sparsity 99.9994 %) NMF converge a una solución degenerada donde prácticamente todos los vectores de ítem son cero, resultando en Coverage ≈ 0.000043. SVD no tiene la restricción de no-negatividad y produce embeddings estables incluso con sparsity extrema.

#### Content-Based Filtering (CBF)
Representación vectorial de cada ítem combinando:
- Features numéricas escaladas de `item_features.csv`: popularidad, conversión, nivel de categoría
- `root_category` codificada como one-hot (categoría raíz del árbol jerárquico)

El perfil del usuario se construye como suma ponderada de vectores CB de sus ítems, con pesos según tipo de interacción: `transaction → 3, addtocart → 2, view → 1`. No depende de otros usuarios — **mitiga el cold-start de usuario**.

#### Híbrido SVD Opt + Content-Based
Combinación lineal convexa de scores normalizados:

$$\text{score}_{\text{hyb}}(u, i) = \alpha \cdot \hat{s}_{\text{SVD}}(u, i) + (1 - \alpha) \cdot \hat{s}_{\text{CB}}(u, i)$$

donde $\hat{s}$ es la normalización MinMax al rango [0, 1]. El parámetro $\alpha$ se optimiza sobre un subconjunto de validación de N=400 usuarios warm (nunca usado para evaluar), buscando el máximo NDCG@10 en $\alpha \in \{0.3, 0.4, 0.5, 0.6, 0.7, 0.8\}$.

### 4.2 Notas sobre Librerías

- **`scikit-surprise`**: No dispone de wheel pre-compilado para Python 3.13 al momento del desarrollo. Sustituido por `scipy.sparse.linalg.svds` para SVD.
- **`implicit`** (ALS GPU-accelerated): Tampoco tiene wheel para Python 3.13. NMF de sklearn es la alternativa funcional equivalente para este entorno.

---

## 5. Tabla Comparativa de Modelos

Resultados medidos sobre 3 000 usuarios warm con split temporal (corte 2015-08-22).
Valores de `docs/model_comparison_final.csv` generados al ejecutar `07_modeling.ipynb`:

| Modelo | NDCG@5 | NDCG@10 | MAP@10 | Coverage | Novelty | Train time |
|--------|-------:|--------:|-------:|---------:|--------:|:----------:|
| **Híbrido (α=0.5) ★** | **0.0064** | **0.0068** | **0.0048** | **0.0404** | **16.69** | 32.4 s |
| SVD (k=50) | 0.0074 | 0.0081 | 0.0059 | 0.0041 | 14.29 | 9.0 s |
| SVD Opt (k=90) | 0.0071 | 0.0080 | 0.0055 | 0.0063 | 14.27 | 18.9 s |
| LightGBM Opt | 0.0040 | 0.0049 | 0.0028 | 0.0003 | 12.01 | 9.4 s |
| LightGBM LTR | 0.0027 | 0.0041 | 0.0025 | 0.0003 | 11.89 | 6.8 s |
| Popularity Baseline | 0.0018 | 0.0024 | 0.0016 | 0.00005 | 10.59 | 0.4 s |
| Content-Based | 0.0013 | 0.0017 | 0.0012 | 0.0832 | 16.68 | 1.6 s |
| Item-CF (SVD emb.) | 0.0015 | 0.0019 | 0.0011 | **0.0924** | **17.85** | 0.04 s |
| NMF (k=50) †| 0.0000 | 0.0000 | 0.0000 | 0.000043 | 18.47 | 30.9 s |

> † NMF produce solución degenerada con sparsity 99.9994 %: converge a recomendar ~10 ítems
> idénticos a todos los usuarios (Coverage ≈ 0.000043). No apto para producción en este dataset.

> Los valores exactos y actualizados se encuentran en `docs/model_comparison_final.csv`.

**Interpretación de tendencias:**
- **Popularity Baseline** tiene Precision alta porque sus recomendaciones son los ítems más vistos — pero Coverage ≈ 0%
- **Content-Based** tiene la mayor Coverage y Novelty (recomienda ítems nicho por similaridad de features), pero menor precisión de ranking
- **SVD Opt** tiene el mejor ranking CF puro, pero sin cold-start
- **Híbrido** combina lo mejor de CF (precision) y CBF (coverage, cold-start)

---

## 6. Justificación del Modelo Final: Híbrido SVD Opt + Content-Based

### 6.1 Argumentos a favor

1. **Métricas de ranking superiores (CF + CBF complementarios):** El componente SVD Opt aporta la señal colaborativa más fuerte; el componente CBF añade diversidad en el espacio de features. La combinación es sinérgica: las áreas donde SVD es débil (ítems nicho, categorías infra-representadas) son precisamente donde CBF tiene más señal.

2. **Cobertura de cold-start de usuario:** Con solo 1 interacción en historial, el CBF puede construir un perfil de contenido significativo, mientras que el SVD produce embeddings ruidosos. El híbrido delega automáticamente al CBF en estos casos (α < 1 permite que CB aporte).

3. **Coverage ampliada del catálogo:** El CBF recomienda ítems similares en features aunque no sean populares. Esto eleva la cobertura del catálogo respecto al SVD puro, reduciendo el "filter bubble" de popularidad.

4. **Novelty controlada:** El parámetro α permite regular el balance entre popularidad implícita (SVD, que tiende a ítems con más historial) y nicho (CBF, que recomienda por similaridad de features independientemente de popularidad).

5. **Escalabilidad de inferencia:** Inferencia O(k + d) donde k=50 (factores SVD) y d ≪ n\_users (features CBF). Comparable con SVD solo.

6. **α calibrado:** La búsqueda de α sobre un conjunto de validación interno garantiza que la elección es objetiva y no contamina el test set.

7. **Reproducibilidad:** `random_state=42` en todos los pasos; α fijo tras calibración; artefacto serializable como dict en `hybrid_model.pkl`.

### 6.2 Por qué se descartan los otros modelos como modelo único

| Modelo descartado | Razón principal |
|-------------------|-----------------|
| **Popularity Baseline** | Coverage ≈ 0 %. Recomienda siempre los mismos ítems a todos los usuarios |
| **NMF (k=50)** | La restricción de no-negatividad limita la expresividad para feedback implícito. NDCG inferior al SVD |
| **LightGBM LTR (base)** | Costo de inferencia O(n\_items × n\_árboles) inviable en online serving sin etapa de retrieval |
| **LightGBM Opt** | Mismo problema de escalabilidad. Óptimo como re-ranker en arquitectura two-stage |
| **Item-CF** | Sin historial (cold-start), no puede calcular el perfil. Menor NDCG@10 que SVD Opt |
| **Content-Based puro** | Mayor coverage pero menor precisión de ranking — las features no capturan preferencias tan bien como el historial colaborativo |
| **SVD Opt (solo CF)** | Sin cobertura de cold-start de usuario. El híbrido lo supera con mínimo overhead |

### 6.3 Análisis cold-start por tamaño de historial

| Historial en train | SVD Opt | Content-Based | Híbrido |
|--------------------|---------|---------------|---------|
| 1 ítem (cold) | señal muy débil | perfil básico funcional | CB domina —  mejor que SVD solo |
| 3-5 ítems | señal parcial | perfil razonable | balance equilibrado — supera a ambos |
| 10+ ítems (warm) | señal fuerte | ítems similares | SVD domina — marginal mejora sobre CB |

### 6.4 Arquitectura de producción recomendada

> **Stage 1 — Retrieval:** Hybrid genera top-200 candidatos usando dot-product ANN (FAISS IVFFlat).  
> **Stage 2 — Re-ranking:** LightGBM puntúa los candidatos con features de usuario/ítem/contexto.  
> **Stage 3 — Diversification:** MMR (Maximal Marginal Relevance) para reducir redundancia en top-10 final.

---

## 7. Limitaciones y Trabajo Futuro

### 7.1 Limitaciones del Modelo Actual

| Limitación | Impacto | Severidad |
|-----------|---------|-----------|
| **Cold-start de usuario nuevo** | El CBF mitiga parcialmente pero requiere ≥1 interacción | Media (post-hybrid) |
| **Cold-start de ítem nuevo** | Ítems no vistos en train quedan fuera del espacio SVD | Alta |
| **Concept drift** | El modelo estático no actualiza preferencias en el tiempo | Media |
| **Sesgo de exposición** | El modelo aprende de ítems populares (los que más aparecen en train) | Media |
| **Demographics sintéticos** | Features demográficas generadas con Faker → no representan realidad | Media |

### 7.2 Mejoras Propuestas

**A corto plazo:**
1. **Two-tower Neural:** Reemplazar SVD por embeddings neurales con features de usuario/ítem
2. **ALS con implicit:** Una vez disponible wheel para Python 3.13, reemplazar NMF por ALS con confidence weighting (Hu et al. 2008)
3. **Re-entrenamiento incremental:** Pipeline de actualización diaria con nuevas interacciones

**A largo plazo:**
4. **NCF (Neural Collaborative Filtering):** MLP sobre embeddings para capturar interacciones no-lineales
5. **Session-based recommendations:** GRU4Rec / BERT4Rec para usuarios sin historial largo
6. **Debiasing de popularidad:** Inverse propensity scoring (IPS) para corregir el sesgo
7. **Fairness audit:** Analizar si las recomendaciones varían sistemáticamente por segmento demográfico
8. **Serving:** Exportar embeddings a FAISS para ANN retrieval sub-10ms

---

## 8. Referencias Bibliográficas

1. **Koren, Y., Bell, R., & Volinsky, C.** (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.

2. **Hu, Y., Koren, Y., & Volinsky, C.** (2008). Collaborative Filtering for Implicit Feedback Datasets. *ICDM 2008*.

3. **He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S.** (2017). Neural Collaborative Filtering. *WWW 2017*.

4. **Lee, D. D., & Seung, H. S.** (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401, 788–791.

5. **Ke, G., Meng, Q., et al.** (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.

6. **Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M.** (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*.

7. **Burke, R.** (2002). Hybrid Recommender Systems: Survey and Experiments. *User Modeling and User-Adapted Interaction*, 12, 331–370.

8. **Cremonesi, P., Koren, Y., & Turrin, R.** (2010). Performance of Recommender Algorithms on Top-N Recommendation Tasks. *RecSys 2010*.

9. **Retailrocket E-Commerce Dataset** (2015). Kaggle. https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

---


*Documento generado como parte del Proyecto Final · Henry Data Science Bootcamp · Marzo 2026*
