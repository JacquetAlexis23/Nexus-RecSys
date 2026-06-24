"""
dashboard/app.py
Dashboard interactivo — RetailRocket Recommender System
Sprint 4 — NexusDataCo | SoyHenry Data Science

Ejecutar:
  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Configuracion ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetailRocket Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

COLORS = {
    "popularity":    "#888780",
    "item_cf":       "#378ADD",
    "als":           "#EF9F27",
    "ease":          "#D85A30",
    "content_based": "#534AB7",
    "hybrid":        "#1D9E75",
}

MODEL_NAMES = {
    "popularity":    "Popularidad",
    "item_cf":       "Item-CF",
    "als":           "BPR",
    "ease":          "EASE",
    "content_based": "Content-Based",
    "hybrid":        "Híbrido",
}

# ── Helpers de API ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def api_health():
    try:
        r = requests.get(f"{API_URL}/v1/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except:
        return None

@st.cache_data(ttl=300)
def api_metrics(k=10):
    try:
        r = requests.get(f"{API_URL}/v1/metrics", params={"k": k}, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

def api_recommend(visitor_id: int, model: str = "hybrid", n: int = 10):
    try:
        r = requests.get(
            f"{API_URL}/v1/recommend/{visitor_id}",
            params={"model": model, "n": n},
            timeout=10,
        )
        return r.json() if r.status_code == 200 else {"error": r.json().get("detail", "Error")}
    except Exception as e:
        return {"error": str(e)}

def api_popular(n: int = 10):
    try:
        r = requests.get(f"{API_URL}/v1/recommend/popular", params={"n": n}, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
    st.title("RetailRocket")
    st.caption("NexusDataCo · SoyHenry Data Science")
    st.divider()

    health = api_health()
    if health and health.get("status") == "ok":
        st.success(f"✅ API conectada")
        st.caption(f"Modelos: {', '.join(health.get('models_loaded', []))}")
    elif health:
        st.warning("⚠️ API degradada")
    else:
        st.error("❌ API no disponible")
        st.caption(f"Verifica que la API esté corriendo en {API_URL}")

    st.divider()
    page = st.radio(
        "Navegación",
        ["🏠 Inicio", "📊 KPIs del Funnel", "🤖 Comparativa de Modelos",
         "🎯 Simulador de Recomendaciones", "🔥 Productos Populares"],
        label_visibility="collapsed",
    )

# ── PÁGINA: INICIO ─────────────────────────────────────────────────────────────
if page == "🏠 Inicio":
    st.title("🛒 Sistema de Recomendación RetailRocket")
    st.markdown("**Equipo NexusDataCo · Bootcamp SoyHenry Data Science · Sprint 4**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Visitantes únicos", "1,407,580")
    col2.metric("Eventos analizados", "2,755,641")
    col3.metric("Productos en catálogo", "417,053")
    col4.metric("Tasa de conversión", "0.83%", delta=None)

    st.divider()
    st.subheader("Arquitectura del sistema")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **4 Sprints completados:**
        - ✅ **Sprint 1** — ETL + Análisis Exploratorio
        - ✅ **Sprint 2** — Modelos Baseline + Item-CF
        - ✅ **Sprint 3** — BPR + EASE + Content-Based + Híbrido
        - ✅ **Sprint 4** — API REST + Dashboard (este panel)
        """)
    with col2:
        st.markdown("""
        **6 Modelos entrenados:**
        - 🔘 Popularidad — baseline de referencia
        - 🔵 Item-CF — similitud colaborativa
        - 🟡 BPR — ranking personalizado implícito
        - 🟠 EASE — mejor modelo personalizado ★
        - 🟣 Content-Based — recomendación por contenido
        - 🟢 Híbrido — BPR 60% + Item-CF 40%
        """)

    st.divider()
    st.info("""
    **Cómo usar este dashboard:**
    - 📊 **KPIs del Funnel** — análisis del embudo de conversión
    - 🤖 **Comparativa de Modelos** — métricas de los 6 modelos
    - 🎯 **Simulador** — prueba el sistema con un visitor_id real
    - 🔥 **Populares** — top productos del catálogo
    """)

# ── PÁGINA: KPIs DEL FUNNEL ────────────────────────────────────────────────────
elif page == "📊 KPIs del Funnel":
    st.title("📊 KPIs del Funnel de Conversión")
    st.caption("Análisis del comportamiento de 1,407,580 visitantes — 137 días")
    st.divider()

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Visitantes totales", "1,407,580", help="Usuarios únicos en el dataset")
    col2.metric("Vieron productos", "1,404,179", "99.8%")
    col3.metric("Agregaron al carrito", "37,722", "2.7%")
    col4.metric("Compraron", "11,719", "0.83%", delta_color="off")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Funnel de conversión")
        funnel_df = pd.DataFrame({
            "Etapa": ["Visitantes", "View", "Add to Cart", "Transaction"],
            "Usuarios": [1_407_580, 1_404_179, 37_722, 11_719],
        })
        fig = go.Figure(go.Funnel(
            y=funnel_df["Etapa"],
            x=funnel_df["Usuarios"],
            textinfo="value+percent initial",
            marker={"color": ["#1F4E79", "#2E75B6", "#EF9F27", "#1D9E75"]},
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribución de eventos")
        ev_df = pd.DataFrame({
            "Tipo":    ["View", "Add to Cart", "Transaction"],
            "Eventos": [2_664_312, 69_332, 22_457],
            "Peso":    [1, 4, 10],
        })
        fig2 = px.bar(ev_df, x="Tipo", y="Eventos", color="Tipo",
                      color_discrete_sequence=["#378ADD", "#EF9F27", "#1D9E75"],
                      text="Eventos")
        fig2.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig2.update_layout(showlegend=False, height=350,
                           margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de usuarios por actividad")
        usr_df = pd.DataFrame({
            "Segmento":  ["1 evento\n(cold-start)", "2 eventos", "3-9 eventos", "10+ eventos (VIP)"],
            "Usuarios":  [1_001_560, 205_992, 176_787, 23_241],
            "Porcentaje":["71.2%", "14.6%", "12.6%", "1.65%"],
        })
        fig3 = px.pie(usr_df, values="Usuarios", names="Segmento",
                      color_discrete_sequence=["#D85A30","#EF9F27","#378ADD","#1D9E75"])
        fig3.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig3, use_container_width=True)
        st.warning("⚠️ **71.2% de usuarios son cold-start** — el sistema los atiende con popularidad por categoría")

    with col2:
        st.subheader("Cobertura del catálogo")
        cat_df = pd.DataFrame({
            "Estado":   ["Vistos y con propiedades", "Solo vistos", "Solo propiedades (invisibles)"],
            "Items":    [185_246, 49_815, 231_807],
        })
        fig4 = px.pie(cat_df, values="Items", names="Estado",
                      color_discrete_sequence=["#1D9E75","#378ADD","#888780"])
        fig4.update_layout(height=320, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig4, use_container_width=True)
        st.info("📦 **55.5% del catálogo nunca fue visto** — Content-Based y EASE ayudan a activarlo")

# ── PÁGINA: COMPARATIVA DE MODELOS ────────────────────────────────────────────
elif page == "🤖 Comparativa de Modelos":
    st.title("🤖 Comparativa de los 6 Modelos")
    st.divider()

    k_sel = st.select_slider("Valor de K", options=[5, 10, 20], value=10)

    metrics_data = api_metrics(k=k_sel)

    if metrics_data is None:
        st.error("No se puede conectar a la API. Verifica que esté corriendo.")
    else:
        models_raw = metrics_data.get("models", [])
        df = pd.DataFrame(models_raw)

        if df.empty:
            st.warning("Sin datos de métricas disponibles.")
        else:
            df["model_display"] = df["model"].map(MODEL_NAMES).fillna(df["model"])
            df["color"] = df["model"].map(COLORS).fillna("#888780")

            # Tabla resumen
            st.subheader(f"Métricas @ K={k_sel}")
            disp = df[["model_display","precision","recall","ndcg","hitrate","coverage","novelty"]].copy()
            disp.columns = ["Modelo","Precision","Recall","NDCG","HitRate","Coverage","Novelty"]
            disp = disp.sort_values("NDCG", ascending=False)

            def highlight_best(s):
                is_max = s == s.max()
                return ["background-color: #C6EFCE; font-weight: bold" if v else "" for v in is_max]

            st.dataframe(
                disp.style.apply(highlight_best, subset=["Precision","Recall","NDCG","HitRate"]),
                use_container_width=True, hide_index=True,
            )

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("NDCG@K — calidad del ranking")
                fig_ndcg = px.bar(
                    df.sort_values("ndcg", ascending=True),
                    x="ndcg", y="model_display", orientation="h",
                    color="model", color_discrete_map={k: COLORS.get(k,"#888780") for k in df["model"]},
                    text=df.sort_values("ndcg")["ndcg"].round(4),
                )
                fig_ndcg.update_traces(textposition="outside")
                fig_ndcg.update_layout(showlegend=False, height=320,
                                       margin=dict(l=0, r=0, t=20, b=0),
                                       xaxis_title="NDCG", yaxis_title="")
                st.plotly_chart(fig_ndcg, use_container_width=True)

            with col2:
                st.subheader("HitRate@K — usuarios con al menos 1 acierto")
                fig_hit = px.bar(
                    df.sort_values("hitrate", ascending=True),
                    x="hitrate", y="model_display", orientation="h",
                    color="model", color_discrete_map={k: COLORS.get(k,"#888780") for k in df["model"]},
                    text=df.sort_values("hitrate")["hitrate"].round(4),
                )
                fig_hit.update_traces(textposition="outside")
                fig_hit.update_layout(showlegend=False, height=320,
                                      margin=dict(l=0, r=0, t=20, b=0),
                                      xaxis_title="HitRate", yaxis_title="")
                st.plotly_chart(fig_hit, use_container_width=True)

            st.divider()
            st.subheader("Trade-off: Precision vs Coverage del catálogo")
            fig_scatter = px.scatter(
                df, x=df["coverage"]*100, y="precision",
                color="model", color_discrete_map=COLORS,
                text="model_display", size_max=20,
                labels={"x":"Coverage (%)", "precision":"Precision@K"},
            )
            fig_scatter.update_traces(marker_size=14, textposition="top center")
            fig_scatter.update_layout(showlegend=False, height=400,
                                      margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("**Objetivo:** un buen sistema vive en la esquina superior derecha — alta precisión Y alta cobertura.")

# ── PÁGINA: SIMULADOR ─────────────────────────────────────────────────────────
elif page == "🎯 Simulador de Recomendaciones":
    st.title("🎯 Simulador de Recomendaciones")
    st.caption("Prueba el sistema en tiempo real con cualquier visitor_id")
    st.divider()

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        visitor_id = st.number_input(
            "Visitor ID", min_value=1, value=1160840, step=1,
            help="Ingresa cualquier visitor_id del dataset RetailRocket"
        )
    with col2:
        model_sel = st.selectbox(
            "Modelo",
            options=list(MODEL_NAMES.keys()),
            format_func=lambda x: MODEL_NAMES[x],
            index=list(MODEL_NAMES.keys()).index("hybrid"),
        )
    with col3:
        n_recs = st.number_input("Top-N", min_value=1, max_value=50, value=10)

    col_btn, col_compare = st.columns([1, 3])
    run = col_btn.button("🚀 Recomendar", type="primary", use_container_width=True)
    compare_all = col_compare.checkbox("Comparar todos los modelos para este usuario")

    if run:
        if not compare_all:
            with st.spinner("Consultando la API..."):
                result = api_recommend(visitor_id, model_sel, n_recs)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Modelo usado", MODEL_NAMES.get(result["model_used"], result["model_used"]))
                col2.metric("Estrategia", result.get("strategy", "-"))
                col3.metric("Items devueltos", result["n"])
                col4.metric("Tiempo respuesta", f"{result['response_ms']} ms")

                st.divider()
                items = result.get("items", [])
                if items:
                    df_recs = pd.DataFrame(items)
                    df_recs["score_pct"] = (df_recs["score"] * 100).round(1)

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("Top recomendaciones")
                        st.dataframe(
                            df_recs[["rank","itemid","score","category_id"]].rename(
                                columns={"rank":"#","itemid":"Item ID",
                                         "score":"Score","category_id":"Categoría"}
                            ),
                            use_container_width=True, hide_index=True,
                        )
                    with col2:
                        st.subheader("Distribución de scores")
                        fig_scores = px.bar(
                            df_recs, x="rank", y="score",
                            color="score", color_continuous_scale="Blues",
                            labels={"rank":"Posición","score":"Score"},
                            text=df_recs["score"].round(3),
                        )
                        fig_scores.update_traces(textposition="outside")
                        fig_scores.update_layout(
                            height=350, showlegend=False,
                            margin=dict(l=0, r=0, t=20, b=0),
                            coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig_scores, use_container_width=True)
        else:
            # Comparar todos los modelos
            st.subheader(f"Recomendaciones para visitor_id={visitor_id} — todos los modelos")
            tabs = st.tabs([MODEL_NAMES[m] for m in MODEL_NAMES.keys()])

            for tab, (model_key, model_label) in zip(tabs, MODEL_NAMES.items()):
                with tab:
                    with st.spinner(f"Consultando {model_label}..."):
                        result = api_recommend(visitor_id, model_key, n_recs)

                    if "error" in result:
                        st.warning(f"No disponible: {result['error']}")
                    else:
                        col1, col2 = st.columns(2)
                        col1.metric("Estrategia", result.get("strategy","-"))
                        col2.metric("Tiempo", f"{result['response_ms']} ms")

                        items = result.get("items", [])
                        if items:
                            df_t = pd.DataFrame(items)
                            st.dataframe(
                                df_t[["rank","itemid","score","category_id"]].rename(
                                    columns={"rank":"#","itemid":"Item ID",
                                             "score":"Score","category_id":"Categoría"}
                                ),
                                use_container_width=True, hide_index=True, height=350,
                            )
                        else:
                            st.info("Sin recomendaciones — este modelo no tiene datos de este usuario.")

    else:
        st.info("👆 Ingresa un visitor_id y haz clic en **Recomendar** para ver los resultados.")
        st.markdown("""
        **Usuarios de ejemplo para probar:**
        - `1160840` — usuario warm con historial
        - `862095` — usuario con pocas interacciones
        - `9999999` — usuario nuevo (cold-start)
        """)

# ── PÁGINA: POPULARES ──────────────────────────────────────────────────────────
elif page == "🔥 Productos Populares":
    st.title("🔥 Productos Más Populares")
    st.caption("Top productos del catálogo por score de popularidad ponderado")
    st.divider()

    n_pop = st.slider("Número de productos", min_value=5, max_value=50, value=20)

    with st.spinner("Cargando productos populares..."):
        pop_data = api_popular(n=n_pop)

    if pop_data is None:
        st.error("No se puede obtener los productos. Verifica que la API esté corriendo.")
    else:
        items = pop_data.get("items", [])
        if items:
            df_pop = pd.DataFrame(items)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Tabla de productos")
                st.dataframe(
                    df_pop[["rank","itemid","score","category_id"]].rename(
                        columns={"rank":"#","itemid":"Item ID",
                                 "score":"Score","category_id":"Categoría"}
                    ),
                    use_container_width=True, hide_index=True,
                )

            with col2:
                st.subheader("Score por posición")
                fig_pop = px.bar(
                    df_pop, x="rank", y="score",
                    color="score", color_continuous_scale="RdYlGn",
                    labels={"rank":"Posición","score":"Score de popularidad"},
                    text=df_pop["itemid"].astype(str),
                )
                fig_pop.update_traces(textposition="outside", textfont_size=8)
                fig_pop.update_layout(
                    height=450, showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_pop, use_container_width=True)

            st.caption(f"⏱ Tiempo de respuesta: {pop_data.get('response_ms', '-')} ms")
        else:
            st.warning("Sin datos disponibles.")
