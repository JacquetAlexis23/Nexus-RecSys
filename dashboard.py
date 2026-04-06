import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Nexus Data Co. | Enterprise RecSys",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ENTERPRISE CSS STYLING
# ==========================================
st.markdown("""
<style>
    /* Global Backgrounds */
    .stApp { background-color: #0b0f19; color: #cbd5e1; font-family: 'Inter', sans-serif;}
    
    /* Hide Streamlit components */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography */
    h1, h2, h3, h4 { color: #f8fafc; font-weight: 600; }
    
    /* Metric Cards */
    .metric-container {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-container:hover { transform: translateY(-3px); border-color: #10b981; }
    .metric-title { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 5px; }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #f8fafc; line-height: 1.2; }
    .metric-delta { font-size: 0.9rem; font-weight: 500; margin-top: 5px;}
    .delta-up { color: #10b981; }
    .delta-down { color: #ef4444; }

    /* Panels for Charts */
    .chart-panel {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .chart-panel h3 { margin-top: 0px; font-size: 1.1rem; color: #10b981; border-bottom: 1px solid #1e293b; padding-bottom: 10px; margin-bottom: 15px; }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border-bottom: 2px solid #1e293b; gap: 24px; }
    .stTabs [data-baseweb="tab"] { color: #64748b; font-weight: 500; font-size: 1.05rem; padding-bottom: 10px; padding-top: 10px; border: none !important; background: transparent !important;}
    .stTabs [aria-selected="true"] { color: #10b981 !important; border-bottom: 3px solid #10b981 !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATA GENERATION & SESSION CACHE
# ==========================================
@st.cache_data
def load_enterprise_data():
    np.random.seed(101)
    
    # Timeline smooth data
    dates = pd.date_range(start="2015-05-01", end="2015-09-18", freq="D")
    base_views = np.linspace(12000, 20000, len(dates)) # Crecimiento
    noise = np.random.normal(0, 1000, len(dates))
    seasonality = np.sin(np.arange(len(dates)) * (2 * np.pi / 7)) * 1500 # Semanal
    
    views = np.maximum((base_views + noise + seasonality), 5000).astype(int)
    carts = (views * np.random.uniform(0.02, 0.03, len(dates))).astype(int)
    sales = (carts * np.random.uniform(0.30, 0.35, len(dates))).astype(int)
    
    df_time = pd.DataFrame({"Date": dates, "Views": views, "Carts": carts, "Transactions": sales})
    df_time['Date'] = pd.to_datetime(df_time['Date'])
    
    # Models performance
    df_models = pd.DataFrame({
        "Model": ["Popularity", "SVD Base", "SVD+TD+IPS", "EASE^R", "Mult-VAE", "RP3beta", "RP3+TD", "Ens Spearman", "Ensemble Final"],
        "NDCG10": [0.0035, 0.0081, 0.0093, 0.0193, 0.0255, 0.0258, 0.02859, 0.04069, 0.04310],
        "Prec10": [0.0011, 0.0019, 0.0022, 0.0047, 0.0064, 0.0060, 0.0071, 0.0102, 0.0115],
        "Inf_ms": [0.5, 2.1, 2.5, 55.0, 120.0, 3.5, 3.8, 12.0, 16.5]
    })
    
    # Longtail smoothing
    rank = np.arange(1, 1001)
    interactions = 120000 / (rank ** 0.85)
    df_lt = pd.DataFrame({"Rank": rank, "Volume": interactions})
    
    return df_time, df_models, df_lt

df_time, df_models, df_lt = load_enterprise_data()

# ==========================================
# 4. SIDEBAR GLOBAL FILTERS (SEGMENTADORES)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3214/3214746.png", width=50)
    st.markdown("<h2 style='margin-top:0;'>Nexus Ops</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("**1. Rango de Fechas**")
    min_date, max_date = df_time['Date'].min(), df_time['Date'].max()
    date_range = st.date_input("Periodo de Análisis", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    st.markdown("**2. Segmento de Usuario**")
    segmento_opt = st.selectbox("Grupo Demográfico", ["Global (Todos)", "América Latina (LATAM)", "Norte América (NA)", "Europa (EU)"])
    
    st.markdown("**3. Categoría de Producto**")
    cat_opt = st.selectbox("Filtro de Catálogo", ["Todas las Categorías", "Electrónica", "Ropa & Calzado", "Hogar & Jardín", "Deportes"])
    
    st.markdown("---")
    st.markdown("<p style='color:#10b981; font-weight:bold; font-size:0.9rem;'>● SISTEMA ONLINE</p>", unsafe_allow_html=True)
    st.caption(f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# ==========================================
# 5. FILTERING LOGIC
# ==========================================
# Date Filter Logic
if len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = date_range[0], date_range[0]

# Apply masks
mask_date = (df_time['Date'].dt.date >= start_d) & (df_time['Date'].dt.date <= end_d)
df_filtered = df_time.loc[mask_date]

# Multipliers based on segment to mock real filter interactivity
mult_vol = 1.0
if segmento_opt == "América Latina (LATAM)": mult_vol = 0.4
elif segmento_opt == "Norte América (NA)": mult_vol = 0.35
elif segmento_opt == "Europa (EU)": mult_vol = 0.25

if cat_opt != "Todas las Categorías": mult_vol *= 0.3

df_filtered = df_filtered.copy()
df_filtered['Views'] = (df_filtered['Views'] * mult_vol).astype(int)
df_filtered['Carts'] = (df_filtered['Carts'] * mult_vol).astype(int)
df_filtered['Transactions'] = (df_filtered['Transactions'] * mult_vol).astype(int)

# Real time calculated KPI values
tot_views = df_filtered['Views'].sum()
tot_carts = df_filtered['Carts'].sum()
tot_tx = df_filtered['Transactions'].sum()
cr_base = (tot_tx / tot_views * 100) if tot_views > 0 else 0

# Formatters
def fmt(num):
    if num >= 1_000_000: return f"{num/1_000_000:.2f}M"
    elif num >= 1_000: return f"{num/1_000:.1f}K"
    return str(int(num))

# ==========================================
# 6. HEADER & KPI DASHBOARD
# ==========================================
st.markdown("<h1 style='padding-top:10px;'>📊 Dashboard Ejecutivo de Recomendación</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>Panel empresarial integrado. Los KPIs a continuación responden dinámicamente a los filtros seleccionados.</p>", unsafe_allow_html=True)

# 5 Top KPIs
m1, m2, m3, m4, m5 = st.columns(5)
m1.markdown(f"""
<div class='metric-container'>
    <div class='metric-title'>Vistas Totales</div>
    <div class='metric-value'>{fmt(tot_views)}</div>
    <div class='metric-delta delta-up'>↑ 5.2% vs previo</div>
</div>""", unsafe_allow_html=True)

m2.markdown(f"""
<div class='metric-container'>
    <div class='metric-title'>Carritos (Add to Cart)</div>
    <div class='metric-value'>{fmt(tot_carts)}</div>
    <div class='metric-delta delta-up'>↑ 2.1% vs previo</div>
</div>""", unsafe_allow_html=True)

m3.markdown(f"""
<div class='metric-container'>
    <div class='metric-title'>Tasa de Conversión</div>
    <div class='metric-value'>{cr_base:.2f}%</div>
    <div class='metric-delta delta-down'>↓ 0.05% Riesgo</div>
</div>""", unsafe_allow_html=True)

m4.markdown(f"""
<div class='metric-container'>
    <div class='metric-title'>Lift Esperado RecSys</div>
    <div class='metric-value'>+50.8%</div>
    <div class='metric-delta delta-up'>Basado en NB15</div>
</div>""", unsafe_allow_html=True)

m5.markdown(f"""
<div class='metric-container'>
    <div class='metric-title'>Sparsity Matriz</div>
    <div class='metric-value'>99.9%</div>
    <div class='metric-delta delta-down'>Operación Crítica</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 7. CHART BUILDING (THEME CONFIG)
# ==========================================
# Base layout setup to apply to all charts for an Enterprise look
def apply_enterprise_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=20, r=20, t=40, b=20),
        title_font=dict(size=14, color="#f8fafc", family="Inter"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#334155")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", linecolor="#334155", zeroline=False)
    return fig

color_palette = ["#10b981", "#3b82f6", "#f59e0b", "#6366f1", "#ec4899", "#14b8a6"]

# ==========================================
# 8. TABS DEFINITION
# ==========================================
t_biz, t_problem, t_model, t_ai = st.tabs([
    "📈 Operatividad Comercial",
    "⚠️ Diagnóstico de Catálogo",
    "🔬 Comparativa de Modelos IA",
    "🔌 Simulador API"
])

# ----------------- TAB 1: Comercial (5 Gráficas) -----------------
with t_biz:
    c1, c2 = st.columns([2, 1])
    
    # G1: Timeline
    with c1:
        st.markdown("<div class='chart-panel'><h3>1. Comportamiento Omnicanal a lo largo del tiempo</h3>", unsafe_allow_html=True)
        fig1 = px.line(df_filtered, x="Date", y=["Views", "Carts", "Transactions"], color_discrete_sequence=["#475569", "#3b82f6", "#10b981"])
        fig1 = apply_enterprise_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # G2: Funnel real-time
    with c2:
        st.markdown("<div class='chart-panel'><h3>2. Embudo de Conversión Crítico</h3>", unsafe_allow_html=True)
        fun_data = dict(number=[tot_views, tot_carts, tot_tx], stage=["Vistas de Detalle", "Agregado al Carro", "Transacción Pagada"])
        fig2 = go.Figure(go.Funnel(y=fun_data["stage"], x=fun_data["number"], textinfo="value+percent initial",
                                  marker={"color": ["#1e293b", "#3b82f6", "#10b981"]}))
        fig2 = apply_enterprise_theme(fig2)
        fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4, c5 = st.columns(3)
    
    # G3: Day of week
    with c3:
        st.markdown("<div class='chart-panel'><h3>3. Densidad por Día</h3>", unsafe_allow_html=True)
        df_filtered['Weekday'] = df_filtered['Date'].dt.day_name()
        df_dw = df_filtered.groupby('Weekday')['Views'].sum().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
        fig3 = px.bar(df_dw, x="Views", y="Weekday", orientation='h', color="Views", color_continuous_scale="Tealgrn")
        fig3 = apply_enterprise_theme(fig3)
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # G4: Return vs New
    with c4:
        st.markdown("<div class='chart-panel'><h3>4. Retención Global</h3>", unsafe_allow_html=True)
        fig4 = px.pie(names=['Usuarios Nuevos', 'Usuarios con Historial'], values=[72, 28], hole=0.6, color_discrete_sequence=["#1e293b", "#10b981"])
        fig4 = apply_enterprise_theme(fig4)
        fig4.update_layout(showlegend=False)
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # G5: Purchase Frequency
    with c5:
        st.markdown("<div class='chart-panel'><h3>5. Frecuencia de Compra</h3>", unsafe_allow_html=True)
        freq_df = pd.DataFrame({"Compras Mensuales": ["1 Compra", "2-3 Compras", "4+ Compras"], "Usuarios": [8500, 1200, 150]})
        fig5 = px.bar(freq_df, x="Compras Mensuales", y="Usuarios", color="Compras Mensuales", color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"])
        fig5 = apply_enterprise_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 2: Diagnóstico de Problemas (5 Gráficas) -----------------
with t_problem:
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem;'>Las siguientes gráficas exponen gráficamente la problemática matemática del sistema actual de negocio y dictan las métricas a resolver por los modelos IA.</p>", unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    
    # G6: Long Tail
    with c6:
        st.markdown("<div class='chart-panel'><h3>6. Curva The Long-Tail (Popularidad de Catálogo)</h3>", unsafe_allow_html=True)
        fig6 = px.area(df_lt, x="Rank", y="Volume", log_y=True, color_discrete_sequence=["#ec4899"])
        fig6 = apply_enterprise_theme(fig6)
        fig6.add_vline(x=200, line_dash="dash", line_color="#cbd5e1", annotation_text="Concentración 80/20")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # G7: Cold start segments
    with c7:
        st.markdown("<div class='chart-panel'><h3>7. Problema Cold-Start en Usuarios</h3>", unsafe_allow_html=True)
        cs_df = pd.DataFrame({"Tipo": ["Cold Range (≤ 2 eventos)", "Warm Range (> 2 eventos)"], "Pct": [52.3, 47.7]})
        fig7 = px.bar(cs_df, x="Tipo", y="Pct", text="Pct", color="Tipo", color_discrete_sequence=["#ef4444", "#10b981"])
        fig7 = apply_enterprise_theme(fig7)
        fig7.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c8, c9, c10 = st.columns(3)
    
    # G8: Interaction matrix size limit comparison
    with c8:
        st.markdown("<div class='chart-panel'><h3>8. Espacio Dimensional</h3>", unsafe_allow_html=True)
        dim_data = pd.DataFrame({"Eje": ["Usuarios", "Productos"], "Cantidad": [1407580, 235061]})
        fig8 = px.bar(dim_data, x="Eje", y="Cantidad", text="Cantidad", color="Eje", color_discrete_sequence=["#3b82f6", "#6366f1"])
        fig8 = apply_enterprise_theme(fig8)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # G9: Event type dist
    with c9:
        st.markdown("<div class='chart-panel'><h3>9. Distribución Imbalanceada de Logs</h3>", unsafe_allow_html=True)
        imbal_df = pd.DataFrame({"Evento": ["Views", "Add to Cart", "Sales"], "Share": [96.7, 2.5, 0.8]})
        fig9 = px.funnel(imbal_df, x="Share", y="Evento", color="Evento", color_discrete_sequence=["#1e293b", "#3b82f6", "#10b981"])
        fig9 = apply_enterprise_theme(fig9)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # G10: Categories Coverage
    with c10:
        st.markdown("<div class='chart-panel'><h3>10. Cobertura Histórica de Catálogo</h3>", unsafe_allow_html=True)
        cov_df = pd.DataFrame({"Estado": ["Vistos al menos 1 vez", "Cero Visualizaciones"], "Cnt": [115340, 119721]})
        fig10 = px.pie(cov_df, names="Estado", values="Cnt", hole=0.5, color_discrete_sequence=["#10b981", "#ff4444"])
        fig10 = apply_enterprise_theme(fig10)
        fig10.update_layout(showlegend=False)
        fig10.update_traces(textinfo='label+percent')
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: Modelos IA & Métricas (7 Gráficas) -----------------
with t_model:
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem;'>Evaluación rigurosa de familias de algoritmos y progreso contra el Baseline, entrenados con el log histórico.</p>", unsafe_allow_html=True)
    c11, c12 = st.columns([5, 3])
    
    # G11: Historical NDCG Progress
    with c11:
        st.markdown("<div class='chart-panel'><h3>11. Salto de Calidad Histórica del Pipeline (Métrica Oficial: NDCG@10)</h3>", unsafe_allow_html=True)
        fig11 = px.line(df_models, x="Model", y="NDCG10", text="NDCG10", markers=True, line_shape="spline", color_discrete_sequence=["#10b981"])
        fig11.update_traces(textposition="top center", texttemplate='%{text:.4f}', marker=dict(size=10))
        fig11 = apply_enterprise_theme(fig11)
        fig11.update_layout(yaxis_range=[0, 0.05])
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # G12: NDCG vs Precision bubble
    with c12:
        st.markdown("<div class='chart-panel'><h3>12. Trade-Off: Recall vs Precision</h3>", unsafe_allow_html=True)
        fig12 = px.scatter(df_models, x="NDCG10", y="Prec10", color="Model", size="Prec10", hover_name="Model", color_discrete_sequence=px.colors.qualitative.Plotly)
        fig12 = apply_enterprise_theme(fig12)
        st.plotly_chart(fig12, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c13, c14, c15 = st.columns(3)
    # G13: Inference times
    with c13:
        st.markdown("<div class='chart-panel'><h3>13. Riesgo Computacional (Latencia de Predicción en ms)</h3>", unsafe_allow_html=True)
        # remove baseline for clearer scale
        models_latency = df_models.iloc[3:]
        fig13 = px.bar(models_latency, x="Model", y="Inf_ms", color="Inf_ms", color_continuous_scale="Reds")
        fig13 = apply_enterprise_theme(fig13)
        fig13.update_coloraxes(showscale=False)
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # G14: ROI Lift
    with c14:
        st.markdown("<div class='chart-panel'><h3>14. ROI: Elevación de Conversión Estimada</h3>", unsafe_allow_html=True)
        up_df = pd.DataFrame({"Stage": ["A/B Base", "Aportación ML", "Aportación Ensamblado"], "Lift": [0, 25.5, 25.3]})
        fig14 = go.Figure(go.Waterfall(name="20", orientation="v", measure=["absolute", "relative", "relative"], 
                                      x=up_df["Stage"], textposition="outside", text=["0%", "+25%", "+25.3%"], y=up_df["Lift"]))
        fig14 = apply_enterprise_theme(fig14)
        st.plotly_chart(fig14, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # G15: Champion Weight Dist
    with c15:
        st.markdown("<div class='chart-panel'><h3>15. Radiografía del Champion Ensemble Optimizado</h3>", unsafe_allow_html=True)
        fig15 = px.pie(names=["RP3+TemporalDecay", "Graph EASE_500", "RP3+MatrixB+TD"], values=[2.3, 2.1, 95.6], hole=0.7, color_discrete_sequence=["#3b82f6", "#ec4899", "#10b981"])
        fig15 = apply_enterprise_theme(fig15)
        fig15.update_layout(showlegend=False)
        fig15.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig15, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ----------------- TAB 4: API Simulator (5 Gráficas) -----------------
with t_ai:
    st.markdown("<p style='color:#94a3b8; font-size:1.1rem;'>Frontend técnico de prueba para el Endpoint de inferencia recomendada en Producción.</p>", unsafe_allow_html=True)
    c_form, c_dash1, c_dash2 = st.columns([1, 1.5, 1.5])
    
    with c_form:
        st.markdown("<div class='chart-panel' style='background:#0f172a; border-color:#3b82f6;'><h3>🧪 Entorno de Petición</h3>", unsafe_allow_html=True)
        req_user = st.text_input("User Hash ID", "3A89C1X")
        req_k = st.slider("Longitud Output (K)", 3, 10, 5)
        st.markdown(" Contexto (Features Inyectados): `{country: 'AR', age: 34, history_len: 3}`")
        execute_req = st.button("Ejecutar Inferencia", use_container_width=True, type="primary")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # G16: Historic
        st.markdown("<div class='chart-panel'><h3>16. Input Vector Context</h3>", unsafe_allow_html=True)
        h_df = pd.DataFrame({"Item Actividad": ["View Item 44", "View Item 99", "Cart Item 99"], "Min_Pass": [15, 12, 2]})
        fig16 = px.bar(h_df, x="Min_Pass", y="Item Actividad", orientation="h", color_discrete_sequence=["#64748b"])
        fig16 = apply_enterprise_theme(fig16)
        fig16.update_xaxes(autorange="reversed", title="Minutos Atrás")
        fig16.update_layout(height=200)
        st.plotly_chart(fig16, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_dash1:
        if execute_req:
            with st.spinner("Conectando con Backend..."):
                time.sleep(1)
            
            # G17: Recommendation Scores
            st.markdown("<div class='chart-panel' style='border-color:#10b981;'><h3>17. Salida RAW y Niveles de Confianza (Probabilidad)</h3>", unsafe_allow_html=True)
            res_df = pd.DataFrame({
                "Ranking": [f"Rank {i+1}" for i in range(req_k)], 
                "Score": sorted([np.random.uniform(0.60, 0.98) for _ in range(req_k)], reverse=True)
            })
            fig17 = px.bar(res_df, x="Ranking", y="Score", text="Score", color="Score", color_continuous_scale="Greens")
            fig17.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig17 = apply_enterprise_theme(fig17)
            fig17.update_layout(yaxis_range=[0, 1.1])
            st.plotly_chart(fig17, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # G18: Strategy Contribution Bar
            st.markdown("<div class='chart-panel'><h3>18. Contribución de Strategy (Introspección)</h3>", unsafe_allow_html=True)
            strat_df = pd.DataFrame({"Str": ["Content-Based", "Collab Filtering", "Cold-Start Logic"], "Weight": [0.15, 0.80, 0.05]})
            fig18 = px.bar(strat_df, x="Str", y="Weight", color="Str", color_discrete_sequence=["#3b82f6", "#10b981", "#ec4899"])
            fig18 = apply_enterprise_theme(fig18)
            st.plotly_chart(fig18, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with c_dash2:
        if execute_req:
            # G19: Category Spread
            st.markdown("<div class='chart-panel'><h3>19. Dispersión Categórica del Output</h3>", unsafe_allow_html=True)
            cat_out = pd.DataFrame({"Categoría Recomendada": ["Tech", "Móviles", "Wearables"], "Proporción": [60, 30, 10]})
            fig19 = px.funnel(cat_out, x="Proporción", y="Categoría Recomendada", color_discrete_sequence=["#f59e0b", "#3b82f6", "#10b981"])
            fig19 = apply_enterprise_theme(fig19)
            st.plotly_chart(fig19, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # G20: Heatmap user similarity simulated
            st.markdown("<div class='chart-panel'><h3>20. Mapa de Similaridad Vecinal (Top Usuarios)</h3>", unsafe_allow_html=True)
            z_sim = np.random.uniform(0.5, 1, size=(5, 5))
            np.fill_diagonal(z_sim, 1.0)
            fig20 = px.imshow(z_sim, color_continuous_scale="Viridis", text_auto=".2f")
            fig20 = apply_enterprise_theme(fig20)
            fig20.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig20, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='chart-panel' style='text-align:center; padding-top: 50px; padding-bottom: 50px;'>
                <h3 style='color:#64748b; border:none;'>Esperando ejecución de Inferencia...</h3>
                <p style='color:#475569;'>Pulse el botón "Ejecutar Inferencia" para calcular las métricas predictivas (Graficas 17, 18, 19, 20).</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #334155; margin-top:40px; padding-top:20px; border-top: 1px solid #1e293b;">
    <strong>NEXUS DATA CO.® 2026</strong> | Módulo Integrado de Data Science (Nivel Empresarial) | Panel Generado V2.0 
</div>
""", unsafe_allow_html=True)
