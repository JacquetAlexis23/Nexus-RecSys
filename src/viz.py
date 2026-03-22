"""
viz.py
======
Utilidades de visualización reutilizables para el proyecto NexusDataCo.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────

def setup_style(palette: str = "viridis", font_scale: float = 1.1) -> None:
    """Aplica el estilo visual estándar del proyecto."""
    sns.set_theme(style="whitegrid", palette=palette, font_scale=font_scale)
    plt.rcParams.update({
        "figure.dpi"       : 100,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.titlesize"   : 14,
        "axes.labelsize"   : 12,
    })


# ──────────────────────────────────────────────────────────
# GRÁFICAS REUTILIZABLES
# ──────────────────────────────────────────────────────────

def plot_funnel(funnel_df: pd.DataFrame, title: str = "Conversion Funnel") -> None:
    """Bar chart horizontal con etiquetas de porcentaje."""
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.barh(funnel_df["event"], funnel_df["count"],
                   color=colors[:len(funnel_df)])
    for bar, pct in zip(bars, funnel_df["pct_total"]):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=11)
    ax.set_xscale("log")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Conteo (escala log)")
    plt.tight_layout()
    plt.show()


def plot_lorenz(series: pd.Series, label: str = "Curve") -> float:
    """
    Dibuja la Curva de Lorenz y devuelve el coeficiente de Gini.
    """
    sorted_vals   = np.sort(series.values)
    cum_share_pop = np.linspace(0, 1, len(sorted_vals) + 1)
    cum_share_val = np.concatenate([[0], np.cumsum(sorted_vals) / sorted_vals.sum()])
    gini = 1 - 2 * np.trapz(cum_share_val, cum_share_pop)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cum_share_pop, cum_share_val, lw=2, label=f"{label} (Gini={gini:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Igualdad perfecta")
    ax.fill_between(cum_share_pop, cum_share_val, alpha=0.2)
    ax.set_title("Curva de Lorenz – Concentración de Interacciones", fontweight="bold")
    ax.set_xlabel("% de Usuarios (acumulado)")
    ax.set_ylabel("% de Interacciones (acumulado)")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return gini


def plot_time_heatmap(df: pd.DataFrame,
                      datetime_col: str = "datetime",
                      title: str = "Heatmap Actividad Semanal") -> None:
    """Crea un heatmap de interacciones hora × día."""
    df = df.copy()
    df["hour"] = df[datetime_col].dt.hour
    df["dow"]  = df[datetime_col].dt.day_name()
    DOW_ORDER  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df.groupby(["dow","hour"]).size().unstack(fill_value=0).reindex(DOW_ORDER)
    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Día de la semana")
    plt.tight_layout()
    plt.show()


def plot_cold_start_pie(cold_dict: dict, title: str = "Segmentación por Interacciones") -> None:
    """Pie chart de segmentos de usuarios."""
    labels = [
        "1 Interacción (Cold-Start extremo)",
        "2 Interacciones",
        "3-5 Interacciones",
        "+5 Interacciones (Activos)",
    ]
    sizes = [
        cold_dict["cold_1"],
        cold_dict["cold_2"],
        cold_dict["low_3_5"],
        cold_dict["active_gt5"],
    ]
    colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71"]
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_long_tail(item_counts: pd.Series,
                   title: str = "Distribución Long-Tail de Ítems") -> None:
    """Gráfica de la distribución long-tail."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    # Rank plot
    axes[0].plot(range(len(item_counts)), item_counts.values, color="#8e44ad", lw=1)
    axes[0].set_title("Rank vs Interacciones (Long-Tail)", fontweight="bold")
    axes[0].set_xlabel("Rank del ítem")
    axes[0].set_ylabel("Nº de interacciones")
    axes[0].set_yscale("log")
    # Histogram log-log
    axes[1].hist(item_counts.values, bins=50, color="#1abc9c", edgecolor="white")
    axes[1].set_title("Histograma de Interacciones por Ítem", fontweight="bold")
    axes[1].set_xlabel("Nº interacciones")
    axes[1].set_ylabel("Cantidad de ítems")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
