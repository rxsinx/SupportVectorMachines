# components/charts.py — Chart builders (Matplotlib + Plotly)
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from config import PALETTE as P, FEATURE_LABELS


# ─── Matplotlib global defaults ───────────────────────────────────────────────

def _mpl_dark():
    plt.rcParams.update({
        "font.family"          : "monospace",
        "axes.facecolor"       : P["panel"],
        "figure.facecolor"     : P["bg"],
        "text.color"           : P["text"],
        "axes.edgecolor"       : P["grid"],
        "xtick.color"          : P["dim_text"],
        "ytick.color"          : P["dim_text"],
        "grid.color"           : P["grid"],
        "axes.labelcolor"      : P["dim_text"],
        "axes.titlecolor"      : P["text"],
    })


# ─── Main SVM scatter ─────────────────────────────────────────────────────────

def svm_scatter(
    result, 
    X_all: np.ndarray,
    y_all: np.ndarray, symbol: str,
    live_point_2d=None
) :-> plt.Figure:
    """
    2-D PCA projection of feature space with SVM decision boundary,
    margin lines, support-vector rings, and coloured region fills.
    """
    _mpl_dark()

    # ── PCA reduction ──
    from core.model import pca_2d
    X_2d, pca, evr = pca_2d(result, X_all)

    # Fit a 2-D SVM purely for boundary rendering
    vis = SVC(kernel=result.kernel, C=result.C,
              gamma="scale", random_state=42)
    vis.fit(X_2d, y_all)

    # Mesh
    pad  = 0.7
    x0, x1 = X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad
    y0, y1 = X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad
    xx, yy  = np.meshgrid(np.linspace(x0, x1, 400),
                           np.linspace(y0, y1, 400))
    Z = vis.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(11, 7), facecolor=P["bg"])
    ax.set_facecolor(P["bg"])

    # Region fills
    ax.contourf(xx, yy, Z, levels=[-np.inf, 0],
                colors=[P["bull_fill"]], alpha=0.60)
    ax.contourf(xx, yy, Z, levels=[0, np.inf],
                colors=[P["bear_fill"]], alpha=0.60)

    # Margin zone
    ax.contourf(xx, yy, Z, levels=[-1, 1],
                colors=[P["boundary"]], alpha=0.07)

    # Boundary lines
    ax.contour(xx, yy, Z, levels=[-1], colors=[P["margin_line"]],
               linestyles="--", linewidths=1.8, alpha=0.9)
    ax.contour(xx, yy, Z, levels=[0],  colors=[P["boundary"]],
               linestyles="-",  linewidths=3.0, alpha=1.0)
    ax.contour(xx, yy, Z, levels=[1],  colors=[P["margin_line"]],
               linestyles="--", linewidths=1.8, alpha=0.9)

    # Data points
    bull = y_all == 1
    bear = y_all == 0
    ax.scatter(X_2d[bull, 0], X_2d[bull, 1],
               c=P["bull_dot"], s=60, alpha=0.80, edgecolors="none", zorder=3)
    ax.scatter(X_2d[bear, 0], X_2d[bear, 1],
               c=P["bear_dot"], s=60, alpha=0.80, edgecolors="none", zorder=3)

    # Support vectors
    sv = vis.support_
    ax.scatter(X_2d[sv, 0], X_2d[sv, 1],
               s=180, facecolors="none",
               edgecolors=P["sv_ring"], linewidths=2.4, zorder=4)

    # Labels
    ax.text(x0 + 0.15, y1 - 0.4, "▲  BULLISH",
            color=P["bull_dot"], fontsize=13, fontweight="bold")
    ax.text(x1 - 2.2,  y1 - 0.4, "▼  BEARISH",
            color=P["bear_dot"], fontsize=13, fontweight="bold")
    ax.text((x0 + x1) / 2, y0 + 0.2, "── Decision Boundary ──",
            color=P["boundary"], fontsize=8.5, ha="center", alpha=0.9)

    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel(f"PC1  ({evr[0]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC2  ({evr[1]:.1%} var)", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    ax.set_title(
        f"{symbol}  ·  Kernel={result.kernel.upper()}  "
        f"C={result.C}  ·  Support Vectors: {len(sv)}",
        fontsize=12, pad=10, color=P["text"],
    )

    
    # ── Live market dot ──────────────────────────────────────────────
    if live_point_2d is not None:
        lx, ly = float(live_point_2d[0]), float(live_point_2d[1])
        # Outer glow ring
        ax.scatter(lx, ly, s=420, facecolors="none",
                   edgecolors="#ffffff", linewidths=2.0,
                   zorder=6, alpha=0.5)
        # Inner coloured ring (bull/bear colour based on decision function)
        df_val = vis.decision_function([[lx, ly]])[0]
        dot_color = P["bull_dot"] if df_val > 0 else P["bear_dot"]
        ax.scatter(lx, ly, s=220, facecolors="none",
                   edgecolors=dot_color, linewidths=3.0, zorder=7)
        # Solid centre dot
        ax.scatter(lx, ly, s=80, c="#ffffff",
                   edgecolors="none", zorder=8)
        # Label
        side = "BULL" if df_val > 0 else "BEAR"
        ax.annotate(
            f"◀ LIVE  {side}  (dist={df_val:+.2f})",
            xy=(lx, ly),
            xytext=(lx + 0.25, ly + 0.25),
            color="#ffffff",
            fontsize=8.5,
            fontfamily="monospace",
            arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.2),
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=dot_color, alpha=0.7, edgecolor="none"),
        )
    
    if live_point_2d is not None:
        try:
            lx = float(live_point_2d[0])
            ly = float(live_point_2d[1])
            df_val     = float(vis.decision_function([[lx, ly]])[0])
            dot_color  = P["bull_dot"] if df_val > 0 else P["bear_dot"]
            side_label = "BULL" if df_val > 0 else "BEAR"
            ax.scatter(lx, ly, s=480, facecolors="none",
                       edgecolors="#ffffff", linewidths=1.5,
                       zorder=6, alpha=0.35)
            ax.scatter(lx, ly, s=260, facecolors="none",
                       edgecolors=dot_color, linewidths=2.8, zorder=7)
            ax.scatter(lx, ly, s=90, c="#ffffff",
                       edgecolors="none", zorder=8)
            ax.annotate(
                f"◀ LIVE  {side_label}  ({df_val:+.2f})",
                xy=(lx, ly),
                xytext=(lx + 0.35, ly + 0.35),
                color="#ffffff", fontsize=8, fontfamily="monospace",
                arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1.2),
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=dot_color,
                          alpha=0.75, edgecolor="none"),
            )
        except Exception:
            pass

    handles = [
        mpatches.Patch(color=P["bull_dot"], label=f"Bull days ({bull.sum()})"),
        mpatches.Patch(color=P["bear_dot"], label=f"Bear days ({bear.sum()})"),
        mpatches.Patch(facecolor="none", edgecolor=P["sv_ring"],
                       label=f"Support vectors ({len(sv)})"),
        mpatches.Patch(color=P["boundary"],  label="Decision boundary"),
        mpatches.Patch(color=P["margin_line"], label="Margin ±1"),
    ]
    if live_point_2d is not None:
        handles.append(
            mpatches.Patch(facecolor="none", edgecolor="#ffffff",
                           label="Live market position")
        )
    ax.legend(handles=handles, loc="lower right",
              facecolor=P["panel"], edgecolor=P["grid"],
              labelcolor=P["text"], fontsize=9)

    plt.tight_layout()
    return fig


# ─── Probability histogram ────────────────────────────────────────────────────

def prob_histogram(y_prob_all: np.ndarray, y_all: np.ndarray) -> go.Figure:
    """Stacked histogram of P(Bull) for bull vs bear days."""
    fig = go.Figure()
    for label, name, color in [(1, "Bull days", P["bull_dot"]),
                                (0, "Bear days", P["bear_dot"])]:
        mask = y_all == label
        fig.add_trace(go.Histogram(
            x=y_prob_all[mask], name=name,
            nbinsx=25, opacity=0.75,
            marker_color=color,
            histnorm="probability density",
        ))
    fig.add_vline(x=0.5, line_dash="dash", line_color=P["boundary"],
                  annotation_text="Decision threshold",
                  annotation_font_color=P["boundary"])
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor=P["bg"], plot_bgcolor=P["panel"],
        font_color=P["text"], font_family="monospace",
        title="P(Bull) Distribution",
        xaxis_title="Predicted P(Bull)",
        yaxis_title="Density",
        legend=dict(bgcolor=P["panel"], bordercolor=P["grid"]),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
    )
    _plotly_dark_axes(fig)
    return fig


# ─── Confusion matrix heatmap ─────────────────────────────────────────────────

def confusion_heatmap(cm: np.ndarray) -> go.Figure:
    labels = ["Bear", "Bull"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18, color=P["bg"]),
        colorscale=[[0, "#111820"], [1, P["boundary"]]],
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix (Test Set)",
        paper_bgcolor=P["bg"], plot_bgcolor=P["panel"],
        font_color=P["text"], font_family="monospace",
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
    )
    _plotly_dark_axes(fig)
    return fig


# ─── Feature importance bar ───────────────────────────────────────────────────

def feature_importance_bar(importances: np.ndarray) -> go.Figure:
    """Horizontal bar chart of permutation importance scores."""
    top_n   = 12
    top_idx = np.argsort(importances)[-top_n:]
    vals    = importances[top_idx]
    names   = [FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f"F{i}"
               for i in top_idx]
    colors  = [P["bull_dot"] if v >= 0 else P["bear_dot"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors, opacity=0.85,
    ))
    fig.update_layout(
        title="Feature Importance (Permutation)",
        paper_bgcolor=P["bg"], plot_bgcolor=P["panel"],
        font_color=P["text"], font_family="monospace",
        xaxis_title="Importance score",
        margin=dict(l=110, r=20, t=50, b=40),
        height=380,
    )
    _plotly_dark_axes(fig)
    return fig


# ─── Price + signal overlay ───────────────────────────────────────────────────

def price_signal_chart(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    dates: pd.DatetimeIndex,
    symbol: str,
) -> go.Figure:
    """
    Candlestick + P(Bull) probability ribbon.
    Dates and y_prob correspond to the feature-engineered subset of df.
    """
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        increasing_line_color=P["bull_dot"],
        decreasing_line_color=P["bear_dot"],
        name="OHLC",
    ))

    # P(Bull) area (secondary y)
    fig.add_trace(go.Scatter(
        x=dates, y=y_prob,
        mode="lines", fill="tozeroy",
        name="P(Bull)",
        line=dict(color=P["boundary"], width=1.5),
        fillcolor="rgba(255,140,0,0.15)",
        yaxis="y2",
    ))
    fig.add_hline(y=0.5, line_dash="dot",
                  line_color=P["sv_ring"], line_width=1,
                  annotation_text="0.50", annotation_font_color=P["sv_ring"],
                  annotation_font_size=9)

    fig.update_layout(
        title=f"{symbol} — Price & SVM P(Bull) Signal",
        paper_bgcolor=P["bg"], plot_bgcolor=P["panel"],
        font_color=P["text"], font_family="monospace",
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price ₹", side="left"),
        yaxis2=dict(title="P(Bull)", overlaying="y", side="right",
                    range=[0, 1], showgrid=False,
                    tickformat=".0%"),
        legend=dict(bgcolor=P["panel"], bordercolor=P["grid"]),
        margin=dict(l=60, r=60, t=50, b=40),
        height=450,
    )
    _plotly_dark_axes(fig)
    return fig


# ─── C sensitivity plot ───────────────────────────────────────────────────────

def c_sensitivity_chart(c_vals: list, acc_list: list, sv_list: list) -> go.Figure:
    """Dual-axis: test accuracy and support-vector count vs. C."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c_vals, y=acc_list, name="Test Accuracy",
        mode="lines+markers",
        line=dict(color=P["bull_dot"], width=2),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        x=c_vals, y=sv_list, name="Support Vectors",
        mode="lines+markers",
        line=dict(color=P["sv_ring"], width=2, dash="dot"),
        marker=dict(size=7),
        yaxis="y2",
    ))
    fig.update_layout(
        title="C Sensitivity  —  Accuracy vs. Support Vectors",
        xaxis=dict(title="C value", type="log"),
        yaxis=dict(title="Test Accuracy", tickformat=".1%"),
        yaxis2=dict(title="# Support Vectors", overlaying="y", side="right"),
        paper_bgcolor=P["bg"], plot_bgcolor=P["panel"],
        font_color=P["text"], font_family="monospace",
        legend=dict(bgcolor=P["panel"], bordercolor=P["grid"]),
        margin=dict(l=60, r=60, t=50, b=40),
        height=350,
    )
    _plotly_dark_axes(fig)
    return fig


# ─── Helper ───────────────────────────────────────────────────────────────────

def _plotly_dark_axes(fig: go.Figure):
    fig.update_xaxes(gridcolor=P["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=P["grid"], zeroline=False)
