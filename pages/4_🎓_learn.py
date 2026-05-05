# pages/4_🎓_learn.py — Interactive SVM Education Module
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config import APP_ICON, APP_TITLE, PALETTE as P

st.set_page_config(
    page_title=f"Learn SVM — {APP_TITLE}",
    page_icon=APP_ICON,
    layout="wide",
)

st.title("🎓  Understanding SVM Margins")
st.caption(
    "An interactive sandbox — no Kite credentials required. "
    "Adjust the controls to build intuition about how SVMs classify regimes."
)

# ─── Controls ────────────────────────────────────────────────────────────────
col_ctrl, col_chart = st.columns([1, 2])

with col_ctrl:
    st.markdown("#### Synthetic dataset")
    n_bull = st.slider("Bull samples",  30, 200, 80)
    n_bear = st.slider("Bear samples",  30, 200, 80)
    noise  = st.slider("Class overlap (σ)", 0.1, 3.0, 0.8, step=0.1)
    seed   = st.number_input("Random seed", 0, 999, 42)

    st.markdown("#### SVM settings")
    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
    C_val  = st.slider("C (margin ↔ regularisation)",
                       0.01, 50.0, 1.0, step=0.1, format="%.2f")
    gamma_opt = st.selectbox("Gamma", ["scale", "auto", "custom"])
    gamma_val = 0.1
    if gamma_opt == "custom":
        gamma_val = st.number_input("Custom γ", 0.001, 10.0, 0.1, format="%.3f")
    gamma = gamma_val if gamma_opt == "custom" else gamma_opt

    degree = 3
    if kernel == "poly":
        degree = st.slider("Polynomial degree", 2, 5, 3)

# ─── Generate data ───────────────────────────────────────────────────────────
rng   = np.random.default_rng(int(seed))
X_bull = rng.normal(loc=[-1.5, 1.5], scale=noise, size=(n_bull, 2))
X_bear = rng.normal(loc=[ 1.5,-1.5], scale=noise, size=(n_bear, 2))
X_syn  = np.vstack([X_bull, X_bear])
y_syn  = np.array([1] * n_bull + [0] * n_bear)

# ─── Fit 2-D SVM ─────────────────────────────────────────────────────────────
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_syn)
svm     = SVC(kernel=kernel, C=C_val, gamma=gamma, degree=degree,
              probability=True, random_state=42)
svm.fit(X_sc, y_syn)

# ─── Mesh ─────────────────────────────────────────────────────────────────────
pad  = 1.0
x0   = X_sc[:, 0].min() - pad;  x1 = X_sc[:, 0].max() + pad
y0   = X_sc[:, 1].min() - pad;  y1 = X_sc[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
Z  = svm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ─── Plot ─────────────────────────────────────────────────────────────────────
fig = go.Figure()

# Region fills
fig.add_trace(go.Contour(
    x=np.linspace(x0, x1, 300), y=np.linspace(y0, y1, 300), z=Z,
    contours=dict(start=-1e9, end=0, size=1e9),
    colorscale=[[0, P["bull_fill"]], [1, P["bull_fill"]]],
    showscale=False, opacity=0.55, name="Bull zone",
))
fig.add_trace(go.Contour(
    x=np.linspace(x0, x1, 300), y=np.linspace(y0, y1, 300), z=Z,
    contours=dict(start=0, end=1e9, size=1e9),
    colorscale=[[0, P["bear_fill"]], [1, P["bear_fill"]]],
    showscale=False, opacity=0.55, name="Bear zone",
))

# Margin lines
for level, style, lw in [(-1, "dash", 2), (0, "solid", 3), (1, "dash", 2)]:
    fig.add_trace(go.Contour(
        x=np.linspace(x0, x1, 300), y=np.linspace(y0, y1, 300), z=Z,
        contours_coloring="lines",
        contours=dict(start=level, end=level, size=0.01),
        line=dict(
            color=P["boundary"] if level == 0 else P["margin_line"],
            width=lw, dash=style,
        ),
        showscale=False,
        name="Decision boundary" if level == 0 else f"Margin {level:+d}",
    ))

# Data points
fig.add_trace(go.Scatter(
    x=X_sc[y_syn == 1, 0], y=X_sc[y_syn == 1, 1],
    mode="markers", name=f"Bull ({n_bull})",
    marker=dict(color=P["bull_dot"], size=9, opacity=0.8),
))
fig.add_trace(go.Scatter(
    x=X_sc[y_syn == 0, 0], y=X_sc[y_syn == 0, 1],
    mode="markers", name=f"Bear ({n_bear})",
    marker=dict(color=P["bear_dot"], size=9, opacity=0.8),
))

# Support vectors
sv_idx = svm.support_
fig.add_trace(go.Scatter(
    x=X_sc[sv_idx, 0], y=X_sc[sv_idx, 1],
    mode="markers", name=f"Support vectors ({len(sv_idx)})",
    marker=dict(size=15, color="rgba(0,0,0,0)",
                line=dict(color=P["sv_ring"], width=2.5)),
))

fig.update_layout(
    title=f"SVM  kernel={kernel.upper()}  C={C_val}  γ={gamma}  "
          f"| Support vectors: {len(sv_idx)}",
    paper_bgcolor=P["bg"], plot_bgcolor=P["panel"],
    font_color=P["text"], font_family="monospace",
    xaxis=dict(range=[x0, x1], gridcolor=P["grid"], zeroline=False),
    yaxis=dict(range=[y0, y1], gridcolor=P["grid"], zeroline=False),
    legend=dict(bgcolor=P["panel"], bordercolor=P["grid"]),
    height=520,
    margin=dict(l=20, r=20, t=50, b=20),
)

with col_chart:
    st.plotly_chart(fig, use_container_width=True)

# ─── Concept explainers ───────────────────────────────────────────────────────
st.divider()
st.subheader("📖  SVM Concepts — applied to market regimes")

tabs = st.tabs([
    "What is SVM?", "Margin & C", "Kernels",
    "Support Vectors", "Bull / Bear Labels",
])

with tabs[0]:
    st.markdown("""
**Support Vector Machine** finds the *widest corridor* (margin) that separates
Bull days from Bear days in a high-dimensional indicator space.

- Days that sit on the margin boundary are **support vectors** — the critical
  observations that define the boundary.
- All other points could be removed and the model would stay identical.
- In market terms: SVM identifies the *archetypal* bullish and bearish indicator
  configurations, then draws a rule around them.
""")

with tabs[1]:
    st.markdown("""
**Margin** = the gap between the boundary and the nearest Bull / Bear points.

| C value | Margin | Effect |
|---------|--------|--------|
| Small (0.1) | Wide | Tolerates misclassifications — more robust, less overfit |
| C = 1.0 | Balanced | Good default starting point |
| Large (50+) | Narrow | Fits every training point — risk of overfitting noise |

**Intuition for NSE data:**  
Markets are noisy. A *wide margin* (low C) tends to generalise better to
unseen sessions than a tight boundary that memorises historical data.
""")

with tabs[2]:
    st.markdown("""
| Kernel | Shape | Best for |
|--------|-------|---------|
| **RBF** | Radial / circular blobs | Most market regimes — non-linear, robust |
| **Linear** | Flat hyperplane | When indicators have a clear linear trend |
| **Poly** (d=3) | Curved polynomial | Captures interaction effects (e.g. RSI × momentum) |
| **Sigmoid** | S-curve | Neural-net analogue; useful for probability-like indicators |

**Trick:** Start with RBF, check test accuracy, then try linear as a baseline.
If RBF >> linear, your features have meaningful non-linear structure.
""")

with tabs[3]:
    st.markdown("""
**Support vectors** are the training days that sit exactly on the margin edge
(decision function value = ±1).

- Fewer SVs → tighter, more decisive boundary
- Many SVs → boundary "samples" most of the dataset; may indicate overlap

**Ring markers** in the chart highlight support-vector sessions.  
Mouse over them to see the corresponding trading date.
""")

with tabs[4]:
    st.markdown("""
**Label construction in this app:**

```
label[t] = 1 (Bull)  if  close[t + 5] > close[t]
label[t] = 0 (Bear)  otherwise
```

The **5-session forward return** horizon is adjustable (sidebar → "Label horizon").

Implications:
- The model learns *which indicator configurations precede a 5-day up-move*
- Label noise increases with shorter horizons (3d) and decreases with longer (10d)
- The last 5 rows are dropped because their future returns are unknown
""")
