# pages/2_📊_visualiser.py — Core SVM Margin Visualiser
from __future__ import annotations

import streamlit as st
import numpy as np

from config import APP_ICON, APP_TITLE
from components.sidebar import (
    render_connection_status, render_credential_form,
    render_svm_controls, render_symbol_picker,
)
from components.charts import (
    svm_scatter, prob_histogram, confusion_heatmap,
    feature_importance_bar, price_signal_chart, c_sensitivity_chart,
)
from components.metrics import (
    metric_row, signal_badge, classification_report_table, kernel_explainer,
)
from core.features import engineer_features, get_xy, train_test_split_temporal
from core.model import train_svm, predict_latest, save_model
from utils.helpers import compute_permutation_importance, now_ist, timer

st.set_page_config(
    page_title=f"Visualiser — {APP_TITLE}",
    page_icon=APP_ICON,
    layout="wide",
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
api_key, access_token = render_credential_form()
symbol  = render_symbol_picker()
params  = render_svm_controls()

connected = bool(api_key and access_token)
render_connection_status(connected, symbol if connected else None)

# ─── Page header ─────────────────────────────────────────────────────────────
st.title("📡  SVM Margin Visualiser")
st.caption(f"Zerodha Kite Connect  ·  NSE  ·  {now_ist()}")

kernel_explainer(params["kernel"])

if not connected:
    st.warning("Enter your Kite Connect credentials in the sidebar to begin.")
    st.stop()

# ─── Run button ──────────────────────────────────────────────────────────────
run_col, save_col = st.columns([3, 1])
run = run_col.button(
    f"▶  Train SVM  —  {symbol} / {params['kernel'].upper()} / C={params['C']}",
    type="primary", use_container_width=True,
)
save_model_flag = save_col.checkbox("💾 Save model", value=False)

if not run:
    st.info("Configure parameters in the sidebar and click **Train SVM**.")
    st.stop()

# ─── Data fetch ──────────────────────────────────────────────────────────────
try:
    from core.kite_client import get_kite_session, fetch_ohlcv

    with st.spinner(f"Connecting to Kite …"):
        kite = get_kite_session(api_key, access_token)

    with st.spinner(f"Fetching {params['days']}-session history for {symbol} …"):
        with timer("Data fetch"):
            df = fetch_ohlcv(kite, symbol, days=params["days"])

    st.success(
        f"✔  {len(df)} sessions  "
        f"({df.index[0].date()} → {df.index[-1].date()})"
    )

except Exception as exc:
    st.error(f"Data fetch failed: {exc}")
    st.stop()

# ─── Feature engineering ─────────────────────────────────────────────────────
with st.spinner("Engineering 20-indicator feature matrix …"):
    feat_df  = engineer_features(df, forward_days=params["forward_days"])
    X, y     = get_xy(feat_df)
    dates    = feat_df.index
    X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y, params["test_ratio"])

st.caption(
    f"Features: {X.shape[1]} indicators  ·  "
    f"Train: {len(X_tr)}  ·  Test: {len(X_te)}  ·  "
    f"Bull: {y.sum()}  Bear: {len(y)-y.sum()}"
)

# ─── Train ───────────────────────────────────────────────────────────────────
with st.spinner(
    f"Training SVM  (kernel={params['kernel']}  C={params['C']}  "
    f"γ={params['gamma']}) …"
):
    with timer("SVM training"):
        result = train_svm(
            X_tr, y_tr, X_te, y_te, X,
            kernel=params["kernel"],
            C=params["C"],
            gamma=params["gamma"],
            degree=params["degree"],
        )

# ─── KPI row ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Performance")
metric_row(result)

# ─── Latest signal ───────────────────────────────────────────────────────────
st.divider()
st.subheader("📍  Latest Regime Signal")
latest_x      = X[-1]
signal_info   = predict_latest(result, latest_x)
sig_col, _pad = st.columns([1, 2])
with sig_col:
    signal_badge(signal_info)

# ─── Main tabs ───────────────────────────────────────────────────────────────
st.divider()
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 SVM Boundary",
    "📈 Price & Signal",
    "📊 Distributions",
    "🔬 Feature Importance",
    "⚗️  C Sensitivity",
])

# ── Tab 1 — Decision boundary scatter ──
with tab1:
    st.subheader("SVM Decision Boundary (PCA 2-D Projection)")
    st.caption(
        "The scatter projects the 20-D feature space onto 2 principal components. "
        "The orange line is the hyperplane; dashed lines are the ±1 margin. "
        "Rings = support vectors."
    )
    with st.spinner("Rendering SVM scatter …"):
        fig_scatter = svm_scatter(result, X, y, symbol)
    st.pyplot(fig_scatter, use_container_width=True)

# ── Tab 2 — Price + P(Bull) overlay ──
with tab2:
    st.subheader("Price Chart + SVM Probability Ribbon")
    fig_price = price_signal_chart(df, result.y_prob_all, dates, symbol)
    st.plotly_chart(fig_price, use_container_width=True)

# ── Tab 3 — Distributions ──
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("P(Bull) Distribution")
        st.plotly_chart(
            prob_histogram(result.y_prob_all, y),
            use_container_width=True,
        )
    with col_b:
        st.subheader("Confusion Matrix")
        st.plotly_chart(
            confusion_heatmap(result.cm),
            use_container_width=True,
        )

    st.subheader("Classification Report")
    classification_report_table(result.report)

# ── Tab 4 — Feature importance ──
with tab4:
    st.subheader("Permutation Feature Importance")
    st.caption(
        "Each feature is shuffled independently; the drop in accuracy estimates its importance. "
        "Uses the full dataset (train + test)."
    )
    with st.spinner("Computing permutation importance (5 repeats) …"):
        importances = compute_permutation_importance(result, X, y)
    st.plotly_chart(
        feature_importance_bar(importances),
        use_container_width=True,
    )

# ── Tab 5 — C sensitivity ──
with tab5:
    st.subheader("Margin Width Sensitivity  (C sweep)")
    st.caption(
        "Sweep C across a log-scale range and show how test accuracy and "
        "support-vector count change.  Your current C is highlighted."
    )
    c_vals = sorted(set([
        0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0
    ] + [params["C"]]))

    acc_list, sv_list = [], []
    prog = st.progress(0, text="Sweeping C …")
    for i, c_val in enumerate(c_vals):
        r = train_svm(X_tr, y_tr, X_te, y_te, X,
                      kernel=params["kernel"], C=c_val,
                      gamma=params["gamma"], degree=params["degree"])
        acc_list.append(r.acc_test)
        sv_list.append(r.n_sv)
        prog.progress((i + 1) / len(c_vals), text=f"C = {c_val}")
    prog.empty()

    st.plotly_chart(
        c_sensitivity_chart(c_vals, acc_list, sv_list),
        use_container_width=True,
    )

# ─── Save model ──────────────────────────────────────────────────────────────
if save_model_flag:
    path = save_model(result, symbol)
    st.sidebar.success(f"Model saved → `{path}`")
