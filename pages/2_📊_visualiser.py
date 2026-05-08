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

    # ── Cache trained objects for live refresh ───────────────────────────
    from core.model import pca_2d
    from sklearn.svm import SVC
    
    X_2d_full, _pca_obj, _evr = pca_2d(result, X)
    _vis2d = SVC(kernel=result.kernel, C=result.C, gamma="scale", random_state=42)
    _vis2d.fit(X_2d_full, y)
    
    st.session_state.update({
        "live_result":  result,
        "live_X":       X,
        "live_y":       y,
        "live_df":      df.copy(),
        "live_symbol":  symbol,
        "live_params":  params,
        "live_pca":     _pca_obj,
        "live_vis2d":   _vis2d,
    })

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
        "Orange line = hyperplane · dashed = ±1 margin · rings = support vectors · "
        "⚪ white dot = live market position (market hours only)."
    )

    # ── Live mode controls ─────────────────────────────────────────
    from utils.helpers import nse_market_open

    live_col1, live_col2, live_col3 = st.columns([1, 1, 3])
    live_mode     = live_col1.toggle("🔴 Live Mode", value=False,
                                      help="Poll Kite LTP every N seconds and "
                                           "show current market position on chart")
    refresh_secs  = live_col2.selectbox("Refresh", [5, 10, 30], index=1,
                                         label_visibility="collapsed")
    market_open   = nse_market_open()

    if live_mode and not market_open:
        st.warning("NSE is currently closed (09:15–15:30 IST Mon–Fri). "
                   "Live mode will show last available LTP.")

    # ── Compute live point if live mode is on ──────────────────────
    live_point_2d = None
    live_signal_text = ""

    if live_mode and "live_pca" in st.session_state:
        try:
            from core.kite_client import fetch_ohlcv, get_ltp
            from core.features import engineer_features, get_xy

            _kite      = get_kite_session(api_key, access_token)
            _sym       = st.session_state["live_symbol"]
            _df_hist   = st.session_state["live_df"].copy()
            _pca       = st.session_state["live_pca"]
            _res       = st.session_state["live_result"]

            # Get current LTP
            current_ltp = get_ltp(_kite, _sym)

            # Splice LTP into last row of OHLCV as synthetic "current" close
            _df_live = _df_hist.copy()
            _df_live.iloc[-1, _df_live.columns.get_loc("close")] = current_ltp
            _df_live.iloc[-1, _df_live.columns.get_loc("high")] = max(
                _df_live.iloc[-1]["high"], current_ltp)
            _df_live.iloc[-1, _df_live.columns.get_loc("low")] = min(
                _df_live.iloc[-1]["low"], current_ltp)

            # Recompute features on updated df
            _feat_live = engineer_features(
                _df_live,
                forward_days=st.session_state["live_params"]["forward_days"]
            )
            _X_live, _ = get_xy(_feat_live)

            if len(_X_live) > 0:
                # Scale with the trained scaler, project with stored PCA
                _x_latest_sc = _res.scaler.transform(_X_live[-1:])
                live_point_2d = _pca.transform(_x_latest_sc)[0]

                # Signal from full 20-D model
                _sig = float(_res.model.predict_proba(_x_latest_sc)[0, 1])
                _label = "▲ BULL" if _sig > 0.5 else "▼ BEAR"
                live_signal_text = (
                    f"**Live LTP:** ₹{current_ltp:,.2f}  ·  "
                    f"**Signal:** {_label}  ·  "
                    f"**P(Bull):** {_sig:.1%}"
                )

        except Exception as _e:
            st.warning(f"Live data error: {_e}")

    # ── Render chart ───────────────────────────────────────────────
    chart_placeholder = st.empty()

    with chart_placeholder:
        with st.spinner("Rendering SVM scatter …"):
            fig_scatter = svm_scatter(
                result, X, y, symbol,
                live_point_2d=live_point_2d,
            )
        st.pyplot(fig_scatter, use_container_width=True)

    if live_signal_text:
        st.markdown(live_signal_text)

    # ── Auto-refresh loop ──────────────────────────────────────────
    if live_mode:
        import time as _time
        _time.sleep(refresh_secs)
        st.rerun()


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
        "Each feature is shuffled; drop in accuracy estimates importance. "
        "Cached after first run."
    )

    @st.cache_data(show_spinner=False)
    def _get_importance(_result, _X, _y):
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(
            _result.model,
            _result.scaler.transform(_X),
            _y,
            n_repeats=3,        # reduced from 5 → faster
            random_state=42,
            n_jobs=-1,
        )
        return pi.importances_mean

    with st.spinner("Computing permutation importance …"):
        importances = _get_importance(result, X, y)
    st.plotly_chart(feature_importance_bar(importances), use_container_width=True)
    
# ── Tab 5 — C sensitivity ──

with tab5:
    st.subheader("Margin Width Sensitivity  (C sweep)")
    st.caption("Sweeps C on a log scale. Cached after first run.")

    # Reduced sweep — 8 points instead of 12
    c_vals = sorted({0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 50.0, 100.0, params["C"]})

    @st.cache_data(show_spinner=False)
    def _c_sweep(_X_tr, _y_tr, _X_te, _y_te, _X, kernel, gamma, degree, c_vals):
        acc_list, sv_list = [], []
        for c_val in c_vals:
            r = train_svm(_X_tr, _y_tr, _X_te, _y_te, _X,
                          kernel=kernel, C=c_val,
                          gamma=gamma, degree=degree)
            acc_list.append(r.acc_test)
            sv_list.append(r.n_sv)
        return acc_list, sv_list

    with st.spinner("Sweeping C values …"):
        acc_list, sv_list = _c_sweep(
            X_tr, y_tr, X_te, y_te, X,
            params["kernel"], params["gamma"], params["degree"],
            tuple(c_vals),
        )
    st.plotly_chart(c_sensitivity_chart(c_vals, acc_list, sv_list),
                    use_container_width=True)


# ─── Save model ──────────────────────────────────────────────────────────────
if save_model_flag:
    path = save_model(result, symbol)
    st.sidebar.success(f"Model saved → `{path}`")

