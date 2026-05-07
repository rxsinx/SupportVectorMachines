# pages/3_📈_signals.py — Live SVM Regime Signal Scanner
from __future__ import annotations

import time
import streamlit as st
import pandas as pd
import numpy as np

from config import APP_ICON, APP_TITLE, QUICK_PICK_SYMBOLS, PALETTE as P
from components.sidebar import render_credential_form, render_svm_controls
from core.kite_client import get_kite_session, fetch_ohlcv, get_ltp
from core.features import engineer_features, get_xy, train_test_split_temporal
from core.model import train_svm, predict_latest

st.set_page_config(
    page_title=f"Signal Scanner — {APP_TITLE}",
    page_icon=APP_ICON,
    layout="wide",
)

st.title("📡  Live Regime Signal Scanner")
st.caption(
    "Trains an SVM on 252-session history for each symbol and displays "
    "the current Bull / Bear regime signal."
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
api_key, access_token = render_credential_form()
params = render_svm_controls()

st.sidebar.markdown("### 🎯 Watchlist")
selected = st.sidebar.multiselect(
    "Pick symbols to scan",
    QUICK_PICK_SYMBOLS,
    default=QUICK_PICK_SYMBOLS[:6],
)
custom = st.sidebar.text_area(
    "Add custom symbols (one per line)",
    placeholder="TATAMOTORS\nHINDALCO",
)
if custom.strip():
    selected += [s.strip().upper() for s in custom.splitlines() if s.strip()]
selected = list(dict.fromkeys(selected))   # deduplicate, preserve order

auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 min", value=False)

if not api_key or not access_token:
    st.warning("Enter Kite credentials in the sidebar.")
    st.stop()

if not selected:
    st.info("Select at least one symbol in the sidebar.")
    st.stop()

# ─── Scan ────────────────────────────────────────────────────────────────────
if st.button("▶  Run Scanner", type="primary", use_container_width=True) or auto_refresh:
    kite = get_kite_session(api_key, access_token)
    results_rows = []
    prog = st.progress(0, text="Initialising …")

    for i, sym in enumerate(selected):
        prog.progress((i + 1) / len(selected), text=f"Processing {sym} …")
        try:
            df        = fetch_ohlcv(kite, sym, days=params["days"])
            ltp       = get_ltp(kite, sym)
            feat_df   = engineer_features(df, forward_days=params["forward_days"])
            X, y      = get_xy(feat_df)
            X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y, params["test_ratio"])
            result    = train_svm(
                X_tr, y_tr, X_te, y_te, X,
                kernel=params["kernel"], C=params["C"],
                gamma=params["gamma"],  degree=params["degree"],
            )
            sig        = predict_latest(result, X[-1])
            results_rows.append({
                "Symbol":       sym,
                "LTP":          ltp,
                "Signal":       sig["signal"],
                "P(Bull)":      sig["probability"],
                "Confidence":   sig["confidence_pct"] / 100,
                "Test Acc":     result.acc_test,
                "F1":           result.f1,
                "SVs":          result.n_sv,
            })
        except Exception as exc:
            results_rows.append({
                "Symbol": sym, "LTP": None,
                "Signal": "ERROR", "P(Bull)": None,
                "Confidence": None, "Test Acc": None,
                "F1": None, "SVs": None,
            })
            st.sidebar.warning(f"{sym}: {exc}")

    prog.empty()

    # ─── Results table ────────────────────────────────────────────────────
    df_res = pd.DataFrame(results_rows).set_index("Symbol")

    def color_signal(val):
        if val == "Bull":
            return f"color: {P['bull_dot']}; font-weight: bold"
        if val == "Bear":
            return f"color: {P['bear_dot']}; font-weight: bold"
        return "color: grey"

    styled = (
        df_res.style
        .map(color_signal, subset=["Signal"])
        .format({
            "LTP":        "₹{:,.2f}",
            "P(Bull)":    "{:.1%}",
            "Confidence": "{:.1%}",
            "Test Acc":   "{:.1%}",
            "F1":         "{:.2f}",
        }, na_rep="—")
        .background_gradient(subset=["P(Bull)"], cmap="RdYlGn")
        .background_gradient(subset=["Test Acc"], cmap="Blues")
    )

    st.subheader(f"Scan Results  ·  {len(results_rows)} symbols")
    st.dataframe(styled, use_container_width=True, height=420)

    # ─── Summary ──────────────────────────────────────────────────────────
    valid = df_res[df_res["Signal"].isin(["Bull", "Bear"])]
    bull_count = (valid["Signal"] == "Bull").sum()
    bear_count = (valid["Signal"] == "Bear").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Bull signals", bull_count)
    c2.metric("🔴 Bear signals", bear_count)
    c3.metric(
        "Market breadth",
        f"{bull_count / max(bull_count + bear_count, 1):.0%} bullish",
    )

    if auto_refresh:
        st.info("Auto-refresh in 5 minutes …")
        time.sleep(300)
        st.rerun()
