# components/sidebar.py — Shared sidebar controls
from __future__ import annotations

import streamlit as st
from config import (
    C_DEFAULT, C_RANGE, DAYS_DEFAULT, DAYS_MAX, DAYS_MIN,
    DEGREE_OPTIONS, GAMMA_OPTIONS, KERNEL_OPTIONS,
    QUICK_PICK_SYMBOLS, TEST_RATIO_DEFAULT,
)


def render_credential_form() -> tuple[str, str]:
    """
    Sidebar credential section.
    Returns (api_key, access_token) — may be empty strings.
    """
    st.sidebar.markdown("### 🔑 Kite Credentials")
    with st.sidebar.expander("Enter API credentials", expanded=False):
        api_key = st.text_input(
            "API Key", type="password", key="sb_api_key",
            value=st.session_state.get("api_key", ""),
        )
        access_token = st.text_input(
            "Access Token", type="password", key="sb_access_token",
            value=st.session_state.get("access_token", ""),
        )
        if st.button("Connect", key="sb_connect"):
            st.session_state["api_key"]      = api_key
            st.session_state["access_token"] = access_token
            st.rerun()
    return (
        st.session_state.get("api_key", ""),
        st.session_state.get("access_token", ""),
    )


def render_symbol_picker() -> str:
    """Quick-pick + free-text symbol selector."""
    st.sidebar.markdown("### 📌 Symbol")
    quick = st.sidebar.selectbox(
        "Quick pick", ["(type below)"] + QUICK_PICK_SYMBOLS,
        key="sb_quick_symbol",
    )
    manual = st.sidebar.text_input(
        "Or type NSE ticker", key="sb_manual_symbol",
        placeholder="e.g. HDFCBANK",
    ).strip().upper()

    symbol = manual if manual else (
        quick if quick != "(type below)" else "RELIANCE"
    )
    return symbol


def render_svm_controls() -> dict:
    """
    Render kernel / C / gamma / degree / days sliders.
    Returns a dict with all SVM hyper-parameters.
    """
    st.sidebar.markdown("### ⚙️ SVM Parameters")

    kernel = st.sidebar.selectbox(
        "Kernel", KERNEL_OPTIONS,
        index=0, key="sb_kernel",
        help="rbf handles non-linear boundaries; linear is interpretable.",
    )

    C = st.sidebar.slider(
        "C  (margin width ↔ regularisation)",
        min_value=float(C_RANGE[0]),
        max_value=float(C_RANGE[1]),
        value=float(C_DEFAULT),
        step=0.05,
        key="sb_C",
        help="Low C → wide margin (softer).  High C → narrow margin (tighter fit).",
        format="%.2f",
    )

    gamma_str = st.sidebar.selectbox(
        "Gamma", GAMMA_OPTIONS + ["custom"], key="sb_gamma_type",
    )
    if gamma_str == "custom":
        gamma = st.sidebar.number_input(
            "Custom gamma", min_value=0.0001, max_value=10.0,
            value=0.1, step=0.01, key="sb_gamma_val", format="%.4f",
        )
    else:
        gamma = gamma_str

    degree = 3
    if kernel == "poly":
        degree = st.sidebar.selectbox(
            "Polynomial degree", DEGREE_OPTIONS, index=1, key="sb_degree",
        )

    st.sidebar.markdown("### 📅 Data Window")
    days = st.sidebar.slider(
        "Training sessions (trading days)",
        min_value=DAYS_MIN,
        max_value=DAYS_MAX,
        value=DAYS_DEFAULT,
        step=21,
        key="sb_days",
        help="252 ≈ 1 calendar year of NSE trading sessions.",
    )

    test_ratio = st.sidebar.slider(
        "Test set %",
        min_value=0.05, max_value=0.40,
        value=TEST_RATIO_DEFAULT,
        step=0.05, key="sb_test_ratio",
        format="%.0%%",
    )

    forward_days = st.sidebar.slider(
        "Label horizon (forward days)",
        min_value=1, max_value=20,
        value=5, key="sb_fwd",
        help="Number of future sessions used to define Bull/Bear label.",
    )

    return dict(
        kernel=kernel, C=C, gamma=gamma,
        degree=degree, days=days,
        test_ratio=test_ratio,
        forward_days=forward_days,
    )


def render_connection_status(connected: bool, symbol: str | None = None):
    """Small status badge in the sidebar."""
    if connected:
        st.sidebar.success(f"✔ Kite Connected{' · ' + symbol if symbol else ''}")
    else:
        st.sidebar.error("✘ Not connected — enter credentials above")
