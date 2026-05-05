# pages/1_🔐_auth.py — Kite Connect authentication
import streamlit as st
from config import APP_ICON, APP_TITLE
from core.kite_client import get_credentials_from_env, get_kite_session

st.set_page_config(
    page_title=f"Auth — {APP_TITLE}",
    page_icon=APP_ICON,
    layout="centered",
)

st.title("🔑  Kite Connect — Authentication")
st.caption("Your credentials are stored only in Streamlit session state "
           "and are never written to disk from this page.")

# ── Auto-load from .env ──────────────────────────────────────────────────────
auto_key, auto_token = get_credentials_from_env()
if auto_key and auto_token:
    st.success("✔  Credentials detected in `.env` file.  "
               "You can use them directly or override below.")

# ── Manual entry ─────────────────────────────────────────────────────────────
with st.form("kite_auth_form"):
    st.markdown("#### Enter Credentials")
    api_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.get("api_key", auto_key or ""),
        placeholder="e.g. abcdef1234567890",
    )
    access_token = st.text_input(
        "Access Token",
        type="password",
        value=st.session_state.get("access_token", auto_token or ""),
        placeholder="Paste your daily access token here",
    )
    submitted = st.form_submit_button("🔗  Connect to Kite", type="primary",
                                       use_container_width=True)

if submitted:
    if not api_key or not access_token:
        st.error("Both API Key and Access Token are required.")
    else:
        with st.spinner("Verifying with Kite Connect …"):
            try:
                kite = get_kite_session(api_key, access_token)
                profile = kite.profile()
                st.session_state["api_key"]      = api_key
                st.session_state["access_token"] = access_token
                st.session_state["kite_profile"] = profile
                st.success(
                    f"✔  Connected as **{profile.get('user_name', 'Unknown')}** "
                    f"({profile.get('email', '')})"
                )
                st.balloons()
            except Exception as exc:
                st.error(f"Connection failed: {exc}")

# ── Profile display ──────────────────────────────────────────────────────────
if "kite_profile" in st.session_state:
    prof = st.session_state["kite_profile"]
    st.divider()
    st.subheader("Session Profile")
    cols = st.columns(3)
    cols[0].metric("Name",    prof.get("user_name", "—"))
    cols[1].metric("Broker",  prof.get("broker",    "—"))
    cols[2].metric("User ID", prof.get("user_id",   "—"))

# ── How to get credentials ───────────────────────────────────────────────────
with st.expander("ℹ  How to get your Kite API credentials"):
    st.markdown("""
1. Log in at [kite.trade/connect/apps](https://kite.trade/connect/apps)
2. Create an app (or use an existing one) — note the **API Key** and **API Secret**
3. On each trading day, generate a new **Access Token** via the Kite login flow  
   *(Access tokens expire at midnight IST)*
4. Paste both values above, or save them in your `.env` file:

```bash
KITE_API_KEY=your_key_here
KITE_ACCESS_TOKEN=your_token_here
```
""")
