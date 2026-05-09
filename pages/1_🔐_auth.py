# pages/1_🔐_auth.py
from __future__ import annotations
import streamlit as st
from config import APP_ICON, APP_TITLE
from core.kite_client import (
    generate_access_token,
    get_credentials_from_env,
    get_kite_session,
    get_login_url,
)

st.set_page_config(page_title=f"Auth — {APP_TITLE}", page_icon=APP_ICON, layout="centered")

st.title("🔑  Kite Connect — Authentication")

env         = get_credentials_from_env()
url_params  = st.query_params
auto_rtoken = url_params.get("request_token", "")
auto_status = url_params.get("status", "")

if auto_rtoken and auto_status == "success":
    st.success("✔  request_token detected in URL — scroll to Step 3.")
elif auto_status == "error":
    st.error("Kite login failed or was cancelled. Please try again.")

# ── STEP 1 ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 1 — API Key & Secret")

col1, col2 = st.columns(2)
with col1:
    api_key = st.text_input(
        "API Key", type="password",
        value=st.session_state.get("api_key", env["api_key"] or ""),
        placeholder="abcdef1234567890",
    )
with col2:
    api_secret = st.text_input(
        "API Secret", type="password",
        value=st.session_state.get("api_secret", env["api_secret"] or ""),
        placeholder="••••••••••••••••",
    )

if api_key:    st.session_state["api_key"]    = api_key
if api_secret: st.session_state["api_secret"] = api_secret

# ── STEP 2 ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 2 — Log in via Kite")

if st.button("🔗  Generate Login URL", type="primary",
             disabled=not (api_key and api_secret)):
    try:
        st.session_state["login_url"] = get_login_url(api_key)
    except Exception as exc:
        st.error(f"Could not build login URL: {exc}")

if "login_url" in st.session_state:
    url = st.session_state["login_url"]
    st.markdown(
        f'<a href="{url}" target="_blank">'
        f'<button style="background:#ff8c00;color:#0b0f14;border:none;'
        f'padding:10px 20px;border-radius:6px;font-weight:700;cursor:pointer;">'
        f'↗  Open Kite Login</button></a>',
        unsafe_allow_html=True,
    )
    st.code(url, language=None)

# ── STEP 3 ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Step 3 — Generate Access Token")

request_token = st.text_input(
    "request_token  (from redirect URL)",
    value=auto_rtoken or st.session_state.get("request_token", ""),
    placeholder="Paste request_token here …",
)
if request_token:
    st.session_state["request_token"] = request_token

if st.button("✅  Generate Session", type="primary",
             disabled=not (api_key and api_secret and request_token)):
    with st.spinner("Exchanging request_token for access_token …"):
        try:
            access_token, session_data = generate_access_token(
                api_key, api_secret, request_token
            )
            kite = get_kite_session(api_key, access_token)   # verify + cache

            st.session_state["access_token"]  = access_token
            st.session_state["kite_profile"]  = session_data
            st.session_state["auth_complete"] = True
            st.query_params.clear()

            st.success(
                f"✔  Session active for "
                f"**{session_data.get('user_name','?')}** "
                f"({session_data.get('email','')})"
            )
            st.balloons()

        except Exception as exc:
            st.error(
                f"**Session generation failed:** {exc}\n\n"
                "Common causes:\n"
                "- request_token already used (single-use) — log in again\n"
                "- API Secret incorrect\n"
                "- request_token older than 5 minutes"
            )

# ── Active session panel ──────────────────────────────────────────────
if st.session_state.get("auth_complete"):
    prof = st.session_state.get("kite_profile", {})
    st.markdown("---")
    st.subheader("✅  Active Session")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Name",    prof.get("user_name", "—"))
    c2.metric("User ID", prof.get("user_id",   "—"))
    c3.metric("Broker",  prof.get("broker",    "—"))
    c4.metric("Email",   prof.get("email",     "—"))

    if st.button("🚪  Clear Session"):
        for k in ["api_key","api_secret","access_token",
                  "kite_profile","auth_complete","login_url","request_token"]:
            st.session_state.pop(k, None)
        st.rerun()

# ── .env shortcut ─────────────────────────────────────────────────────
elif env.get("access_token"):
    st.markdown("---")
    st.info("A KITE_ACCESS_TOKEN was found in your .env file. Click to verify it.")
    if st.button("⚡  Use token from .env"):
        with st.spinner("Verifying …"):
            try:
                kite    = get_kite_session(env["api_key"], env["access_token"])
                profile = kite.profile()
                st.session_state.update({
                    "api_key": env["api_key"],
                    "api_secret": env.get("api_secret",""),
                    "access_token": env["access_token"],
                    "kite_profile": profile,
                    "auth_complete": True,
                })
                st.success(f"✔  Logged in as **{profile.get('user_name','?')}**")
                st.rerun()
            except Exception as exc:
                st.error(f"Token invalid or expired: {exc}")

# ── Reference ─────────────────────────────────────────────────────────
with st.expander("📖  How Kite OAuth works"):
    st.code("""
Step 1  You provide:  API Key + API Secret   (from kite.trade/connect/apps)

Step 2  App builds:   login_url = kite.login_url()
        You visit →   log in with Zerodha credentials + 2FA
        Kite redirects to your Redirect URL with:
                      ?request_token=XXXX&status=success

Step 3  App calls:    kite.generate_session(request_token, api_secret)
        Returns:      access_token  ← valid until midnight IST
                      + user_name, email, user_id …

        request_token = SINGLE USE. If Step 3 fails, repeat Step 2.
        access_token  = expires MIDNIGHT IST. Repeat daily.
""", language=None)
