# app.py — SVM Margin Visualiser · Landing Page
import streamlit as st
from config import APP_ICON, APP_TITLE, APP_VERSION, PALETTE as P

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Hero ──────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style="
        background:{P['panel']};
        border:1px solid {P['grid']};
        border-left:4px solid {P['boundary']};
        border-radius:10px;
        padding:32px 36px 24px;
        font-family:monospace;
        margin-bottom:28px;
    ">
      <div style="font-size:2.4rem;font-weight:900;
                  color:{P['boundary']};letter-spacing:3px;">
        📡 SVM MARGIN VISUALISER
      </div>
      <div style="color:{P['dim_text']};font-size:0.9rem;margin-top:6px;">
        Zerodha Kite Connect  ·  NSE  ·  Bull / Bear Regime Classification
        <span style="float:right;color:{P['grid']};font-size:0.75rem;">v{APP_VERSION}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Architecture overview ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

for col, icon, title, body in [
    (c1, "🔐", "Auth",       "Enter your Kite API key and daily access token once. Credentials live in session state only."),
    (c2, "📊", "Visualiser", "Train SVM on 252-session NSE data. Explore decision boundary, probability ribbon, C sweep."),
    (c3, "📈", "Signals",    "Batch-scan your watchlist. See Bull/Bear regime signal and market breadth in one table."),
    (c4, "🎓", "Learn",      "Toy sandbox with synthetic data. Build intuition about kernels, margin width, and support vectors."),
]:
    col.markdown(
        f"""
        <div style="
            background:{P['panel']};border:1px solid {P['grid']};
            border-radius:8px;padding:18px 16px;height:160px;
            font-family:monospace;
        ">
          <div style="font-size:1.6rem;">{icon}</div>
          <div style="color:{P['boundary']};font-weight:700;
                      font-size:1rem;margin:6px 0 4px;">{title}</div>
          <div style="color:{P['dim_text']};font-size:0.82rem;
                      line-height:1.5;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Feature matrix table ─────────────────────────────────────────────────
st.divider()
st.subheader("📐  Feature Matrix  —  20 Technical Indicators")

import pandas as pd
from config import FEATURE_LABELS

feat_meta = [
    ("Return 1d",    "Price",      "1-session log-return"),
    ("Return 3d",    "Price",      "3-session momentum"),
    ("Return 5d",    "Price",      "5-session momentum"),
    ("Return 10d",   "Price",      "10-session momentum"),
    ("Return 20d",   "Price",      "Monthly momentum"),
    ("Vol 5d",       "Volatility", "Rolling 5-day σ of returns"),
    ("Vol 10d",      "Volatility", "Rolling 10-day σ"),
    ("Vol 20d",      "Volatility", "Rolling 20-day σ"),
    ("RSI-14",       "Oscillator", "Relative Strength Index (14)"),
    ("MACD",         "Trend",      "12/26 EMA spread"),
    ("MACD Hist",    "Trend",      "MACD minus signal line"),
    ("BB Position",  "Band",       "(close − SMA20) / (2 × σ20)"),
    ("Dist EMA21",   "MA",         "% distance from 21-EMA"),
    ("Dist SMA50",   "MA",         "% distance from 50-SMA"),
    ("Dist SMA200",  "MA",         "% distance from 200-SMA"),
    ("Stoch %K",     "Oscillator", "14-period Stochastic"),
    ("Williams %R",  "Oscillator", "14-period Williams %R"),
    ("Vol Ratio",    "Volume",     "Volume / 20-day avg volume"),
    ("OBV Slope",    "Volume",     "On-balance volume momentum"),
    ("ATR %",        "Volatility", "14-period ATR as % of close"),
]

df_feat = pd.DataFrame(feat_meta, columns=["Feature", "Category", "Description"])
df_feat.index = range(1, len(df_feat) + 1)
st.dataframe(
    df_feat.style
    .applymap(lambda v: f"color:{P['boundary']};font-weight:600", subset=["Feature"])
    .applymap(lambda v: f"color:{P['accent']}", subset=["Category"]),
    use_container_width=True,
    height=600,
)

# ─── Quick-start guide ────────────────────────────────────────────────────
st.divider()
st.subheader("🚀  Quick Start")
st.markdown(f"""
1. **Copy** `.env.example` → `.env` and fill in your Kite API Key and Access Token  
2. Run the app:  `streamlit run app.py`  
3. Go to **🔐 Auth** in the sidebar — verify your session  
4. Go to **📊 Visualiser** — pick a symbol, adjust kernel & C, click **Train SVM**  
5. Explore the five analysis tabs; download the chart PNG  

```bash
# First-time setup (macOS / Linux)
git clone <repo>
cd svm_visualiser
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in credentials
streamlit run app.py
```
""")

# ─── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://zerodha.com/static/images/logo.svg",
    use_column_width=True,
)
st.sidebar.caption("Navigate using the pages above.")
st.sidebar.divider()
st.sidebar.markdown(
    f"**{APP_TITLE}**  `v{APP_VERSION}`\n\n"
    "Built with Streamlit + scikit-learn + Kite Connect"
)
