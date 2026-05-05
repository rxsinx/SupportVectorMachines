# 📡 SVM Margin Visualiser — Kite Connect Edition

> **Bull / Bear regime classification** using Support Vector Machines,  
> powered by live NSE data from Zerodha Kite Connect.

---

## Project Structure

```
svm_visualiser/
│
├── app.py                         # Landing page + quick-start guide
├── config.py                      # Palette, constants, feature labels
├── requirements.txt
├── run.sh                         # macOS Apple CLI launcher
├── .env.example                   # Credential template
│
├── .streamlit/
│   └── config.toml                # Dark theme, font, server settings
│
├── core/                          # Business logic (no Streamlit)
│   ├── __init__.py
│   ├── kite_client.py             # Kite Connect API wrapper + caching
│   ├── features.py                # 20-indicator feature engineering
│   └── model.py                   # SVM train / predict / save / load
│
├── components/                    # Reusable UI building blocks
│   ├── __init__.py
│   ├── sidebar.py                 # Credential form + SVM controls
│   ├── charts.py                  # Matplotlib + Plotly chart builders
│   └── metrics.py                 # KPI cards, signal badge, report table
│
├── pages/                         # Streamlit multi-page app
│   ├── 1_🔐_auth.py               # Kite Connect authentication
│   ├── 2_📊_visualiser.py         # Core SVM visualiser (5 tabs)
│   ├── 3_📈_signals.py            # Live watchlist scanner
│   └── 4_🎓_learn.py              # Interactive SVM education sandbox
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                 # Timer, INR formatter, IST clock, permutation importance
│
└── models/                        # Saved .pkl model files (auto-created)
```

---

## Quick Start

```bash
# 1. Clone and set up
git clone <repo-url>
cd svm_visualiser
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env — add your KITE_API_KEY and KITE_ACCESS_TOKEN

# 3. Run (macOS / Linux)
chmod +x run.sh && ./run.sh

# Alternative — direct Streamlit
streamlit run app.py
```

---

## Feature Matrix — 20 Indicators

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1–5 | Return 1/3/5/10/20d | Price | Rolling log-returns |
| 6–8 | Vol 5/10/20d | Volatility | Rolling σ of daily returns |
| 9 | RSI-14 | Oscillator | Relative Strength Index |
| 10–11 | MACD, MACD Hist | Trend | 12/26 EMA spread + histogram |
| 12 | BB Position | Band | (close − SMA20) / (2 × σ20) |
| 13–15 | Dist EMA21/SMA50/SMA200 | MA | % gap from key moving averages |
| 16–17 | Stoch %K, Williams %R | Oscillator | 14-period momentum oscillators |
| 18–19 | Vol Ratio, OBV Slope | Volume | Volume strength indicators |
| 20 | ATR % | Volatility | 14-period ATR as % of close |

**Label:** `1 (Bull)` if `close[t+5] > close[t]`, else `0 (Bear)`.  
Forward horizon is adjustable (1–20 sessions) in the sidebar.

---

## SVM Kernel Guide

| Kernel | Best For | Trade-off |
|--------|----------|-----------|
| **RBF** | Non-linear regime boundaries | Slower; tune γ |
| **Linear** | Interpretable; fast | Only linear separation |
| **Poly** | Interaction effects | Risk of overfitting with high degree |
| **Sigmoid** | Neural-net analogue | Can produce non-convex regions |

---

## Margin Width (C parameter)

```
C = 0.1   →  Wide margin  →  Tolerates misclassifications  →  Robust, general
C = 1.0   →  Balanced (default)
C = 10+   →  Narrow margin →  Fits training data tightly   →  Risk of overfit
```

---

## Pages

| Page | Description |
|------|-------------|
| **🔐 Auth** | Kite Connect session — paste API key + access token |
| **📊 Visualiser** | Full SVM analysis: boundary scatter, price ribbon, confusion matrix, feature importance, C sweep |
| **📈 Signals** | Batch watchlist scanner — Bull/Bear signal + market breadth |
| **🎓 Learn** | No-credentials sandbox with synthetic data + concept explainers |

---

## CLI Launcher Options

```bash
./run.sh                    # Default: port 8501, opens browser
./run.sh --port 8502        # Custom port
./run.sh --browser false    # Headless (server mode)
./run.sh --reset-venv       # Wipe and recreate venv
```

---

## Notes

- **Access tokens expire at midnight IST** — regenerate daily via Kite login flow.
- The `@st.cache_data(ttl=900)` on `fetch_ohlcv` caches OHLCV for 15 minutes.
- Saved models are stored as `.pkl` files in `models/` via `joblib`.
- The visualiser uses PCA to project the 20-D feature space to 2-D for plotting;  
  the actual SVM decision uses all 20 features.
