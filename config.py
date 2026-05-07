# config.py — Central configuration for SVM Margin Visualiser
from __future__ import annotations

# ─── Colour palette ────────────────────────────────────────────────────────────
PALETTE = dict(
    bg          = "#0b0f14",
    panel       = "#111820",
    grid        = "#1a2230",
    bull_fill   = "#0d2b1a",
    bear_fill   = "#2b0d0d",
    bull_dot    = "#00e676",
    bear_dot    = "#ff1744",
    sv_ring     = "#ffc107",
    boundary    = "#ff8c00",
    margin_line = "#ff6b00",
    text        = "#cdd6e0",
    dim_text    = "#4a5568",
    accent      = "#00b0ff",
)

# ─── SVM knobs ─────────────────────────────────────────────────────────────────
KERNEL_OPTIONS   = ["rbf", "linear", "poly", "sigmoid"]
C_RANGE          = (0.01, 100.0)
C_DEFAULT        = 1.0
GAMMA_OPTIONS    = ["scale", "auto"]
DEGREE_OPTIONS   = [2, 3, 4, 5]
DAYS_MIN         = 60
DAYS_MAX         = 504
DAYS_DEFAULT     = 252
TEST_RATIO_DEFAULT = 0.20

# ─── Popular NSE tickers for quick-pick ────────────────────────────────────────
# Add after QUICK_PICK_SYMBOLS
INDEX_MAP = {
    "NIFTY50":     "NSE:NIFTY 50",
    "BANKNIFTY":   "NSE:NIFTY BANK",
    "FINNIFTY":    "NSE:NIFTY FIN SERVICE",

}

QUICK_PICK_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "WIPRO", "HINDUNILVR", "SBIN", "BAJFINANCE", "MARUTI",
    "AXISBANK", "KOTAKBANK", "LT", "TITAN", "ASIANPAINT",
    "NIFTY50", "BANKNIFTY", "FINNIFTY",
]

# ─── Feature labels (matches features.py order) ────────────────────────────────
FEATURE_LABELS = [
    "Return 1d", "Return 3d", "Return 5d", "Return 10d", "Return 20d",
    "Vol 5d",    "Vol 10d",   "Vol 20d",
    "RSI-14",    "MACD",      "MACD Hist",
    "BB Position",
    "Dist EMA21", "Dist SMA50", "Dist SMA200",
    "Stoch %K",  "Williams %R",
    "Vol Ratio", "OBV Slope", "ATR %",
]

# ─── Intervals available on Kite Connect ───────────────────────────────────────
KITE_INTERVALS = ["day", "60minute", "30minute", "15minute"]

APP_TITLE   = "SVM Margin Visualiser"
APP_ICON    = "📡"
APP_VERSION = "1.0.0"
