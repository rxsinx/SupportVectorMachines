# core/features.py — Feature engineering for SVM regime classification
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """
    Build a 20-indicator feature matrix from raw OHLCV data.

    Label definition
    ----------------
    label = 1 (Bull)  if  close[t + forward_days] > close[t]
    label = 0 (Bear)  otherwise

    Feature order matches config.FEATURE_LABELS.
    """
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)

    # ── 1-5  Price momentum ──────────────────────────────────────────
    for n in (1, 3, 5, 10, 20):
        f[f"ret_{n}d"] = c.pct_change(n)

    # ── 6-8  Rolling volatility ──────────────────────────────────────
    ret1 = c.pct_change(1)
    for n in (5, 10, 20):
        f[f"vol_{n}d"] = ret1.rolling(n).std()

    # ── 9  RSI-14 ────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    f["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ── 10-11  MACD + histogram ──────────────────────────────────────
    ema12    = c.ewm(span=12, adjust=False).mean()
    ema26    = c.ewm(span=26, adjust=False).mean()
    macd     = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    f["macd"]      = macd
    f["macd_hist"] = macd - macd_sig

    # ── 12  Bollinger Band position ──────────────────────────────────
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f["bb_pos"] = (c - sma20) / (2 * std20 + 1e-9)

    # ── 13-15  Distance from key MAs ─────────────────────────────────
    f["dist_ema21"]  = (c - c.ewm(span=21, adjust=False).mean()) / c
    f["dist_sma50"]  = (c - c.rolling(50).mean()) / c
    f["dist_sma200"] = (c - c.rolling(200).mean()) / c

    # ── 16-17  Stochastic %K & Williams %R ──────────────────────────
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    span = hi14 - lo14 + 1e-9
    f["stoch_k"] = 100 * (c - lo14) / span
    f["willr"]   = -100 * (hi14 - c) / span

    # ── 18-19  Volume momentum ───────────────────────────────────────
    vol_avg20   = v.rolling(20).mean() + 1e-9
    f["vol_ratio"]  = v / vol_avg20
    f["obv_slope"]  = (
        (v * np.sign(c.diff())).rolling(10).sum()
        / (v.rolling(10).mean() + 1e-9)
    )

    # ── 20  ATR % ────────────────────────────────────────────────────
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c

    # ── Label ────────────────────────────────────────────────────────
    f["label"] = (c.shift(-forward_days) > c).astype(int)

    return f.dropna()


def get_xy(feat_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split feature DataFrame into X (features) and y (labels)."""
    FEAT_COLS = [col for col in feat_df.columns if col != "label"]
    return feat_df[FEAT_COLS].values, feat_df["label"].values


def train_test_split_temporal(
    X: np.ndarray, y: np.ndarray, test_ratio: float = 0.20
) -> tuple[np.ndarray, ...]:
    """Chronological split — NO shuffling (avoids look-ahead bias)."""
    n     = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]
