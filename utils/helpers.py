# utils/helpers.py — Shared utilities
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime

import streamlit as st


@contextmanager
def timer(label: str):
    """Context manager that shows elapsed time in an st.toast."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    st.toast(f"{label} — {elapsed:.2f}s", icon="⏱")


def format_inr(value: float) -> str:
    """Format a number as Indian Rupees with ₹ symbol."""
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    return f"₹{value:,.2f}"


def pct_delta(new: float, old: float) -> str:
    """Return coloured percentage delta string."""
    pct = (new - old) / abs(old) * 100 if old else 0.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def session_key(*parts: str) -> str:
    """Build a namespaced session-state key."""
    return "__".join(str(p) for p in parts)


def now_ist() -> str:
    """Current datetime as IST string (no pytz dependency)."""
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")


def nse_market_open() -> bool:
    """
    Returns True if current IST time is within NSE regular session
    (09:15 – 15:30, Mon-Fri).  Ignores holidays.
    """
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


def compute_permutation_importance(
    result,
    X_all,
    y_all,
    n_repeats: int = 5,
) -> "np.ndarray":
    """
    Run sklearn permutation importance on the full dataset.
    Returns array of mean importance scores.
    """
    import numpy as np
    from sklearn.inspection import permutation_importance

    pi  = permutation_importance(
        result.model,
        result.scaler.transform(X_all),
        y_all,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    return pi.importances_mean
