# core/kite_client.py — Zerodha Kite Connect data layer
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

_EXCHANGE = "NSE"


@st.cache_resource(ttl=3600)
def get_kite_session(api_key: str, access_token: str):
    """
    Build and cache a KiteConnect session for the lifetime of the server process.
    Returns the KiteConnect object or raises on invalid credentials.
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise ImportError(
            "kiteconnect package not installed. Run:  pip install kiteconnect"
        ) from exc

    kc = KiteConnect(api_key=api_key)
    kc.set_access_token(access_token)
    # Lightweight sanity-check — will raise on bad token
    kc.profile()
    return kc


def resolve_token(kite, symbol: str) -> int:
    """Return the numeric instrument token for NSE:<symbol>."""
    key = f"{_EXCHANGE}:{symbol}"
    data = kite.ltp([key])
    if key not in data:
        raise ValueError(f"Symbol '{symbol}' not found on {_EXCHANGE}.")
    return data[key]["instrument_token"]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(
    _kite,          # leading underscore keeps st.cache_data from hashing the object
    symbol: str,
    days: int = 252,
    interval: str = "day",
) -> pd.DataFrame:
    """
    Pull OHLCV history from Kite Connect.

    Parameters
    ----------
    symbol   : NSE ticker, e.g. "RELIANCE"
    days     : number of *trading* sessions required
    interval : Kite interval string — "day", "60minute", etc.

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    indexed by timezone-naive datetime, sorted ascending.
    """
    token     = resolve_token(_kite, symbol)
    to_date   = datetime.now()
    # Buffer ≈ 1.5× to account for weekends / NSE holidays
    from_date = to_date - timedelta(days=int(days * 1.55))

    raw = _kite.historical_data(
        instrument_token=token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )

    if not raw:
        raise RuntimeError(
            f"No data returned for {symbol}. "
            "Check if the market is open or try a larger day range."
        )

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = (
        df.set_index("date")
        [["open", "high", "low", "close", "volume"]]
        .sort_index()
        .tail(days)
    )
    return df


def get_ltp(kite, symbol: str) -> float:
    """Return the last traded price for a single symbol."""
    key  = f"{_EXCHANGE}:{symbol}"
    data = kite.ltp([key])
    return data[key]["last_price"]


def get_credentials_from_env() -> tuple[str | None, str | None]:
    """Read API key + access token from environment / .env file."""
    from dotenv import load_dotenv
    load_dotenv()
    return (
        os.getenv("KITE_API_KEY"),
        os.getenv("KITE_ACCESS_TOKEN"),
    )
