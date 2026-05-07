# core/kite_client.py
from __future__ import annotations
import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

_EXCHANGE = "NSE"


def build_unauthenticated_kite(api_key: str):
    try:
        from kiteconnect import KiteConnect
    except ImportError as exc:
        raise ImportError("Run: pip install kiteconnect") from exc
    return KiteConnect(api_key=api_key)


def get_login_url(api_key: str) -> str:
    return build_unauthenticated_kite(api_key).login_url()


def generate_access_token(api_key: str, api_secret: str, request_token: str):
    """
    Exchange request_token → access_token.
    Returns (access_token_str, session_dict).
    """
    kite = build_unauthenticated_kite(api_key)
    session = kite.generate_session(
        request_token=request_token.strip(),
        api_secret=api_secret.strip(),
    )
    return session["access_token"], session


@st.cache_resource(ttl=82800)   # 23 h — tokens expire midnight IST
def get_kite_session(api_key: str, access_token: str):
    from kiteconnect import KiteConnect
    kc = KiteConnect(api_key=api_key)
    kc.set_access_token(access_token)
    kc.profile()   # raises TokenException on stale token
    return kc


def resolve_token(kite, symbol: str) -> int:
    """
    Return the numeric instrument token for a symbol.
    Handles NSE indices (NIFTY50, BANKNIFTY) which need special exchange strings.
    """
    from config import INDEX_MAP
    # Use mapped exchange string if it's a known index, else default NSE
    exchange_key = INDEX_MAP.get(symbol.upper(), f"{_EXCHANGE}:{symbol}")
    data = kite.ltp([exchange_key])
    if exchange_key not in data:
        raise ValueError(
            f"Symbol '{symbol}' not found. "
            f"Tried key: '{exchange_key}'. Check the NSE ticker spelling."
        )
    return data[exchange_key]["instrument_token"]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(_kite, symbol: str, days: int = 252, interval: str = "day") -> pd.DataFrame:
    token     = resolve_token(_kite, symbol)
    to_date   = datetime.now()
    from_date = to_date - timedelta(days=int(days * 1.55))

    raw = _kite.historical_data(
        instrument_token=token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
    )
    if not raw:
        raise RuntimeError(f"No data returned for {symbol}.")

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
    from config import INDEX_MAP
    exchange_key = INDEX_MAP.get(symbol.upper(), f"{_EXCHANGE}:{symbol}")
    data = kite.ltp([exchange_key])
    return data[exchange_key]["last_price"]


def get_credentials_from_env() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return dict(
        api_key      = os.getenv("KITE_API_KEY"),
        api_secret   = os.getenv("KITE_API_SECRET"),
        access_token = os.getenv("KITE_ACCESS_TOKEN"),
    )

from kiteconnect import KiteTicker

def start_stream(api_key, access_token, instrument_token):
    kws = KiteTicker(api_key, access_token)

    def on_ticks(ws, ticks):
        # Update a global variable or a shared queue with the latest LTP
        st.session_state.latest_price = ticks[0]['last_price']

    kws.on_ticks = on_ticks
    kws.connect(threaded=True)
