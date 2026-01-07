import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_datareader.data as web


# -----------------------------
# App Configuration
# -----------------------------

APP_TITLE = "Market Overview"
APP_ICON = "📈"

TICKERS: Dict[str, str] = {
    "S&P 500": "^GSPC",
    "DXY (Dollar Index)": "DX-Y.NYB",
    "Crude Oil (WTI)": "CL=F",
    "Bitcoin": "BTC-USD",
}

FRED_SERIES: List[str] = ["DGS2", "DGS5", "DGS10", "DGS30", "PAYEMS", "CPIAUCSL", "PPIACO"]

# Default cache TTLs (tune based on how “live” you want it)
DEFAULT_MARKET_TTL_SEC = 120     # Yahoo prices: 2 min
DEFAULT_FRED_TTL_SEC = 3600      # FRED: 1 hr (rates update daily; CPI/PPI monthly)

# Per-user refresh throttle to avoid “Refresh spam”
MIN_REFRESH_INTERVAL_SEC = 15


@dataclass(frozen=True)
class MarketMetric:
    price: float
    pct_change: float


# -----------------------------
# Helpers
# -----------------------------

def _now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _pct_delta(pct: float) -> str:
    return f"{pct:+.2f}%"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_close_prices(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Robustly extract Close prices from yfinance output across versions.

    yfinance output can be:
    - MultiIndex columns: (PriceField, Ticker) or (Ticker, PriceField)
    - Single-index columns for one ticker
    """
    if raw is None or raw.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers)

    cols = raw.columns

    # Case 1: MultiIndex columns
    if isinstance(cols, pd.MultiIndex):
        # Try common layouts:
        # (field, ticker)
        if "Close" in cols.get_level_values(0):
            closes = raw["Close"]
            # closes columns are tickers
            return closes.reindex(columns=tickers)
        # (ticker, field)
        if "Close" in cols.get_level_values(1):
            closes = raw.xs("Close", level=1, axis=1)
            return closes.reindex(columns=tickers)

    # Case 2: Single index
    # If single ticker, raw might have "Close" column
    if "Close" in raw.columns and len(tickers) == 1:
        return raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Fallback: try to find columns matching tickers
    existing = [c for c in tickers if c in raw.columns]
    if existing:
        return raw[existing]

    return pd.DataFrame(index=raw.index, columns=tickers)


# -----------------------------
# Cached Data Fetching
# -----------------------------

@st.cache_data(ttl=DEFAULT_MARKET_TTL_SEC, show_spinner=False)
def fetch_yahoo_closes(ticker_list: List[str]) -> pd.DataFrame:
    """
    Fetch recent close prices for tickers.

    Cached across sessions on the same Streamlit server process to reduce load.
    """
    raw = yf.download(
        tickers=ticker_list,
        period="5d",
        progress=False,
        auto_adjust=False,
        threads=True,
    )
    return _extract_close_prices(raw, ticker_list)


def compute_market_snapshot() -> Dict[str, MarketMetric]:
    """
    Returns {Display Name: MarketMetric(price, pct_change)} for configured tickers.
    """
    ticker_list = list(TICKERS.values())
    closes = fetch_yahoo_closes(ticker_list)

    out: Dict[str, MarketMetric] = {}
    for name, ticker in TICKERS.items():
        s = closes[ticker].dropna() if (closes is not None and ticker in closes.columns) else pd.Series(dtype=float)

        if len(s) >= 2:
            current = _safe_float(s.iloc[-1])
            prev = _safe_float(s.iloc[-2])
            pct = ((current - prev) / prev) * 100 if prev and pd.notna(prev) else 0.0
            out[name] = MarketMetric(price=current, pct_change=pct)
        elif len(s) == 1:
            current = _safe_float(s.iloc[-1])
            out[name] = MarketMetric(price=current, pct_change=0.0)
        else:
            out[name] = MarketMetric(price=0.0, pct_change=0.0)

    return out


@st.cache_data(ttl=DEFAULT_FRED_TTL_SEC, show_spinner=False)
def fetch_fred_series(series_ids: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    return web.DataReader(series_ids, "fred", start, end)


def compute_fred_snapshot() -> Optional[Dict[str, object]]:
    """
    Returns a dict with yields, curve spread, and econ metrics.
    None if FRED fails.
    """
    end = datetime.now()
    start = end - timedelta(days=365)

    try:
        df = fetch_fred_series(FRED_SERIES, start, end)
    except Exception:
        return None

    data: Dict[str, object] = {}

    # Treasuries (latest)
    def latest(series_name: str) -> float:
        s = df[series_name].dropna()
        return _safe_float(s.iloc[-1]) if not s.empty else float("nan")

    y2 = latest("DGS2")
    y5 = latest("DGS5")
    y10 = latest("DGS10")
    y30 = latest("DGS30")

    data["2Y"] = y2
    data["5Y"] = y5
    data["10Y"] = y10
    data["30Y"] = y30

    # 2s10s in bps (yields are in %)
    data["2s10s_bps"] = (y10 - y2) * 100.0 if pd.notna(y2) and pd.notna(y10) else float("nan")

    # NFP (PAYEMS) - thousands
    payems = df["PAYEMS"].dropna()
    data["NFP"] = f"{int(payems.iloc[-1]):,} k" if not payems.empty else "N/A"

    # CPI YoY (approx, compare to 12 months ago)
    cpi = df["CPIAUCSL"].dropna()
    if len(cpi) > 12:
        cpi_yoy = ((cpi.iloc[-1] / cpi.iloc[-13]) - 1) * 100
        data["CPI_YoY"] = f"{cpi_yoy:.2f}%"
    else:
        data["CPI_YoY"] = "N/A"

    # PPI YoY (approx, compare to 12 months ago)
    ppi = df["PPIACO"].dropna()
    if len(ppi) > 12:
        ppi_yoy = ((ppi.iloc[-1] / ppi.iloc[-13]) - 1) * 100
        data["PPI_YoY"] = f"{ppi_yoy:.2f}%"
    else:
        data["PPI_YoY"] = "N/A"

    return data


# -----------------------------
# Refresh Logic (throttled)
# -----------------------------

def can_refresh() -> Tuple[bool, str]:
    last = st.session_state.get("last_refresh_epoch", 0.0)
    elapsed = time.time() - last
    if elapsed < MIN_REFRESH_INTERVAL_SEC:
        return False, f"Please wait {int(MIN_REFRESH_INTERVAL_SEC - elapsed)}s before refreshing again."
    return True, ""


def do_refresh():
    ok, msg = can_refresh()
    if not ok:
        st.toast(msg, icon="⏳")
        return

    # Clear caches so next run refetches
    fetch_yahoo_closes.clear()
    fetch_fred_series.clear()

    st.session_state["last_refresh_epoch"] = time.time()
    st.session_state["last_updated_str"] = _now_local_str()


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon=APP_ICON)

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-size: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)

top = st.columns([1, 1, 2])
with top[0]:
    st.button("Refresh Data 🔄", type="primary", on_click=do_refresh)
with top[1]:
    st.caption("Uses cached API calls to handle multiple users efficiently.")
with top[2]:
    last_updated = st.session_state.get("last_updated_str", _now_local_str())
    st.caption(f"Last updated: {last_updated}")

st.divider()

# 1) Major Markets
st.subheader("Major Markets")

with st.spinner("Loading market data…"):
    try:
        market = compute_market_snapshot()
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        market = {}

c1, c2, c3, c4 = st.columns(4)

with c1:
    m = market.get("S&P 500", MarketMetric(0.0, 0.0))
    st.metric("S&P 500", f"{m.price:,.2f}", _pct_delta(m.pct_change))

with c2:
    m = market.get("DXY (Dollar Index)", MarketMetric(0.0, 0.0))
    st.metric("DXY", f"{m.price:,.2f}", _pct_delta(m.pct_change))

with c3:
    m = market.get("Crude Oil (WTI)", MarketMetric(0.0, 0.0))
    st.metric("Crude Oil", f"${m.price:,.2f}", _pct_delta(m.pct_change))

with c4:
    m = market.get("Bitcoin", MarketMetric(0.0, 0.0))
    st.metric("Bitcoin (BTC)", f"${m.price:,.0f}", _pct_delta(m.pct_change))

st.divider()

# 2) Rates & Curve
st.subheader("US Treasury Curve")

with st.spinner("Loading rates & macro data…"):
    fred = compute_fred_snapshot()

if fred is None:
    st.warning("Could not connect to FRED right now. Try refreshing.")
else:
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("2 Year", f"{fred['2Y']:.2f}%" if pd.notna(fred["2Y"]) else "N/A")
    with r2:
        st.metric("5 Year", f"{fred['5Y']:.2f}%" if pd.notna(fred["5Y"]) else "N/A")
    with r3:
        st.metric("10 Year", f"{fred['10Y']:.2f}%" if pd.notna(fred["10Y"]) else "N/A")
    with r4:
        st.metric("30 Year", f"{fred['30Y']:.2f}%" if pd.notna(fred["30Y"]) else "N/A")
    with r5:
        spread = fred["2s10s_bps"]
        st.metric("2s/10s Spread", f"{spread:.1f} bps" if pd.notna(spread) else "N/A")

st.divider()

# 3) Economic Data
st.subheader("Latest Economic Releases")

if fred is None:
    st.info("Economic tiles will appear when FRED is available.")
else:
    e1, e2, e3 = st.columns(3)
    with e1:
        st.metric("NFP (Total Nonfarm)", fred["NFP"], help="Total employees, nonfarm (thousands)")
    with e2:
        st.metric("CPI (YoY)", fred["CPI_YoY"], help="Consumer Price Index for All Urban Consumers")
    with e3:
        st.metric("PPI (YoY)", fred["PPI_YoY"], help="Producer Price Index for All Commodities")

st.caption(f"Rendered at: {_now_local_str()}")
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
