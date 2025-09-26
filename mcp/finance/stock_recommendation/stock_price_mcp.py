# mcp_yfinance_simple.py
"""
Tool: stock_summary(ticker: str)

Returns:
- last_open, last_close, prev_close
- day_high, day_low, volume
- 52-week high/low
- 1-month high/low

Install:
  pip install mcp yfinance pandas

Run:
  python mcp_yfinance_simple.py
"""

from __future__ import annotations
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd

mcp = FastMCP("YFinanceSimple")

@mcp.tool()
def stock_summary(ticker: str) -> dict:
    """
    Fetch stock summary data for a ticker.
    Args:
      ticker: stock symbol, e.g. "AAPL"
    """
    t = ticker.upper().strip()

    df = yf.download(t, period="2d", interval="1d", progress=False)
    if df.empty:
        return {"ticker": t, "error": "No data returned"}

    last_open = float(df["Open"].iloc[-1])
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
    day_high = float(df["High"].iloc[-1])
    day_low = float(df["Low"].iloc[-1])
    last_vol = int(df["Volume"].iloc[-1])

    # 52-week high/low
    df_1y = yf.download(t, period="1y", interval="1d", progress=False)
    wk52_high = float(df_1y["High"].max()) if not df_1y.empty else None
    wk52_low = float(df_1y["Low"].min()) if not df_1y.empty else None

    # 1-month high/low
    df_1m = yf.download(t, period="1mo", interval="1d", progress=False)
    mo_high = float(df_1m["High"].max()) if not df_1m.empty else None
    mo_low = float(df_1m["Low"].min()) if not df_1m.empty else None

    return {
        "ticker": t,
        "last_open": last_open,
        "last_close": last_close,
        "prev_close": prev_close,
        "day_high": day_high,
        "day_low": day_low,
        "volume": last_vol,
        "52w_high": wk52_high,
        "52w_low": wk52_low,
        "1mo_high": mo_high,
        "1mo_low": mo_low,
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")