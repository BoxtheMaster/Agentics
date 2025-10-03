# mcp_news_marketaux.py
"""
MCP Server — MarketAux stock news
=================================
Tool: get_news_mx(ticker: str, count: int = 10, days: int = 7, language: str = "en")

Env:
  export MARKETAUX_API_KEY="your_marketaux_key"

Install:
  pip install mcp requests

Run:
  python mcp_news_marketaux.py
"""

from __future__ import annotations
from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP
import os, requests

mcp = FastMCP("MarketAuxNews")

def _clean_item(x: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": x.get("title"),
        "url": x.get("url"),
        "source": x.get("source"),
        "published_at": x.get("published_at"),
        "description": x.get("description"),
        "symbols": x.get("symbols"),            # list of tickers MarketAux tags
        "entities": x.get("entities"),          # optional enrichment
    }

@mcp.tool()
def get_news_mx(ticker: str, count: int = 10, days: int = 7, language: str = "en"):
    """
    Fetch recent news for a ticker from MarketAux.
    Args:
      ticker: e.g., "AAPL"
      count: max number of items (1..50)
      days: lookback window in days (1..30)
      language: "en" (default) or others supported by MarketAux
    """
    api_key = os.getenv("MARKETAUX_API_KEY")
    if not api_key:
        raise RuntimeError("MARKETAUX_API_KEY not set. export MARKETAUX_API_KEY=...")

    t = ticker.upper().strip()
    n = max(1, min(50, int(count)))
    d = max(1, min(30, int(days)))

    # MarketAux: https://api.marketaux.com/v1/news/all
    # Common params: symbols, filter_entities=true (to enrich), group_similar, limit, published_after
    params = {
        "symbols": t,
        "language": language,
        "filter_entities": "true",
        "limit": n,
        "api_token": api_key,
        # You can also pass 'published_after' in ISO if you want exact windows.
        # With 'days' we can bias freshness via 'group_similar' or leave it simple:
        "group_similar": "true",
    }
    r = requests.get("https://api.marketaux.com/v1/news/all", params=params, timeout=15)
    j = r.json() if r.content else {}

    data = j.get("data", []) if isinstance(j, dict) else []
    items = [_clean_item(x or {}) for x in data]

    return {
        "ticker": t,
        "count": len(items),
        "items": items,
        "params_used": {"language": language, "limit": n, "group_similar": True, "days": d},
    }

if __name__ == "__main__":
    # Important: no prints to stdout—MCP stdio must be clean.
    mcp.run(transport="stdio")