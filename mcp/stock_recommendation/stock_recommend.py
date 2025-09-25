"""

• Single tool: `stock_recommend(ticker: str, days: int = 7, max_results: int = 8, risk: str = "medium")`
  - Fetches 30d price data via yfinance, computes 15‑day momentum
  - Pulls recent headlines via DuckDuckGo (DDGS)
  - Uses Gemini for headline sentiment (requires GEMINI_API_KEY)
  - Returns a BUY / HOLD / SELL **demo** recommendation with rationale + citations

Install:
  pip install mcp yfinance duckduckgo-search google-generativeai numpy pandas

Env:
  export GEMINI_API_KEY="..."
  export GEMINI_MODEL="gemini-1.5-flash"  # or -pro

Run:
  python mcp_stock_min.py

"""

from __future__ import annotations
import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from ddgs import DDGS  # your file used this name


# --- Price data ---
import numpy as np
import pandas as pd
import yfinance as yf

# --- Gemini sentiment ---
import google.generativeai as genai


mcp = FastMCP("StockMini")

# -----------------------------
# Helpers
# -----------------------------

def _price_snapshot(ticker: str, lb_days: int = 30) -> Dict[str, Any]:
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lb_days + 10)
    df = yf.download(ticker, start=start.date().isoformat(), end=end.date().isoformat(), progress=False)
    close = df["Close"].dropna()
    ret_15 = float(close.iloc[-1] / close.iloc[-15] - 1.0) if len(close) >= 15 else None
    return {"ticker": ticker.upper(), "last_close": float(close.iloc[-1]), "ret_15": ret_15}


def _ddg_news(query: str, days: int = 7, max_results: int = 8) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    tl = f"d{max(1, min(30, days))}"
    with DDGS() as dd:
        for r in dd.news(query, max_results=max_results, timelimit=tl):
            results.append({
                "title": r.get("title") or "",
                "url": r.get("url") or r.get("href") or "",
                "source": r.get("source") or "",
                "published": r.get("date") or r.get("published") or "",
                "snippet": r.get("body") or r.get("excerpt") or "",
            })
    return results


def _company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def _combine(overall: float, ret_15: Optional[float], risk: str = "medium") -> Dict[str, str]:
    risk_mult = {"low": 0.5, "medium": 1.0, "high": 1.5}.get(risk)
    m = float(ret_15) if ret_15 is not None else 0.0
    z = overall + 0.8 * np.tanh(5 * m)
    thr_buy, thr_sell = 0.6 / risk_mult, -0.6 / risk_mult
    if z >= thr_buy:
        act = "BUY"
        why = f"Sentiment {overall:+.2f} plus 15d momentum {m:+.2%} gives composite {z:+.2f}."
    elif z <= thr_sell:
        act = "SELL/AVOID"
        why = f"Negative: sentiment {overall:+.2f}, 15d momentum {m:+.2%}, composite {z:+.2f}."
    else:
        act = "HOLD/WAIT"
        why = f"Mixed: sentiment {overall:+.2f}, 15d momentum {m:+.2%}, composite {z:+.2f}."
    return {"action": act, "rationale": why}


def _gemini_sentiment(items: List[Dict[str, str]]) -> Dict[str, Any]:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))

    blob = [{k: it.get(k, "") for k in ("title","url","source","published","snippet")} for it in items]
    prompt = (
        "Score each headline in [-1,1] (neg..pos), give label (positive/neutral/negative), and a 1‑sentence rationale.\n"
        "Also return overall_score and a 1‑sentence summary. Respond as strict JSON with keys: per_item, overall_score, summary.\n"
    )  #use transduction here
    resp = model.generate_content([
        {"role":"user", "parts":[prompt, json.dumps(blob)[:120000]]}
    ], generation_config={"response_mime_type": "application/json"})
    text = getattr(resp, "text", None) or (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else None)
    data = json.loads(text)
    return data


# -----------------------------
# Minimal one‑shot tool
# -----------------------------
@mcp.tool()
def stock_recommend(ticker: str, days: int = 7, max_results: int = 8, risk: str = "medium") -> Dict[str, Any]:
    """Return a minimal stock analysis + demo recommendation.
    Args:
      ticker: e.g., AAPL
      days: news window (1–30)
      max_results: number of headlines
      risk: low|medium|high to adjust thresholds
    """
    ticker = ticker.upper().strip()
    name = _company_name(ticker)
    snap = _price_snapshot(ticker, 30)

    query = f"{name} ({ticker}) stock news"
    items = _ddg_news(query, days=days, max_results=max_results)

    senti = _gemini_sentiment(items)
    overall = float(senti.get("overall_score", 0.0))
    rec = _combine(overall, snap.get("ret_15") if isinstance(snap, dict) else None, risk=risk)

    return {
        "ticker": ticker,
        "company": name,
        "price_snapshot": snap,
        "sentiment_overall": overall,
        "recommendation": rec["action"],
        "rationale": rec["rationale"],
        "citations": [{"title": x.get("title"), "url": x.get("url")} for x in senti.get("per_item", [])[:5]],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
