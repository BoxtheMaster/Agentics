#!/usr/bin/env python3
"""
DJIA-5 Agentics pipeline:
1) LLM-only 'headlines/themes' per ticker (no browsing; synthesis + confidence)
2) 90-day pairwise correlations for 5 tickers (10 pairs total)
3) Correlation-aware Buy/Sell/Hold per ticker with sentiment, confidence, reasoning, risks
4) Save two CSVs

Run:
  python djia5_llm_corr_recos.py --corr-window 90 \
    --out-headlines djia5_llm_headlines.csv --out-recos djia5_llm_recos.csv

Prereqs:
  python -m pip install pandas numpy pydantic yfinance "git+https://github.com/IBM/Agentics.git"

LLM:
  You must define/import `gemini_llm` before running Agentics.
  Example (uncomment to define here):
    from agentics.providers import Gemini
    import os
    gemini_llm = Gemini(model=os.environ.get("GEMINI_MODEL_ID","gemini-1.5-pro"), temperature=0.2)
"""

import os, re, json, argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, ValidationError, field_validator
from agentics import Agentics as AG

from agentics.core.llm_connections import gemini_llm
# -------------------------------
# Choose 5 DJIA names (distinct sectors)
# -------------------------------
DJIA_5 = [
    ("AAPL", "Information Technology"),
    ("UNH",  "Health Care"),
    ("JPM",  "Financials"),
    ("CAT",  "Industrials"),
    ("PG",   "Consumer Staples"),
]

# # If you want to define the LLM here, uncomment:
# from agentics.providers import Gemini
# gemini_llm = Gemini(model=os.environ.get("GEMINI_MODEL_ID","gemini-1.5-pro"), temperature=0.2)

# -------------------------------
# Pydantic Schemas
# -------------------------------
class LLMHeadlines(BaseModel):
    headlines: List[str] = Field(default_factory=list, description="3–6 concise bullets on current themes/news")
    confidence: float = Field(default=0.5, description="Self-rated confidence in [0,1]")

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, x: float) -> float:
        try: x = float(x)
        except Exception: return 0.5
        return min(1.0, max(0.0, x))

class StockReco(BaseModel):
    action: str = Field(default="Hold", description="Buy, Sell, or Hold")
    confidence: float = Field(default=0.5, description="Confidence [0,1]")
    sentiment_label: str = Field(default="Neutral", description="Positive/Negative/Neutral")
    sentiment_score: float = Field(default=0.0, description="Polarity [-1,1]")
    reasoning: str = Field(default="(no reasoning returned)", description="Headline- & correlation-grounded reasoning")
    risks: str = Field(default="(no risks returned)", description="Key risks/caveats incl. correlation tail-risks")

    @field_validator("action")
    @classmethod
    def _val_action(cls, v: str) -> str:
        if not isinstance(v, str): return "Hold"
        v = v.strip().capitalize()
        return v if v in {"Buy","Sell","Hold"} else "Hold"

    @field_validator("confidence")
    @classmethod
    def _val_conf(cls, x: float) -> float:
        try: x = float(x)
        except Exception: return 0.5
        return min(1.0, max(0.0, x))

    @field_validator("sentiment_label")
    @classmethod
    def _val_label(cls, v: str) -> str:
        if not isinstance(v, str): return "Neutral"
        v = v.strip().capitalize()
        return v if v in {"Positive","Negative","Neutral"} else "Neutral"

    @field_validator("sentiment_score")
    @classmethod
    def _val_score(cls, x: float) -> float:
        try: x = float(x)
        except Exception: return 0.0
        return min(1.0, max(-1.0, x))

# -------------------------------
# Helpers
# -------------------------------
def _clean(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

def coerce(obj: Any, model):
    """
    Generic coercer. For StockReco, also map 'rationale' -> 'reasoning' if needed.
    """
    def _map_keys(d: Dict[str, Any]) -> Dict[str, Any]:
        if model is StockReco and "reasoning" not in d and "rationale" in d:
            d = {**d, "reasoning": d["rationale"]}
        return d

    if isinstance(obj, model): return obj
    if isinstance(obj, dict):
        try: return model.model_validate(_map_keys(obj))
        except ValidationError: return model()
    if isinstance(obj, str):
        m = re.search(r"\{.*\}", obj, flags=re.S)
        if m:
            try: return model.model_validate(_map_keys(json.loads(m.group(0))))
            except Exception: return model()
        return model()
    return model()

# -------------------------------
# LLM prompts
# -------------------------------
SYSTEM_HEADLINES = (
    "You are a financial news summarizer. Without browsing, and relying on your internal knowledge, "
    "produce a compact list of CURRENT THEMES/HEADLINES for the given US stock. If unsure, prefer stable themes. "
    "Include 3–6 bullets max and self-rate your confidence in [0,1]. "
    'Return ONLY JSON: {"headlines":[...], "confidence":<float>}.'
)
def make_headline_prompt(ticker: str, sector: str) -> str:
    return (
        f"TICKER: {ticker}\nSECTOR: {sector}\n"
        "TASK: Provide 3–6 bullet 'recent themes/headlines' and your confidence [0,1].\n"
        'Return ONLY JSON with keys "headlines" and "confidence".'
    )

SYSTEM_RECO = (
    "You are a professional equity analyst. Based ONLY on the provided LLM themes/headlines summary "
    "and the correlation context for this stock within a 5-name Dow slice, produce a SAME-DAY view. "
    "Explicitly consider correlation implications (e.g., cluster risk, hedges, spillover). "
    "Return ONE JSON object with EXACT keys:\n"
    '"action" ("Buy"|"Sell"|"Hold"), '
    '"confidence" (float in [0,1]), '
    '"sentiment_label" ("Positive"|"Negative"|"Neutral"), '
    '"sentiment_score" (float in [-1,1]), '
    '"reasoning" (short, grounded in bullets & correlation notes), '
    '"risks" (short, include correlation-related tail risks).'
)
def make_reco_prompt(ticker: str, sector: str, headlines_json: List[str], conf: float, corr_summary: str) -> str:
    bullets = "\n".join(f"- {h}" for h in headlines_json[:6]) if headlines_json else "- (no headlines)"
    return (
        f"TICKER: {ticker}\nSECTOR: {sector}\n"
        f"LLM THEMES/HEADLINES (self-rated confidence {conf:.2f}):\n{bullets}\n\n"
        f"CORRELATION CONTEXT (90d): {corr_summary}\n\n"
        "TASK: Output one JSON object with the specified keys."
    )

# -------------------------------
# Correlation computation (5 tickers => 10 pairs)
# -------------------------------
def corr_matrix(tickers: List[str], window: int = 90) -> pd.DataFrame:
    data = yf.download(tickers, period=f"{window+5}d", interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    px = data["Close"] if isinstance(data, pd.DataFrame) and "Close" in data.columns else data
    px = px.dropna(how="all").tail(window)
    rets = px.pct_change().dropna(how="all")
    rets = rets[[t for t in tickers if t in rets.columns]]
    return rets.corr()

def corr_summary_for(ticker: str, C: pd.DataFrame, top_k: int = 2) -> str:
    if C.empty or ticker not in C.columns:
        return "Correlation data unavailable."
    s = C[ticker].drop(labels=[ticker], errors="ignore").dropna()
    if s.empty: return "Correlation data insufficient."
    avg_abs = float(s.abs().mean())
    topn = s.abs().sort_values(ascending=False).head(top_k)
    pairs = [f"{k}:{C.loc[k, ticker]:+.2f}" for k in topn.index]
    return f"Avg |corr|: {avg_abs:.2f}; Top {len(pairs)}: " + ", ".join(pairs)

# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr-window", type=int, default=90, help="Correlation window (trading days)")
    ap.add_argument("--out-headlines", type=str, default="djia5_llm_headlines.csv",
                    help="CSV for LLM-only 'pulled' headline/themes per ticker")
    ap.add_argument("--out-recos", type=str, default="djia5_llm_recos.csv",
                    help="CSV for correlation-aware recommendations per ticker")
    args = ap.parse_args()

    if "gemini_llm" not in globals():
        raise RuntimeError(
            "Please define/import `gemini_llm` before running.\n"
            "Example:\n"
            "  from agentics.providers import Gemini\n"
            "  gemini_llm = Gemini(model=os.environ.get('GEMINI_MODEL_ID','gemini-1.5-pro'), temperature=0.2)\n"
        )

    tickers = [t for (t, _) in DJIA_5]
    sectors = {t: s for (t, s) in DJIA_5}

    # 1) LLM-only “headlines/themes”
    hl_prompts = [make_headline_prompt(t, sectors[t]) for t in tickers]
    hl_agent = AG(atype=LLMHeadlines, llm=gemini_llm, system=SYSTEM_HEADLINES)

    import asyncio
    print("[info] LLM synthesizing headlines/themes for 5 tickers...")
    hl_out = asyncio.run(hl_agent << hl_prompts)
    raw_h = getattr(hl_out, "states", [])
    if len(raw_h) < len(hl_prompts):
        raw_h = list(raw_h) + [{}] * (len(hl_prompts) - len(raw_h))
    llm_heads = [coerce(x, LLMHeadlines) for x in raw_h]

    # Save headlines CSV
    head_rows = []
    for (t, sec), hl in zip(DJIA_5, llm_heads):
        head_rows.append({
            "ticker": t,
            "sector": sec,
            "headlines_confidence": float(hl.confidence),
            "llm_headlines_json": json.dumps(hl.headlines, ensure_ascii=False)
        })
    df_heads = pd.DataFrame(head_rows)
    Path(args.out_headlines).write_text(df_heads.to_csv(index=False))
    print(f"[saved] {Path(args.out_headlines).resolve()}")

    # 2) Correlations (5 tickers -> 10 pairs)
    print("[info] Computing 90-day correlations for 5 tickers (10 pairs)...")
    C = corr_matrix(tickers, window=args.corr_window)

    # 3) Correlation-aware recommendations
    reco_prompts = []
    for (t, sec), hl in zip(DJIA_5, llm_heads):
        reco_prompts.append(
            make_reco_prompt(t, sec, hl.headlines, hl.confidence, corr_summary_for(t, C))
        )
    reco_agent = AG(atype=StockReco, llm=gemini_llm, system=SYSTEM_RECO)
    print("[info] LLM generating correlation-aware recommendations...")
    r_out = asyncio.run(reco_agent << reco_prompts)
    raw_r = getattr(r_out, "states", [])
    if len(raw_r) < len(reco_prompts):
        raw_r = list(raw_r) + [{}] * (len(reco_prompts) - len(raw_r))
    recos = [coerce(x, StockReco) for x in raw_r]

    # Save recos CSV
    rows = []
    for (t, sec), reco in zip(DJIA_5, recos):
        rows.append({
            "ticker": t,
            "sector": sec,
            "action": reco.action,
            "confidence": float(reco.confidence),
            "sentiment_label": reco.sentiment_label,
            "sentiment_score": float(reco.sentiment_score),
            "reasoning": reco.reasoning,   # <-- explicit reasoning field
            "risks": reco.risks,
            "corr_summary": corr_summary_for(t, C)
        })
    df_recos = pd.DataFrame(rows)
    Path(args.out_recos).write_text(df_recos.to_csv(index=False))
    print(f"[saved] {Path(args.out_recos).resolve()}")
    print(df_recos[["ticker","action","confidence","sentiment_label","sentiment_score","reasoning","risks","corr_summary"]].to_string(index=False))

if __name__ == "__main__":
    main()