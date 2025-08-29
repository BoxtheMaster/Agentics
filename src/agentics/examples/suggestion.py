#!/usr/bin/env python3
"""
Daily Buy/Sell suggestion from headlines using Agentics (with your gemini_llm)

- Reads Combined_News_DJIA.csv (columns: Date, Label, Top1..Top25)
- Aggregates a day's headlines and asks LLM for: {action, confidence, reason}
- Appends suggestion to a daily table and saves to CSV

Install:
  pip install pandas pydantic "git+https://github.com/IBM/Agentics.git"

LLM:
  You provide `gemini_llm`. Example (uncomment and adjust as needed):
    from agentics.providers import Gemini
    gemini_llm = Gemini(model="gemini-1.5-pro", temperature=0.2)

Usage:
  python suggest_trades.py --csv Combined_News_DJIA.csv --out daily_with_trade_suggestion.csv
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import Counter
from agentics.core.llm_connections import gemini_llm

import pandas as pd
from pydantic import BaseModel, Field, field_validator, ValidationError

# --- Agentics ---
from agentics import Agentics as AG

# Example LLM wiring (uncomment if you want to construct here):
# from agentics.providers import Gemini
# gemini_llm = Gemini(model=os.environ.get("GEMINI_MODEL_ID", "gemini-1.5-pro"), temperature=0.2)

# -----------------------------
# Typed output for suggestions
# -----------------------------
class SuggestionDoc(BaseModel):
    action: str = Field(default="Hold", description="Buy, Sell, or Hold")
    confidence: float = Field(default=0.5, description="Confidence in [0,1]")
    reason: str = Field(default="(no reasoning returned)", description="Short rationale grounded in the headlines")

    @field_validator("action")
    @classmethod
    def _norm_action(cls, v: str) -> str:
        if not isinstance(v, str):
            return "Hold"
        v = v.strip().capitalize()
        return v if v in {"Buy", "Sell", "Hold"} else "Hold"

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, x: float) -> float:
        try:
            x = float(x)
        except Exception:
            return 0.5
        return min(1.0, max(0.0, x))

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip())

def _coerce_suggestion(obj) -> SuggestionDoc:
    """Turn whatever came back from the LLM into a SuggestionDoc safely."""
    if isinstance(obj, SuggestionDoc):
        return obj
    if isinstance(obj, dict):
        safe = {"action": "Hold", "confidence": 0.5, "reason": "(missing)"}
        safe.update({k: obj[k] for k in ("action","confidence","reason") if k in obj})
        try:
            return SuggestionDoc.model_validate(safe)
        except ValidationError:
            return SuggestionDoc()
    if isinstance(obj, str):
        # try to find JSON in text
        m = re.search(r"\{.*\}", obj, flags=re.S)
        if m:
            try:
                return SuggestionDoc.model_validate(json.loads(m.group(0)))
            except Exception:
                return SuggestionDoc()
        return SuggestionDoc()
    return SuggestionDoc()

# -----------------------------
# LLM prompt & Agentics call
# -----------------------------
TRADE_SYSTEM = (
    "You are a market analyst. Based ONLY on the given day's top 25 news headlines, "
    "recommend a trading stance for the U.S. stock market (DJIA proxy). "
    "Choose one of: Buy, Sell, Hold. "
    "Return a JSON object with EXACT keys: "
    '"action" ("Buy"|"Sell"|"Hold"), '
    '"confidence" (float in [0,1]), '
    '"reason" (short, grounded in the headlines).'
)

def make_daily_inputs(df: pd.DataFrame, top_cols: List[str]) -> List[str]:
    """Combine each day's headlines into a single bullet list string."""
    inputs = []
    for _, row in df.iterrows():
        heads = [_clean_text(str(row[c])) for c in top_cols]
        heads = [h for h in heads if h and h.lower() != "nan"]
        if not heads:
            inputs.append("No usable headlines today.")
            continue
        bullets = "\n".join(f"- {h}" for h in heads)
        text = f"DAILY HEADLINES:\n{bullets}\n\nYour task: produce Buy/Sell/Hold, confidence, and a short reason."
        inputs.append(text)
    return inputs

def get_trade_suggestions(daily_inputs: List[str]) -> List[SuggestionDoc]:
    """Run Agentics self-transduction over the list of daily inputs."""
    if "gemini_llm" not in globals():
        raise RuntimeError(
            "Please define/import `gemini_llm` before running.\n"
            "Example:\n  from agentics.providers import Gemini\n"
            "  gemini_llm = Gemini(model='gemini-1.5-pro', temperature=0.2)\n"
        )

    agent = AG(atype=SuggestionDoc, llm=gemini_llm, system=TRADE_SYSTEM)

    import asyncio
    out = asyncio.run(agent << daily_inputs)            # self-transduction pattern (<<)
    raw = getattr(out, "states", [])

    # Pad if fewer outputs than inputs
    if len(raw) < len(daily_inputs):
        raw = list(raw) + [{}] * (len(daily_inputs) - len(raw))

    return [_coerce_suggestion(x) for x in raw]

# -----------------------------
# Main routine
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="Combined_News_DJIA.csv", help="Path to Combined_News_DJIA.csv")
    ap.add_argument("--out", type=str, default="daily_with_trade_suggestion.csv",
                    help="Output CSV with LLM trade suggestion appended")
    ap.add_argument("--limit-days", type=int, default=0, help="Use first N days (0 = all)")
    args = ap.parse_args()

    csv_path = "/Users/boxuanli/Documents/GitHub/Agentics/docs/data/Combined_News_DJIA.csv"

    df = pd.read_csv(csv_path)
    df=df.tail(30)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    if args.limit_days and args.limit_days > 0:
        df = df.head(args.limit_days)

    top_cols = [c for c in df.columns if c.startswith("Top")]
    if not top_cols:
        raise ValueError("No Top1..Top25 columns found in the CSV.")

    daily_inputs = make_daily_inputs(df, top_cols)
    suggestions = get_trade_suggestions(daily_inputs)   # list[SuggestionDoc], len == len(df)

    # Append to table
    df_out = df[["Date", "Label"]].copy()
    df_out["llm_trade_action"] = [s.action for s in suggestions]
    df_out["llm_trade_confidence"] = [float(s.confidence) for s in suggestions]
    df_out["llm_trade_reason"] = [s.reason for s in suggestions]

    # Save
    out_path = Path(args.out)
    df_out.to_csv(out_path, index=False)
    print(f"[saved] {out_path.resolve()}  ({len(df_out)} rows)")

    # (Optional) quick summary
    print(df_out.head().to_string(index=False))

if __name__ == "__main__":
    main()