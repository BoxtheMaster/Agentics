#!/usr/bin/env python3
"""
Lag-1 association with discrete daily sentiment (Agentics) and HAC standard errors.

Usage examples:
  python overall_sentiment_discrete_lag1.py --csv Combined_News_DJIA.csv --days 200
  python overall_sentiment_discrete_lag1.py --csv /path/to/Combined_News_DJIA.csv --hac-lags 7
  # Drop intercept if you want (not recommended for inference):
  python overall_sentiment_discrete_lag1.py --csv Combined_News_DJIA.csv --no-const
"""

import os, re, json, math, argparse
from pathlib import Path
from typing import List, Any, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from agentics import Agentics as AG

try:
    from agentics.core.llm_connections import gemini_llm  # your local config
except Exception:
    gemini_llm = None

class SentimentDoc(BaseModel): #Make this more specific
    label: str = Field(default="Neutral", description="Positive, Negative, or Neutral")
    sentiment_score: float =Field(default=0, description="A number between -1 and 1 that represents the level of setiment, where -1 is very negative, 1 is very positive and 0 is neutral")
    rationale: str = Field(default="(no rationale returned)")

    @field_validator("label")
    @classmethod
    def _norm_label(cls, v: str) -> str:
        if not isinstance(v, str): return "Neutral"
        v = v.strip().capitalize()
        return v if v in {"Positive","Negative","Neutral"} else "Neutral"

SYSTEM_PROMPT = (
    "You are a seasoned financial news sentiment analyst. "
    "Given a day's top headlines, return ONE overall sentiment strictly for STOCK MARKET impact.\n"
    "Return ONLY a single JSON object with EXACT keys: "
    '"label" (Positive|Negative|Neutral), '
    '"rationale" (short, grounded in the headlines).\n'
    "Do NOT include extra text before or after the JSON.\n"
    'Example: {"label":"Positive","rationale":"Earnings beats and optimistic guidance outweigh regulatory noise."}'
)

def _clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

def _coerce_to_sentiment(obj: Any) -> SentimentDoc:
    if isinstance(obj, SentimentDoc):
        return obj
    if isinstance(obj, dict):
        safe = {"label": "Neutral", "rationale": "(missing)"}
        safe.update({k: obj[k] for k in ("label","rationale") if k in obj})
        try: return SentimentDoc.model_validate(safe)
        except ValidationError: return SentimentDoc()
    if isinstance(obj, str):
        m = re.search(r"\{.*\}", obj, flags=re.S)
        if m:
            try: return SentimentDoc.model_validate(json.loads(m.group(0)))
            except Exception: return SentimentDoc()
        return SentimentDoc()
    return SentimentDoc()

def build_daily_inputs(df: pd.DataFrame, top_cols: List[str]) -> List[str]:
    inputs = []
    for _, row in df.iterrows():
        heads = [_clean_text(str(row[c])) for c in top_cols]
        heads = [h for h in heads if h and h.lower() != "nan"]
        if not heads:
            inputs.append("No usable headlines today. Return Neutral sentiment with rationale.")
        else:
            bullets = "\n".join(f"- {h}" for h in heads)
            inputs.append(
                "DAILY HEADLINES:\n"
                f"{bullets}\n\n"
                "Return ONLY JSON with keys label, rationale."
            )
    return inputs

_POS = {'surge','beat','beats','record','profit','growth','optimistic','upgrade','rally','rise','soar','gain','expand','strong','outperform'}
_NEG = {'fall','falls','plunge','loss','lawsuit','fraud','downgrade','cut','drop','decline','miss','slump','bankrupt','probe','weak','recall','strike'}

def heuristic_sentiment_from_prompt(prompt: str) -> SentimentDoc:
    bullets = [b.strip("- ").lower() for b in prompt.splitlines() if b.strip().startswith("- ")]
    sp = sum(any(w in b for w in _POS) for b in bullets)
    sn = sum(any(w in b for w in _NEG) for b in bullets)
    if sp > sn: return SentimentDoc(label="Positive", rationale=f"Heuristic: {sp} pos vs {sn} neg cues")
    if sn > sp: return SentimentDoc(label="Negative", rationale=f"Heuristic: {sp} pos vs {sn} neg cues")
    return SentimentDoc(label="Neutral", rationale=f"Heuristic: {sp} pos vs {sn} neg cues")

def get_overall_sentiments(daily_inputs: List[str], debug: bool = True) -> List[SentimentDoc]:
    """
    Robust Agentics call:
      1) sanity check gemini_llm
      2) batch call
      3) per-item retry if missing/empty
      4) heuristic fallback with explicit rationale
    """
    # --- 0) LLM sanity probe ---
    if "gemini_llm" not in globals() or gemini_llm is None:
        print("[ERROR] gemini_llm is not configured. Using heuristic for all days.")
        return [heuristic_sentiment_from_prompt(p) for p in daily_inputs]
    try:
        # Very quick probe to ensure the LLM is callable
        _probe = str(gemini_llm)
    except Exception as e:
        print(f"[ERROR] gemini_llm object not usable: {e}. Using heuristic for all days.")
        return [heuristic_sentiment_from_prompt(p) for p in daily_inputs]

    agent = AG(atype=SentimentDoc, llm=gemini_llm, instruction=SYSTEM_PROMPT)

    import asyncio
    results: List[SentimentDoc] = [None] * len(daily_inputs)

    raw = []
    try:
        out = asyncio.run(agent << daily_inputs)
        out.to_csv("/tmp/agenticoutput.csv")
        raw = list(getattr(out, "states", []))
        if debug:
            print(f"[debug] batch states received: {len(raw)} / {len(daily_inputs)}")
    except Exception as e:
        print(f"[warn] Batch Agentics call failed: {e}")

    for i in range(min(len(raw), len(daily_inputs))):
        s = _coerce_to_sentiment(raw[i])
        if not s or not s.rationale or s.rationale.strip().lower().startswith("(no rationale"):
            results[i] = None  # force retry
        else:
            results[i] = s

    missing = [i for i, v in enumerate(results) if v is None]
    if missing:
        if debug:
            print(f"[debug] retrying {len(missing)} items individually...")
        try:
            import asyncio as aio
            loop = aio.new_event_loop()
            aio.set_event_loop(loop)
            for i in missing:
                try:
                    out_i = loop.run_until_complete(agent << [daily_inputs[i]])
                    states_i = getattr(out_i, "states", [])
                    if debug and (not states_i or not states_i[0]):
                        print(f"[debug] empty/none state for idx {i}; prompt preview:\n{daily_inputs[i][:200]}...")
                    s = _coerce_to_sentiment(states_i[0] if states_i else {})
                    # if still empty rationale, we’ll fall back to heuristic below
                    results[i] = s if s.rationale and not s.rationale.lower().startswith("(no rationale") else None
                except Exception as e:
                    print(f"[warn] per-item Agentics failed at {i}: {e}")
                    results[i] = None
            loop.close()
        except Exception as e:
            print(f"[warn] per-item retry loop error: {e}")

    # --- 4) Heuristic fallback (never empty, has rationale) ---
    for i, v in enumerate(results):
        if v is None:
            if debug:
                print(f"[debug] using heuristic fallback for idx {i}")
            results[i] = heuristic_sentiment_from_prompt(daily_inputs[i])

    return results
def hac_cov_with_pinv(X: np.ndarray, resid: np.ndarray, maxlags: int = 5) -> np.ndarray:
    XtX = X.T @ X
    XtX_pinv = np.linalg.pinv(XtX)
    v = X * resid[:, None]
    S = v.T @ v
    n = len(resid)
    L = min(maxlags, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        Gamma_k = v[k:].T @ v[:-k]
        S += w * (Gamma_k + Gamma_k.T)
    return XtX_pinv @ S @ XtX_pinv

def fit_ols_hac(X: np.ndarray, y: np.ndarray, hac_lags: int = 5):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = hac_cov_with_pinv(X, resid, maxlags=hac_lags)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = [math.erfc(abs(zi) / math.sqrt(2.0)) for zi in z]
    return beta, se, z, pvals

def drop_near_constant_columns(X: np.ndarray, names: List[str], tol: float = 1e-12) -> Tuple[np.ndarray, List[str], List[int]]:
    keep_idx = [j for j in range(X.shape[1]) if X[:, j].std() > tol]
    return X[:, keep_idx], [names[i] for i in keep_idx], keep_idx

def run_lag1_association(daily_df: pd.DataFrame, hac_lags: int = 5, include_const: bool = True):
    """
    Label_t ~ [const +] overall_score_{t-1} + Label_{t-1}
    """
    df = daily_df.sort_values("Date").reset_index(drop=True).copy()
    df["Label"] = df["Label"].astype(int)
    df["overall_score_lag1"] = df["overall_score"].shift(1)
    df["Label_lag1"] = df["Label"].shift(1)

    X_cols = ["overall_score_lag1"]
    df = df.dropna(subset=X_cols + ["Label"]).copy()
    if len(df) < 10:
        print("Warning: very few rows after lagging; results may be unstable.")

    y = df["Label"].to_numpy(dtype=float)
    X0 = df[X_cols].to_numpy(dtype=float)

    if include_const:
        X = np.column_stack([np.ones(len(X0)), X0])
        names = ["const"] + X_cols
    else:
        X = X0
        names = X_cols

    X, names, _ = drop_near_constant_columns(X, names, tol=1e-12)

    beta, se, z, pvals = fit_ols_hac(X, y, hac_lags=hac_lags)
    out = pd.DataFrame({"coef": beta, "std_err(HAC)": se, "z": z, "p_value": pvals}, index=names)

    note = "Lag-1 association: Label_t ~ overall_score_{t-1} + Label_{t-1}"
    print(f"\n=== OLS (Linear Probability) — {note} ({'with' if include_const else 'NO'} intercept) ===")
    print(f"(Newey–West/HAC, L={hac_lags})\n")
    print(out.to_string(float_format=lambda x: f"{x: .4f}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="Combined_News_DJIA.csv", help="Path to CSV")
    ap.add_argument("--days", type=int, default=0, help="Use first N days (0 = all)")
    ap.add_argument("--out", type=str, default="daily_overall_sentiment.csv", help="Output CSV path")
    ap.add_argument("--hac-lags", type=int, default=5, help="HAC lag length")
    ap.add_argument("--no-const", action="store_true", help="Drop intercept from regression")
    args = ap.parse_args()

    csv_path = "/Users/boxuanli/Documents/GitHub/Agentics/docs/data/Combined_News_DJIA.csv"
  

    # 1) Load data
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    df=df.tail(180)
    if args.days and args.days > 0:
        df = df.head(args.days)

    top_cols = [c for c in df.columns if c.startswith("Top")]
    if not top_cols:
        raise ValueError("Top1..Top25 columns not found in the CSV.")

    daily_inputs = build_daily_inputs(df, top_cols)
    sentiments = get_overall_sentiments(daily_inputs)

    def label_to_score(lbl: str) -> int:
        return 1 if lbl == "Positive" else (-1 if lbl == "Negative" else 0)

    out_df = df[["Date","Label"]].copy()
    out_df["overall_label"] = [s.label for s in sentiments]
    out_df["overall_score"] = [label_to_score(s.label) for s in sentiments]

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"[saved] {out_path.resolve()} ({len(out_df)} rows)")
    print(out_df.head(12).to_string(index=False))

    run_lag1_association(out_df, hac_lags=args.hac_lags, include_const=not args.no_const)

if __name__ == "__main__":
    main()