#!/usr/bin/env python3
"""
Overall daily sentiment with Agentics + association regression (HAC, pseudoinverse).

Usage:
  python overall_sentiment_hac_full.py --csv Combined_News_DJIA.csv --days 120
  python overall_sentiment_hac_full.py --csv Combined_News_DJIA.csv --days 120 --lag1

What it does:
1) For each day, combine Top1..Top25 headlines and ask your LLM (gemini_llm) for ONE overall sentiment:
   {"label": Positive|Negative|Neutral, "score": [-1,1], "rationale": "..."}
2) Save daily table with appended opinion (Date, Label, overall_label, overall_score, overall_rationale)
3) Run OLS (linear probability) with Newey–West/HAC SEs using numpy only.
   - Uses Moore–Penrose pseudoinverse to avoid singular matrix failures.
   - Optionally `--lag1` to test Label_t ~ sentiment_{t-1} (+ Label_{t-1} control).

Prereqs:
  python -m pip install pandas numpy pydantic "git+https://github.com/IBM/Agentics.git"

LLM:
  You must define/import `gemini_llm` before calling Agentics.
"""

import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from agentics import Agentics as AG  # IBM Agentics (self-transduction with <<)
from agentics.core.llm_connections import gemini_llm

# If you want to construct the LLM here, uncomment:
# from agentics.providers import Gemini
# gemini_llm = Gemini(model=os.environ.get("GEMINI_MODEL_ID","gemini-1.5-pro"), temperature=0.2)

# -------------------------
# Typed schema + coercion
# -------------------------
class SentimentDoc(BaseModel):
    label: str = Field(default="Neutral", description="One of: Positive, Negative, Neutral")
    score: float = Field(default=0.0, description="Polarity in [-1, +1]")
    rationale: str = Field(default="(no rationale returned)", description="Short, grounded in the headlines")

    @field_validator("label")
    @classmethod
    def _norm_label(cls, v: str) -> str:
        if not isinstance(v, str): return "Neutral"
        v = v.strip().capitalize()
        return v if v in {"Positive", "Negative", "Neutral"} else "Neutral"

    @field_validator("score")
    @classmethod
    def _clamp_score(cls, x: float) -> float:
        try: x = float(x)
        except Exception: return 0.0
        return max(-1.0, min(1.0, x))

SYSTEM_PROMPT = (
    "You are a seasoned financial news sentiment analyst. "
    "Given a day's top headlines, return ONE overall sentiment strictly for STOCK MARKET impact. "
    'Return a JSON object with EXACT keys: "label" (Positive|Negative|Neutral), '
    '"score" (float in [-1,1]), "rationale" (short, grounded in the headlines).'
)

def _clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    return re.sub(r"\s+", " ", s.strip())

def _coerce_to_sentiment(obj: Any) -> SentimentDoc:
    """Coerce Agentics/LLM output into a SentimentDoc (handles dicts/strings/empties)."""
    if isinstance(obj, SentimentDoc):
        return obj
    if isinstance(obj, dict):
        safe = {"label": "Neutral", "score": 0.0, "rationale": "(missing)"}
        safe.update({k: obj[k] for k in ("label","score","rationale") if k in obj})
        try:
            return SentimentDoc.model_validate(safe)
        except ValidationError:
            return SentimentDoc()
    if isinstance(obj, str):
        m = re.search(r"\{.*\}", obj, flags=re.S)
        if m:
            try: return SentimentDoc.model_validate(json.loads(m.group(0)))
            except Exception: return SentimentDoc()
        return SentimentDoc()
    return SentimentDoc()

# -------------------------
# Build inputs & call Agentics
# -------------------------
def build_daily_inputs(df: pd.DataFrame, top_cols: List[str]) -> List[str]:
    """One string per day: bullet-list all headlines."""
    inputs = []
    for _, row in df.iterrows():
        heads = [_clean_text(str(row[c])) for c in top_cols]
        heads = [h for h in heads if h and h.lower() != "nan"]
        if not heads:
            inputs.append("No usable headlines today.")
        else:
            bullets = "\n".join(f"- {h}" for h in heads)
            inputs.append(
                "DAILY HEADLINES:\n"
                f"{bullets}\n\n"
                "Your task: output ONE JSON object with keys exactly: "
                '"label", "score", "rationale".'
            )
    return inputs

def get_overall_sentiments(daily_inputs: List[str]) -> List[SentimentDoc]:
    """Run Agentics self-transduction to get one sentiment per day using your provided gemini_llm."""
    if "gemini_llm" not in globals():
        raise RuntimeError(
            "Please define/import `gemini_llm` before running.\n"
            "Example:\n"
            "  from agentics.providers import Gemini\n"
            "  import os\n"
            "  gemini_llm = Gemini(model=os.environ.get('GEMINI_MODEL_ID','gemini-1.5-pro'), temperature=0.2)\n"
        )
    agent = AG(atype=SentimentDoc, llm=gemini_llm, system=SYSTEM_PROMPT)

    import asyncio
    out = asyncio.run(agent << daily_inputs)  # self-transduction pattern
    raw = getattr(out, "states", [])

    # Pad if fewer outputs than inputs
    if len(raw) < len(daily_inputs):
        raw = list(raw) + [{}] * (len(daily_inputs) - len(raw))

    return [_coerce_to_sentiment(x) for x in raw]

# -------------------------
# HAC (Newey–West) with pseudoinverse
# -------------------------
def hac_cov_with_pinv(X: np.ndarray, resid: np.ndarray, maxlags: int = 5) -> np.ndarray:
    """
    Newey–West / HAC covariance using Moore–Penrose inverse when X'X is singular.
    Var(beta) = (X'X)^+ * S * (X'X)^+  (Bartlett kernel).
    """
    XtX = X.T @ X
    XtX_pinv = np.linalg.pinv(XtX)     # pseudoinverse (robust to singularity)
    v = X * resid[:, None]             # n x p
    S = v.T @ v                        # k=0 term
    n = len(resid)
    L = min(maxlags, n - 1)
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        Gamma_k = v[k:].T @ v[:-k]
        S += w * (Gamma_k + Gamma_k.T)
    return XtX_pinv @ S @ XtX_pinv

def fit_ols_hac(X: np.ndarray, y: np.ndarray, hac_lags: int = 5):
    """
    OLS via least squares (handles singular X'X) + HAC SEs with pseudoinverse.
    Returns (beta, se, z, pvals).
    """
    # Coefficients via least squares
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    # HAC covariance
    cov = hac_cov_with_pinv(X, resid, maxlags=hac_lags)
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    # z-scores & two-sided p-values (normal approx)
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = [math.erfc(abs(zi) / math.sqrt(2.0)) for zi in z]
    return beta, se, z, pvals

def drop_near_constant_columns(X: np.ndarray, names: List[str], tol: float = 1e-12):
    """
    Drop columns (except intercept) with ~zero variance to reduce collinearity.
    Returns (X_reduced, kept_names, kept_idx_map).
    """
    keep_idx = [0]  # always keep intercept
    for j in range(1, X.shape[1]):
        if X[:, j].std() > tol:
            keep_idx.append(j)
    return X[:, keep_idx], [names[i] for i in keep_idx], keep_idx

# -------------------------
# Association regression (same-day or lagged)
# -------------------------
def run_association_hac(daily_df: pd.DataFrame, use_lag: bool, hac_lags: int = 5):
    """
    Regress Label on overall sentiment (association):
      y_t = α + β1*overall_score (+ β2*is_pos + β3*is_neg) [+ controls] + ε_t
    Neutral is baseline for label dummies. Optionally use t-1 sentiment + Label_{t-1}.
    """
    df = daily_df.sort_values("Date").reset_index(drop=True).copy()
    df=df.tail(10)
    df["Label"] = df["Label"].astype(int)
    df["is_pos"] = (df["overall_label"] == "Positive").astype(int)
    df["is_neg"] = (df["overall_label"] == "Negative").astype(int)

    base = ["overall_score", "is_pos", "is_neg"]  # Neutral omitted
    if use_lag:
        for c in base + ["Label"]:
            df[f"{c}_lag1"] = df[c].shift(1)
        X_cols = [f"{c}_lag1" for c in base] + ["Label_lag1"]  # control yesterday's Label
        df = df.dropna(subset=X_cols + ["Label"]).copy()
        note = "LAGGED: Label_t ~ sentiment_{t-1} + Label_{t-1}"
    else:
        X_cols = base
        df = df.dropna(subset=X_cols + ["Label"]).copy()
        note = "SAME-DAY association"

    if len(df) < 10:
        print("Warning: very few rows remain; results may be unstable.")

    y = df["Label"].to_numpy(dtype=float)
    X0 = df[X_cols].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X0)), X0])  # add intercept
    names = ["const"] + X_cols

    # Drop near-constant columns (except intercept) to avoid singularity
    X, names, _ = drop_near_constant_columns(X, names, tol=1e-12)

    beta, se, z, pvals = fit_ols_hac(X, y, hac_lags=hac_lags)
    out = pd.DataFrame({"coef": beta, "std_err(HAC)": se, "z": z, "p_value": pvals}, index=names)

    print(f"\n=== OLS (Linear Probability) — {note} ===")
    print(f"(Newey–West/HAC, L={hac_lags}; Neutral is the omitted category)\n")
    print(out.to_string(float_format=lambda x: f"{x: .4f}"))

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="Combined_News_DJIA.csv", help="Path to Combined_News_DJIA.csv")
    ap.add_argument("--days", type=int, default=120, help="Use first N days to control LLM calls (0 = all)")
    ap.add_argument("--out", type=str, default="daily_overall_sentiment.csv",
                    help="Output CSV for daily table with overall sentiment")
    ap.add_argument("--lag1", action="store_true", help="Use yesterday's sentiment to explain today's up/down")
    ap.add_argument("--hac_lags", type=int, default=5, help="HAC lag length (e.g., 5–7 for daily)")
    args = ap.parse_args()

    csv_path="/Users/boxuanli/Documents/GitHub/Agentics/docs/data/Combined_News_DJIA.csv"
    
    df = pd.read_csv(csv_path)
    df=df.tail(10)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    if args.days and args.days > 0:
        df = df.head(args.days)

    top_cols = [c for c in df.columns if c.startswith("Top")]
    if not top_cols:
        raise ValueError("Top1..Top25 columns not found in the CSV.")

    # Build daily inputs & call Agentics
    daily_inputs = build_daily_inputs(df, top_cols)
    sentiments = get_overall_sentiments(daily_inputs)  # list[SentimentDoc]

    # Build + save daily table
    out_df = df[["Date", "Label"]].copy()
    out_df["overall_label"] = [s.label for s in sentiments]
    out_df["overall_score"] = [float(s.score) for s in sentiments]
    out_df["overall_rationale"] = [s.rationale for s in sentiments]

    out_path = Path(args.out)

    out_df.to_csv(out_path, index=False)
    print(f"[saved] {out_path.resolve()}  ({len(out_df)} rows)")
    print("\nPreview:")
    print(out_df.head(8).to_string(index=False))

    # Association regression with HAC SEs (numpy only)
    run_association_hac(out_df, use_lag=args.lag1, hac_lags=args.hac_lags)

if __name__ == "__main__":
    main()