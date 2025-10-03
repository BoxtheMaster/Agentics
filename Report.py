# models.py

from datetime import date
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from agentics import AG


class MarketRow(BaseModel):
    """
    One daily observation of market features and aggregated news.
    """

    Date: Optional[date] = Field(alias="Date")

    # Prices / levels
    spx: Optional[float] = Field(None, alias="SPX")
    sx5e: Optional[float] = Field(None, alias="SX5E")
    btcusd: Optional[float] = Field(None, alias="BTCUSD")
    vix: Optional[float] = Field(None, alias="VIX")            # CBOE VIX
    v2x: Optional[float] = Field(None, alias="V2X")            # EURO STOXX 50 vol (if present)
    gsg: Optional[float] = Field(None, alias="GSG")            # iShares S&P GSCI ETF proxy
    usd_index: Optional[float] = Field(None, alias="DTWEXBGS") # Broad USD index
    brent: Optional[float] = Field(None, alias="DCOILBRENTEU")
    gold_usd: Optional[float] = Field(None, alias="GOLDAMUSD") # FRED fix if present
    gld: Optional[float] = Field(None, alias="GLD")            # ETF proxy fallback

    # Rates & curves 
    fedfunds: Optional[float] = Field(None, alias="FEDFUNDS")
    dgs2: Optional[float] = Field(None, alias="DGS2")
    dgs10: Optional[float] = Field(None, alias="DGS10")
    t10y2y_us: Optional[float] = Field(None, alias="T10Y2Y_US")

    # Real estate derived index
    us_real_house_price_idx: Optional[float] = Field(None, alias="USRealHousePriceIdx")

    # Simple derived returns/logs (if your CSV has them)
    spx_ret: Optional[float] = Field(None, alias="SPXRet")
    sx5e_ret: Optional[float] = Field(None, alias="SX5Ret")
    btc_ret: Optional[float] = Field(None, alias="BTCRet")
    logspx: Optional[float] = Field(None, alias="LOGSPX")
    logsx5: Optional[float] = Field(None, alias="LOGSX5")

    # Aggregated Reddit headlines (single column that stores a list of strings)
    reddit_headlines: Optional[List[str]] = Field(None, alias="RedditHeadlines")

  
    

# ---------- Macroeconomic dataset (macro_factors_no_overlap.csv) ----------
class MacroRow(BaseModel):
    """
    One daily observation on macro factors (no overlaps with MarketRow).
    Keep everything Optional so the model remains tolerant to sparse series.
    """

    Date: Optional[date] = Field(alias="Date")

    # Policy / curve (non-overlapping ones only; FEDFUNDS, DGS2, DGS10 removed by design)
    tbill_3m: Optional[float] = Field(None, alias="TB3MS")
    t10y3m_spread: Optional[float] = Field(None, alias="T10Y3M")

    # Inflation indices & core
    cpi_all: Optional[float] = Field(None, alias="CPIAUCSL")
    cpi_core: Optional[float] = Field(None, alias="CPILFESL")
    pce: Optional[float] = Field(None, alias="PCEPI")
    pce_core: Optional[float] = Field(None, alias="PCEPILFE")

    # Derived inflation changes (if present)
    cpi_yoy: Optional[float] = Field(None, alias="CPIAUCSL_YoY")
    cpi_mom: Optional[float] = Field(None, alias="CPIAUCSL_MoM")
    cpi_core_yoy: Optional[float] = Field(None, alias="CPILFESL_YoY")
    cpi_core_mom: Optional[float] = Field(None, alias="CPILFESL_MoM")
    pce_yoy: Optional[float] = Field(None, alias="PCEPI_YoY")
    pce_mom: Optional[float] = Field(None, alias="PCEPI_MoM")
    pce_core_yoy: Optional[float] = Field(None, alias="PCEPILFE_YoY")
    pce_core_mom: Optional[float] = Field(None, alias="PCEPILFE_MoM")

    # Inflation expectations
    breakeven_10y: Optional[float] = Field(None, alias="T10YIE")
    breakeven_5y: Optional[float] = Field(None, alias="T5YIE")
    fivey_inf_fwd: Optional[float] = Field(None, alias="T5YIFR")

    # Labor / activity
    unemployment_rate: Optional[float] = Field(None, alias="UNRATE")
    payrolls_total: Optional[float] = Field(None, alias="PAYEMS")
    jobless_claims: Optional[float] = Field(None, alias="ICSA")
    industrial_production: Optional[float] = Field(None, alias="INDPRO")
    industrial_prod_yoy: Optional[float] = Field(None, alias="INDPRO_YoY")
    industrial_prod_mom: Optional[float] = Field(None, alias="INDPRO_MoM")
    retail_sales: Optional[float] = Field(None, alias="RSAFS")
    retail_sales_mom: Optional[float] = Field(None, alias="RSAFS_MoM")

    # Surveys (modern PMI/UMich might appear if you swapped series)
    consumer_sentiment: Optional[float] = Field(None, alias="UMCSENT")
    manufacturing_pmi: Optional[float] = Field(None, alias="MAN_PMI")   # if you mapped modern PMI
    services_pmi: Optional[float] = Field(None, alias="SERV_PMI")       # if you mapped modern PMI

    # Housing (price index overlap removed already in your pipeline)
    housing_starts: Optional[float] = Field(None, alias="HOUST")
    building_permits: Optional[float] = Field(None, alias="PERMIT")

    # Credit / financial conditions
    nfc_index: Optional[float] = Field(None, alias="NFCI")
    ig_oas: Optional[float] = Field(None, alias="BAMLC0A0CM")
    hy_oas: Optional[float] = Field(None, alias="BAMLH0A0HYM2")

    # FX / commodities: USD, Brent, Gold were intentionally kept in *market* to avoid overlap
   
# ---------- Helpers to load CSVs into typed lists ----------
def load_market_csv(path: str) -> list[MarketRow]:
    df = pd.read_csv(path)
    # normalize Date column name/case just in case
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    return [MarketRow(**row.to_dict()) for _, row in df.iterrows()]

def load_macro_csv(path: str) -> list[MacroRow]:
    df = pd.read_csv(path)
    if "date" in df.columns and "Date" not in df.columns:
        df = df.rename(columns={"date": "Date"})
    return [MacroRow(**row.to_dict()) for _, row in df.iterrows()]






async def main(
    market_csv: str = "market_with_news_agg.csv",
    macro_csv: str = "macro_factors_no_overlap.csv",
):
    # Load typed rows
    market_rows = load_market_csv(market_csv)
    macro_rows  = load_macro_csv(macro_csv)
    market_ag = AG(atype=MarketRow, states=market_rows)
    macro_ag  = AG(atype=MacroRow,  states=macro_rows)
