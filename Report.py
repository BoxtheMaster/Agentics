# models.py

from datetime import date
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from agentics import AG


from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List

class MarketRow(BaseModel):
    """Daily financial market features and aggregated news headlines."""

    Date: Optional[date] = Field(
        None, description="Trading day (calendar date)."
    )
    spx: Optional[float] = Field(
        None, description="S&P 500 Index (^GSPC) daily close price."
    )
    sx5e: Optional[float] = Field(
        None, description="Euro Stoxx 50 Index (^STOXX50E) daily close price."
    )
    btcusd: Optional[float] = Field(
        None, description="Bitcoin (BTC-USD) daily closing price in U.S. dollars."
    )
    vix: Optional[float] = Field(
        None, description="CBOE Volatility Index (^VIX), implied equity market volatility."
    )
    gsg: Optional[float] = Field(
        None, description="S&P GSCI Commodity Index ETF (GSG), proxy for global commodity prices."
    )
    dgs2: Optional[float] = Field(
        None, description="U.S. Treasury 2-Year yield (percent)."
    )
    dgs10: Optional[float] = Field(
        None, description="U.S. Treasury 10-Year yield (percent)."
    )
    usd_index: Optional[float] = Field(
        None, description="Broad U.S. Dollar Index (DTWEXBGS), trade-weighted value of USD."
    )
    brent: Optional[float] = Field(
        None, description="Brent crude oil spot price (USD per barrel)."
    )
    gld: Optional[float] = Field(
        None, description="SPDR Gold Shares ETF (GLD) close price, proxy for gold."
    )
    us10y2y: Optional[float] = Field(
        None, description="U.S. yield curve slope, 10-year minus 2-year Treasury yield."
    )
    headlines: Optional[List[str]] = Field(
        None, description="List of aggregated Reddit news headlines for the day."
    )

  
class MacroRow(BaseModel):
    """Daily macroeconomic and fundamental indicators (forward-filled)."""

    date: Optional[date] = Field(
        None, description="Reference date for macroeconomic indicators."
    )
    fedfunds: Optional[float] = Field(
        None, description="Federal Funds Effective Rate (short-term U.S. policy rate)."
    )
    tb3ms: Optional[float] = Field(
        None, description="3-Month Treasury Bill yield (percent)."
    )
    t10y3m: Optional[float] = Field(
        None, description="Spread between 10-year and 3-month Treasury yields."
    )
    cpiaucsl: Optional[float] = Field(
        None, description="Consumer Price Index (CPI), all items (headline inflation)."
    )
    cpilfesl: Optional[float] = Field(
        None, description="Consumer Price Index, core (excluding food and energy)."
    )
    pcepi: Optional[float] = Field(
        None, description="Personal Consumption Expenditures Price Index (headline PCE)."
    )
    pcepilfe: Optional[float] = Field(
        None, description="Core PCE Price Index (excluding food and energy)."
    )
    unrate: Optional[float] = Field(
        None, description="U.S. unemployment rate (percent of labor force)."
    )
    payems: Optional[float] = Field(
        None, description="Nonfarm payroll employment, total number of jobs (thousands)."
    )
    indpro: Optional[float] = Field(
        None, description="Industrial Production Index, measures manufacturing output."
    )
    rsafs: Optional[float] = Field(
        None, description="Retail Sales Index, measure of consumer spending activity."
    )

   
def load_market_csv(path: str) -> list[MarketRow]:
    df = pd.read_csv(path)
    return [MarketRow(**row.to_dict()) for _, row in df.iterrows()]

def load_macro_csv(path: str) -> list[MacroRow]:
    df = pd.read_csv(path)
    return [MacroRow(**row.to_dict()) for _, row in df.iterrows()]


market_csv: str = "data/market_with_news_agg.csv",
macro_csv: str = "data/macro_factors.csv",

market_rows = load_market_csv(market_csv)
macro_rows  = load_macro_csv(macro_csv)
market_ag = AG(atype=MarketRow, states=market_rows)
macro_ag  = AG(atype=MacroRow,  states=macro_rows)
