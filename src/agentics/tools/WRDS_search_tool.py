import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from agentics import Agentics as AG
import wrds

class WRDSQuery(BaseModel):
    sql: str = Field(description="Single SELECT SQL for WRDS (Postgres). Use library-qualified tables, e.g., comp.funda.")
    libraries_used: list[str] = Field(default_factory=list)

SYSTEM = """
You write PostgreSQL for WRDS (read-only). Return ONE complete SELECT statement.
Prefer canonical sources and simple joins:

- CRSP daily: crsp.dsf(date, permno, ret, prc, vol, shrout)
- CRSP names: crsp.stocknames(permno, permco, namedt, nameenddt, cusip, ticker)
- Compustat fundamentals: comp.funda(gvkey, datadate, indfmt='INDL', datafmt='STD', consol='C', at, lt, sale, ...; tic, conm)
- Compustat security master: comp.secm(gvkey, iid, datadate, tic, prccm, cshoq)
- Compustat company static: comp.company(gvkey, conm, cik, sic, naics, loc)

Use BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD' for dates when the user hints periods.
If a table name is ambiguous, pick the most standard one and proceed.
Output JSON ONLY with keys: sql, libraries_used[]. No markdown.
"""

def search(query: str) -> pd.DataFrame:
    # Pass-through SQL (exactly what the user provides)
    if query.strip().lower().startswith("sql:"):
        sql = query.strip()[4:].strip()
        db = wrds.Connection()
        df = db.raw_sql(sql)
        df.attrs["wrds_sql"] = sql
        return df

    agent = AG(atype=WRDSQuery, llm=gemini_llm, system=SYSTEM)

    async def _run():
        out = await (agent << [query])
        return out.states[0]

    state = asyncio.run(_run())
    sql = (state.sql or "").strip()

    db = wrds.Connection()
    
    df = db.raw_sql(sql)

    df.attrs["wrds_sql"] = sql
    df.attrs["wrds_libraries_used"] = state.libraries_used
    return df


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "sql: SELECT gvkey, conm FROM comp.company LIMIT 5;"
    out = search(q)
    print(out.head())
    print("\nSQL used:\n", out.attrs.get("wrds_sql"))