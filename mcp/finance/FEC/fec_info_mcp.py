"""
FastMCP server exposing an OpenFEC "latest_filings" tool.

Requirements:
  pip install fastmcp pydantic requests
Env:
  FEC_API_KEY or OPENFEC_API_KEY must be set

Run:
  python fec_info_mcp.py

Clients:
  Any MCP-capable client (e.g., Claude Desktop). This server speaks stdio by default.
"""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl
from mcp.server.fastmcp import FastMCP

from openfec_client import OpenFECClient


# ----------------------------
# Pydantic Schemas
# ----------------------------

class LatestFilingsParams(BaseModel):
    """
    Parameters for the latest_filings tool.

    committee: Filter by a specific committee_id (e.g., "C00893149"). If omitted,
               returns filings across all committees.
    form_type: Filter by form type (e.g., "F1", "F3", "F3X", "F99"). Prefixes work
               server-side (e.g., "F3" matches F3/F3X).
    since:     Minimum receipt date (YYYY-MM-DD or ISO datetime). Maps to
               OpenFEC min_receipt_date.
    until:     Maximum receipt date (YYYY-MM-DD or ISO datetime). Maps to
               OpenFEC max_receipt_date.
    per_page:  Page size for OpenFEC requests (API max ~100). Default 10.
    pages:     Number of pages to scan (per_page * pages maximum rows). Default 1.
    with_totals: If true, enrich F3* filings (with committee_id & file_number)
               with processed report totals. Default false.
    show_urls: If true, include fec/pdf/html/csv URLs (when available). Default false.
    """
    committee: Optional[str] = Field(default=None)
    form_type: Optional[str] = Field(default=None)
    since: Optional[str] = Field(default=None)
    until: Optional[str] = Field(default=None)
    per_page: int = Field(default=10, ge=1, le=100)
    pages: int = Field(default=1, ge=1)
    with_totals: bool = Field(default=False)
    show_urls: bool = Field(default=False)


class FilingTotals(BaseModel):
    """Optional processed totals for a report."""
    total_receipts: Optional[float] = None
    total_disbursements: Optional[float] = None
    cash_on_hand_end_period: Optional[float] = None


class FilingRow(BaseModel):
    """One e-file row (common fields + URLs + optional totals)."""
    committee_id: Optional[str] = None
    committee_name: Optional[str] = None
    form_type: Optional[str] = None
    file_number: Optional[int] = None
    fec_file_id: Optional[str] = None

    receipt_date: Optional[str] = None
    filed_date: Optional[str] = None
    coverage_start_date: Optional[str] = None
    coverage_end_date: Optional[str] = None
    load_timestamp: Optional[str] = None
    amendment_number: Optional[int] = None
    amends_file: Optional[int] = None
    beginning_image_number: Optional[str] = None
    ending_image_number: Optional[str] = None

    # URLs (populated only if show_urls=True)
    fec_url: Optional[HttpUrl] = None
    pdf_url: Optional[HttpUrl] = None
    html_url: Optional[HttpUrl] = None
    csv_url: Optional[HttpUrl] = None

    totals: Optional[FilingTotals] = None  # when with_totals=True and available


class LatestFilingsResult(BaseModel):
    """Result list for latest_filings."""
    count: int
    filings: List[FilingRow]


# ----------------------------
# FastMCP Server
# ----------------------------

mcp = FastMCP("fec-info")


@mcp.tool(name="latest_filings",
    description=(
        "Retrieve near real-time e-file filings from OpenFEC with optional "
        "processed totals enrichment. Paginate with 'per_page' and 'pages'."
    ),
)
def latest_filings_tool(params: LatestFilingsParams) -> LatestFilingsResult:
    """
    Fetch latest filings from OpenFEC. Reads API key from FEC_API_KEY or OPENFEC_API_KEY.
    """
    api_key = os.getenv("FEC_API_KEY") or os.getenv("OPENFEC_API_KEY")
    if not api_key:
        # the demo key works but has lower rate limits
        api_key = "DEMO_KEY"
        #raise RuntimeError("Missing OpenFEC API key. Set FEC_API_KEY or OPENFEC_API_KEY.")

    client = OpenFECClient(api_key=api_key)
    rows = client.latest_filings(
        committee_id=params.committee,
        form_type=params.form_type,
        min_receipt_date=params.since,
        max_receipt_date=params.until,
        per_page=params.per_page,
        pages=params.pages,
    )

    out: List[FilingRow] = []
    for r in rows:
        item = FilingRow(
            committee_id=r.committee_id,
            committee_name=r.committee_name,
            form_type=r.form_type,
            file_number=r.file_number,
            fec_file_id=r.fec_file_id,
            receipt_date=r.receipt_date,
            filed_date=getattr(r, "filed_date", None),
            coverage_start_date=r.coverage_start_date,
            coverage_end_date=r.coverage_end_date,
            load_timestamp=getattr(r, "load_timestamp", None),
            amendment_number=getattr(r, "amendment_number", None),
            amends_file=getattr(r, "amends_file", None),
            beginning_image_number=getattr(r, "beginning_image_number", None),
            ending_image_number=getattr(r, "ending_image_number", None),
        )

        if params.show_urls:
            item.fec_url = r.fec_url
            item.pdf_url = r.pdf_url
            item.html_url = r.html_url
            item.csv_url = r.csv_url

        # Optional enrichment (F3*)
        if (
            params.with_totals
            and r.committee_id
            and r.file_number
            and (r.form_type or "").upper().startswith("F3")
        ):
            rpt = client.report_totals_by_file_number(r.committee_id, r.file_number)
            if rpt:
                item.totals = FilingTotals(
                    total_receipts=float(rpt.total_receipts or 0.0),
                    total_disbursements=float(rpt.total_disbursements or 0.0),
                    cash_on_hand_end_period=float(rpt.cash_on_hand_end_period or 0.0),
                )

        out.append(item)

    return LatestFilingsResult(count=len(out), filings=out)


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == "__main__":
    # FastMCP uses stdio by default when run as a script.
    mcp.run()
