# openfec_client.py
from __future__ import annotations

import os
import time
import random
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import requests
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, conint

OPENFEC_BASE_URL = "https://api.open.fec.gov/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_PER_PAGE = 100  # API max, reduces request count


# ---------------- Pydantic Models ----------------

class FECModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Pagination(FECModel):
    page: Optional[int] = None
    pages: Optional[int] = None
    per_page: Optional[int] = None
    count: Optional[int] = None
    count_estimate: Optional[int] = None
    is_count_exact: Optional[bool] = None
    count_exceed_limit: Optional[bool] = None


class APIEnvelope(FECModel):
    status: Optional[str] = None
    results: List[Dict[str, Any]] = Field(default_factory=list)
    pagination: Optional[Pagination] = None
    api_version: Optional[str] = None
    last_updated: Optional[str] = None


class EfileFiling(FECModel):
    # New/explicit fields from the JSON you provided
    amendment_number: Optional[int] = None
    amends_file: Optional[int] = None
    beginning_image_number: Optional[str] = None
    ending_image_number: Optional[str] = None
    filed_date: Optional[str] = None                # e.g. "2025-09-21"
    load_timestamp: Optional[str] = None            # e.g. "2025-09-21T13:41:00"

    # Existing/common fields
    committee_id: Optional[str] = None
    committee_name: Optional[str] = None
    form_type: Optional[str] = None
    receipt_date: Optional[str] = None              # e.g. "2025-09-21T13:40:55"
    coverage_start_date: Optional[str] = None
    coverage_end_date: Optional[str] = None
    file_number: Optional[int] = None
    fec_file_id: Optional[str] = None

    # Amendment helpers (kept for compatibility; may or may not be present)
    is_amended: Optional[bool] = None
    amendment_chain: Optional[List[int]] = None

    # URLs
    fec_url: Optional[HttpUrl] = None
    pdf_url: Optional[HttpUrl] = None
    html_url: Optional[HttpUrl] = None
    csv_url: Optional[HttpUrl] = None



class ScheduleAItem(FECModel):
    # Itemized receipts (can be negative for refunds/chargebacks)
    committee_id: Optional[str] = None
    committee_name: Optional[str] = None
    recipient_committee_type: Optional[str] = None
    contributor_name: Optional[str] = None
    contributor_first_name: Optional[str] = None
    contributor_last_name: Optional[str] = None
    contributor_middle_name: Optional[str] = None
    contributor_occupation: Optional[str] = None
    contributor_employer: Optional[str] = None
    contributor_city: Optional[str] = None
    contributor_state: Optional[str] = None
    contributor_zip: Optional[str] = None
    contribution_receipt_date: Optional[str] = None
    contribution_receipt_amount: Optional[float] = None
    two_year_transaction_period: Optional[int] = Field(default=None, ge=1976)
    memoed_subtotal: Optional[bool] = None
    is_individual: Optional[bool] = None
    image_number: Optional[str] = None
    file_number: Optional[int] = None


class ScheduleBItem(FECModel):
    # Itemized disbursements (can be negative for refunds/voids)
    sub_id: Optional[int] = None
    image_number: Optional[str] = None
    file_number: Optional[int] = None
    committee_id: Optional[str] = None               # spender (donor committee)
    committee_name: Optional[str] = None
    recipient_committee_id: Optional[str] = None     # recipient (committee)
    recipient_name: Optional[str] = None
    recipient_committee_type: Optional[str] = None   # <-- add this
    disbursement_date: Optional[str] = None
    disbursement_amount: Optional[float] = None
    disbursement_purpose: Optional[str] = None
    memoed_subtotal: Optional[bool] = None
    two_year_transaction_period: Optional[int] = Field(default=None, ge=1976)



class ScheduleBByRecipientAgg(FECModel):
    recipient_committee_id: Optional[str] = None
    recipient_name: Optional[str] = None
    total: Optional[float] = None
    count: Optional[int] = None
    cycle: Optional[int] = None
    committee_id: Optional[str] = None
    committee_name: Optional[str] = None


class CommitteeReport(FECModel):
    committee_id: Optional[str] = None
    committee_name: Optional[str] = None
    form_type: Optional[str] = None
    report_type: Optional[str] = None
    report_type_full: Optional[str] = None
    coverage_start_date: Optional[str] = None
    coverage_end_date: Optional[str] = None
    receipt_date: Optional[str] = None
    file_number: Optional[int] = None
    total_receipts: Optional[float] = None
    total_disbursements: Optional[float] = None
    cash_on_hand_end_period: Optional[float] = None
    debts_owed_by_committee: Optional[float] = None


class CommitteeCandidateLink(FECModel):
    candidate_id: Optional[str] = None
    name: Optional[str] = None
    office: Optional[str] = None
    party: Optional[str] = None


class CandidateCommittee(FECModel):
    committee_id: Optional[str] = None
    name: Optional[str] = None
    designation: Optional[str] = None  # 'P','A','L','J', etc.
    committee_type: Optional[str] = None


# -- Replace your CommitteeTotals with this --
class CommitteeTotals(FECModel):
    # Identity / metadata
    cycle: Optional[int] = None
    committee_id: Optional[str] = None
    committee_name: Optional[str] = None
    treasurer_name: Optional[str] = None
    committee_type: Optional[str] = None
    committee_type_full: Optional[str] = None
    committee_designation: Optional[str] = None
    committee_designation_full: Optional[str] = None
    committee_state: Optional[str] = None
    filing_frequency: Optional[str] = None
    filing_frequency_full: Optional[str] = None
    organization_type: Optional[str] = None
    organization_type_full: Optional[str] = None
    party_full: Optional[str] = None

    # Coverage / timing
    coverage_start_date: Optional[str] = None
    coverage_end_date: Optional[str] = None
    transaction_coverage_date: Optional[str] = None
    first_f1_date: Optional[str] = None
    first_file_date: Optional[str] = None
    last_report_type_full: Optional[str] = None
    last_report_year: Optional[int] = None
    last_beginning_image_number: Optional[str] = None

    # Cash / debts (point-in-time)
    cash_on_hand_beginning_period: Optional[float] = None
    last_cash_on_hand_end_period: Optional[float] = None
    cash_on_hand_end_period: Optional[float] = None
    last_debts_owed_by_committee: Optional[float] = None
    last_debts_owed_to_committee: Optional[float] = None
    debts_owed_by_committee: Optional[float] = None

    # Aggregate totals (cycle / FEC aggregation)
    receipts: Optional[float] = None
    fed_receipts: Optional[float] = None
    total_exp_subject_limits: Optional[float] = None
    exp_subject_limits: Optional[float] = None
    exp_prior_years_subject_limits: Optional[float] = None
    disbursements: Optional[float] = None
    fed_disbursements: Optional[float] = None
    net_contributions: Optional[float] = None
    net_operating_expenditures: Optional[float] = None

    # Contributions (receipts side)
    contributions: Optional[float] = None
    individual_contributions: Optional[float] = None
    individual_itemized_contributions: Optional[float] = None
    individual_unitemized_contributions: Optional[float] = None
    political_party_committee_contributions: Optional[float] = None
    other_political_committee_contributions: Optional[float] = None
    contribution_refunds: Optional[float] = None
    refunded_individual_contributions: Optional[float] = None
    refunded_other_political_committee_contributions: Optional[float] = None
    refunded_political_party_committee_contributions: Optional[float] = None

    # Receipts — other categories
    federal_funds: Optional[float] = None
    other_fed_receipts: Optional[float] = None
    other_receipts: Optional[float] = None  # sometimes present on other committee types
    offsets_to_operating_expenditures: Optional[float] = None

    # Transfers (incoming/outgoing)
    total_transfers: Optional[float] = None
    transfers_from_affiliated_party: Optional[float] = None
    transfers_from_nonfed_account: Optional[float] = None
    transfers_from_nonfed_levin: Optional[float] = None
    transfers_to_affiliated_committee: Optional[float] = None

    # Loans (received/made/repayments)
    all_loans_received: Optional[float] = None
    loan_repayments_made: Optional[float] = None
    loan_repayments_received: Optional[float] = None
    loans_made: Optional[float] = None
    loans_and_loan_repayments_made: Optional[float] = None
    loans_and_loan_repayments_received: Optional[float] = None

    # Disbursements (operating/other)
    operating_expenditures: Optional[float] = None
    fed_operating_expenditures: Optional[float] = None
    other_disbursements: Optional[float] = None
    other_fed_operating_expenditures: Optional[float] = None
    fundraising_disbursements: Optional[float] = None

    # Independent/coordinated/party activity
    independent_expenditures: Optional[float] = None
    coordinated_expenditures_by_party_committee: Optional[float] = None
    fed_election_activity: Optional[float] = None
    non_allocated_fed_election_activity: Optional[float] = None
    shared_fed_activity: Optional[float] = None
    shared_fed_activity_nonfed: Optional[float] = None
    shared_fed_operating_expenditures: Optional[float] = None
    shared_nonfed_operating_expenditures: Optional[float] = None
    fed_candidate_committee_contributions: Optional[float] = None
    fed_candidate_contribution_refunds: Optional[float] = None

    # Convention (rare; keep for compatibility)
    convention_exp: Optional[float] = None
    itemized_convention_exp: Optional[float] = None
    unitemized_convention_exp: Optional[float] = None
    refunds_relating_convention_exp: Optional[float] = None
    itemized_refunds_relating_convention_exp: Optional[float] = None
    unitemized_refunds_relating_convention_exp: Optional[float] = None

    # Other income / refunds (granular)
    itemized_other_income: Optional[float] = None
    unitemized_other_income: Optional[float] = None
    other_refunds: Optional[float] = None
    itemized_other_refunds: Optional[float] = None
    unitemized_other_refunds: Optional[float] = None

    # Other disbursements (granular)
    itemized_other_disb: Optional[float] = None
    unitemized_other_disb: Optional[float] = None

    # Derived/percent fields (if present)
    individual_contributions_percent: Optional[float] = None
    party_and_other_committee_contributions_percent: Optional[float] = None
    contributions_ie_and_party_expenditures_made_percent: Optional[float] = None
    operating_expenditures_percent: Optional[float] = None

    # Sponsors (rare)
    sponsor_candidate_ids: Optional[str] = None
    sponsor_candidate_list: Optional[List[str]] = None





# -- Replace your CommitteeAbout with this --
class CommitteeAbout(FECModel):
    # Core identity
    committee_id: str
    name: Optional[str] = None

    # Leadership
    treasurer_name: Optional[str] = None

    # Type / designation (short + full)
    committee_type: Optional[str] = None
    committee_type_full: Optional[str] = None
    designation: Optional[str] = None
    designation_full: Optional[str] = None

    # Filing (fixes your error)
    filing_frequency: Optional[str] = None

    # Party
    party: Optional[str] = None
    party_full: Optional[str] = None

    # Location
    state: Optional[str] = None
    state_full: Optional[str] = None
    city: Optional[str] = None
    zip: Optional[str] = None
    street_1: Optional[str] = None
    street_2: Optional[str] = None

    # Misc
    website: Optional[str] = None




class CandidateHit(FECModel):
    candidate_id: str
    name: str
    office: Optional[str] = None
    party: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    election_years: Optional[List[int]] = None


class DonorToCandidateAgg(FECModel):
    donor_committee_id: str
    donor_committee_name: Optional[str] = None
    total: float
    count: int


class CommitteeSummary(FECModel):
    about: Optional[CommitteeAbout] = None
    totals: Optional[CommitteeTotals] = None
    latest_report: Optional[CommitteeReport] = None


# ---------------- Client ----------------

class OpenFECClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENFEC_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        user_agent: str = "openfec-pydantic-client/1.0.0",
        retry_attempts: int = 5,
        retry_backoff: float = 1.5,
    ) -> None:
        self.api_key = api_key or os.getenv("FEC_API_KEY") or os.getenv("OPENFEC_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenFEC API key. Set FEC_API_KEY or pass api_key=...")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff = retry_backoff

        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": user_agent,
            "X-Api-Key": self.api_key,
        })

    # ---------- helpers ----------

    @staticmethod
    def _normalize_cycle(cycle: Optional[int]) -> Optional[int]:
        if cycle is None:
            return None
        if cycle < 1976:
            raise ValueError("cycle must be >= 1976")
        return cycle if (cycle % 2 == 0) else (cycle - 1)

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> APIEnvelope:
        url = f"{self.base_url}/{path.lstrip('/')}"
        params = params or {}
        params.setdefault("per_page", DEFAULT_PER_PAGE)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            sleep_s = float(ra)
                        except ValueError:
                            sleep_s = self.retry_backoff * attempt
                    else:
                        sleep_s = self.retry_backoff * attempt + random.uniform(0, 0.5 * attempt)
                    time.sleep(sleep_s)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return APIEnvelope.model_validate(data)
            except (requests.RequestException, ValidationError) as e:
                last_err = e
                if attempt >= self.retry_attempts:
                    raise
                time.sleep(self.retry_backoff * attempt + random.uniform(0, 0.2))
        assert last_err is not None
        raise last_err  # type: ignore

    def _paginate(self, path: str, params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        page = int(params.get("page", 1) or 1)
        per_page = int(params.get("per_page", DEFAULT_PER_PAGE) or DEFAULT_PER_PAGE)
        while True:
            envelope = self._request(path, {**params, "page": page, "per_page": per_page})
            results = envelope.results or []
            for row in results:
                yield row
            p = envelope.pagination
            if p and p.page is not None and p.pages is not None:
                if p.page >= p.pages:
                    break
                page = p.page + 1
                continue
            if len(results) < per_page or per_page <= 0:
                break
            page += 1

    # ---------- committee info (About + Totals + Summary) ----------

    def committee_about(self, committee_id: str) -> Optional[CommitteeAbout]:
        env = self._request(f"committee/{committee_id}/", {"per_page": 1})
        rows = env.results or []
        return CommitteeAbout.model_validate(rows[0]) if rows else None

    def committee_totals(self, committee_id: str, *, cycle: Optional[int] = None) -> Optional[CommitteeTotals]:
        params: Dict[str, Any] = {"per_page": 1}
        if cycle:
            params["cycle"] = self._normalize_cycle(cycle)
        env = self._request(f"committee/{committee_id}/totals/", params)
        rows = env.results or []
        return CommitteeTotals.model_validate(rows[0]) if rows else None

    def committee_summary(self, committee_id: str, *, cycle: Optional[int] = None) -> CommitteeSummary:
        return CommitteeSummary(
            about=self.committee_about(committee_id),
            totals=self.committee_totals(committee_id, cycle=cycle),
            latest_report=self.latest_committee_report(committee_id, cycle=cycle),
        )

    def fec_committee_page_url(self, committee_id: str) -> str:
        return f"https://www.fec.gov/data/committee/{committee_id}/"

    # ---------- reports / filings ----------

    def report_totals_by_file_number(self, committee_id: str, file_number: int) -> Optional[CommitteeReport]:
        """
        Best-effort: fetch processed report totals for a given committee + file_number.
        Returns None if the report isn't processed yet or not found.
        """
        params = {"per_page": 1, "file_number": file_number}
        env = self._request(f"committee/{committee_id}/reports/", params)
        rows = env.results or []
        if not rows:
            return None
        return CommitteeReport.model_validate(rows[0])

    def latest_committee_report(self, committee_id: str, *, cycle: Optional[int] = None, form_type: Optional[str] = None) -> Optional[CommitteeReport]:
        params: Dict[str, Any] = {"sort": "-coverage_end_date", "per_page": 1}
        if cycle:
            params["two_year_transaction_period"] = self._normalize_cycle(cycle)
        if form_type:
            params["form_type"] = form_type
        env = self._request(f"committee/{committee_id}/reports/", params)
        rows = env.results or []
        if not rows:
            return None
        return CommitteeReport.model_validate(rows[0])

    def latest_filings(self, *, committee_id: Optional[str] = None, form_type: Optional[str] = None,
                       min_receipt_date: Optional[str] = None, max_receipt_date: Optional[str] = None,
                       per_page: int = 50, pages: int = 1) -> List[EfileFiling]:
        params: Dict[str, Any] = {"sort": "-receipt_date", "per_page": per_page}
        if committee_id: params["committee_id"] = committee_id
        if form_type: params["form_type"] = form_type
        if min_receipt_date: params["min_receipt_date"] = min_receipt_date
        if max_receipt_date: params["max_receipt_date"] = max_receipt_date
        out: List[EfileFiling] = []
        for row in self._paginate("efile/filings/", params):
            out.append(EfileFiling.model_validate(row))
            if len(out) >= per_page * max(1, pages):
                break
        return out

    # ---------- Schedule A ----------

    def contributions_to_committee(self, committee_id: str, *, is_individual: Optional[bool] = None,
                                   cycle: Optional[int] = None, since: Optional[str] = None,
                                   per_page: int = 50, limit: Optional[int] = 200) -> Generator[ScheduleAItem, None, None]:
        params: Dict[str, Any] = {"committee_id": committee_id, "sort": "-contribution_receipt_date", "per_page": per_page}
        if is_individual is not None: params["is_individual"] = str(is_individual).lower()
        if cycle: params["two_year_transaction_period"] = self._normalize_cycle(cycle)
        if since: params["min_date"] = since
        count = 0
        for row in self._paginate("schedules/schedule_a/", params):
            yield ScheduleAItem.model_validate(row)
            count += 1
            if limit and count >= limit: break

    def donor_activity(self, *, contributor_name: Optional[str] = None, contributor_employer: Optional[str] = None,
                       state: Optional[str] = None, cycle: Optional[int] = None,
                       per_page: int = 50, limit: Optional[int] = 200) -> Generator[ScheduleAItem, None, None]:
        if not (contributor_name or contributor_employer):
            raise ValueError("Provide contributor_name or contributor_employer.")
        params: Dict[str, Any] = {"sort": "-contribution_receipt_date", "per_page": per_page}
        if contributor_name: params["contributor_name"] = contributor_name
        if contributor_employer: params["contributor_employer"] = contributor_employer
        if state: params["contributor_state"] = state
        if cycle: params["two_year_transaction_period"] = self._normalize_cycle(cycle)
        count = 0
        for row in self._paginate("schedules/schedule_a/", params):
            yield ScheduleAItem.model_validate(row)
            count += 1
            if limit and count >= limit: break

    # ---------- Schedule B ----------

    def committee_to_committee(self, *, committee_id: Optional[str] = None, recipient_committee_id: Optional[str] = None,
                               cycle: Optional[int] = None, per_page: int = DEFAULT_PER_PAGE, limit: Optional[int] = 200) -> Generator[ScheduleBItem, None, None]:
        if not (committee_id or recipient_committee_id):
            raise ValueError("Provide committee_id (spender) or recipient_committee_id (receiver).")
        params: Dict[str, Any] = {"sort": "-disbursement_date", "per_page": per_page}
        if committee_id: params["committee_id"] = committee_id
        if recipient_committee_id: params["recipient_committee_id"] = recipient_committee_id
        if cycle: params["two_year_transaction_period"] = self._normalize_cycle(cycle)
        count = 0
        for row in self._paginate("schedules/schedule_b/", params):
            yield ScheduleBItem.model_validate(row)
            count += 1
            if limit and count >= limit: break

    def schedule_b_by_recipient_id(self, recipient_committee_id: str, *, cycle: int,
                                   per_page: int = DEFAULT_PER_PAGE, limit: Optional[int] = 200) -> Generator[ScheduleBByRecipientAgg, None, None]:
        params: Dict[str, Any] = {"recipient_id": recipient_committee_id, "cycle": self._normalize_cycle(cycle), "per_page": per_page}
        count = 0
        for row in self._paginate("schedules/schedule_b/by_recipient_id/", params):
            yield ScheduleBByRecipientAgg.model_validate(row)
            count += 1
            if limit and count >= limit: break

    # ---------- Candidates & Cohorts ----------

    def search_candidates(self, q: str, *, cycle: Optional[int] = None,
                          per_page: int = 50, limit: Optional[int] = 200) -> List[CandidateHit]:
        params: Dict[str, Any] = {"q": q, "sort": "name", "per_page": per_page}
        if cycle:
            params["cycle"] = self._normalize_cycle(cycle)
        out: List[CandidateHit] = []
        count = 0
        for row in self._paginate("candidates/search/", params):
            out.append(CandidateHit.model_validate(row))
            count += 1
            if limit and count >= limit:
                break
        return out

    @lru_cache(maxsize=2000)
    def _committee_candidate_links(self, committee_id: str) -> List[CommitteeCandidateLink]:
        env = self._request(f"committee/{committee_id}/candidates/", {"per_page": 50})
        return [CommitteeCandidateLink.model_validate(r) for r in (env.results or [])]

    def candidate_committees(self, candidate_id: str, *, cycle: Optional[int] = None, cohort: str = "authorized") -> List[CandidateCommittee]:
        """
        cohort:
          - 'authorized' (default): principal + authorized ('P','A')
          - 'all_linked': all committees returned by OpenFEC for the candidate (incl leadership PAC 'L', JFC 'J', etc.)
        """
        params: Dict[str, Any] = {"per_page": 100}
        if cycle: params["cycle"] = self._normalize_cycle(cycle)
        env = self._request(f"candidate/{candidate_id}/committees/", params)
        committees = [CandidateCommittee.model_validate(r) for r in (env.results or [])]
        if cohort == "authorized":
            committees = [c for c in committees if (c.designation or "").upper() in {"P", "A"}]
        return committees

    def donors_to_candidate_aggregates(self, candidate_id: str, *, cycle: int,
                                       per_page: int = 100, limit: Optional[int] = None,
                                       cohort: str = "authorized") -> List[DonorToCandidateAgg]:
        recips = self.candidate_committees(candidate_id, cycle=cycle, cohort=cohort)
        recipient_ids = [c.committee_id for c in recips if c.committee_id]
        if not recipient_ids:
            return []

        totals: Dict[str, Dict[str, Any]] = {}
        for rcid in recipient_ids:
            params = {"recipient_id": rcid, "cycle": self._normalize_cycle(cycle), "per_page": per_page}
            scanned = 0
            for row in self._paginate("schedules/schedule_b/by_recipient_id/", params):
                donor_id = row.get("committee_id")
                donor_name = row.get("committee_name")
                total_amt = float(row.get("total") or 0.0)
                cnt = int(row.get("count") or 0)
                if not donor_id or total_amt == 0:
                    continue
                if donor_id not in totals:
                    totals[donor_id] = {"donor_committee_id": donor_id, "donor_committee_name": donor_name, "total": 0.0, "count": 0}
                totals[donor_id]["total"] += total_amt
                totals[donor_id]["count"] += cnt
                scanned += 1
                if limit and scanned >= limit:
                    break

        rows = [DonorToCandidateAgg(donor_committee_id=v["donor_committee_id"], donor_committee_name=v.get("donor_committee_name"),
                                    total=round(float(v["total"]), 2), count=int(v["count"])) for v in totals.values()]
        rows.sort(key=lambda x: x.total, reverse=True)
        return rows

    def committee_payments_to_candidate_items(
        self,
        donor_committee_id: str,
        candidate_id: Optional[str] = None,   # <-- now optional
        *,
        cycle: int,
        per_page: int = DEFAULT_PER_PAGE,
        limit: Optional[int] = 200,
        include_memos: bool = False,
        dedupe: bool = True,
        since: Optional[str] = None,
        until: Optional[str] = None,
        scan_limit: Optional[int] = 5000,
        max_pages: Optional[int] = 50,
        match_aggregate: bool = False,
        cohort: str = "authorized",
        dedupe_jfc_transfers: bool = False,
    ) -> Generator[ScheduleBItem, None, None]:
        """
        Itemized payments from donor_committee_id.

        Modes:
        - candidate_id provided: existing behavior (optionally match aggregate cohort).
        - candidate_id omitted: return ALL payments to candidate authorized committees
            (recipient committee types H/S/P) in the given cycle.

        Guardrails/notes:
        - include_memos=False excludes memo rows (recommended to match aggregates)
        - dedupe=True drops duplicates by sub_id (fallback composite key otherwise)
        - since/until: YYYY-MM-DD server-side filters
        - scan_limit / max_pages: bound scanning
        """
        seen: Set[Tuple[Any, ...]] = set()
        yielded = 0
        scanned_total = 0

        def _yield_rows(params_base: Dict[str, Any]) -> Generator[ScheduleBItem, None, None]:
            nonlocal yielded, scanned_total
            page = 1
            while True:
                if max_pages is not None and page > max_pages:
                    break
                envelope = self._request("schedules/schedule_b/", {**params_base, "page": page, "per_page": per_page})
                rows = envelope.results or []
                if not rows:
                    break
                for row in rows:
                    scanned_total += 1
                    if scan_limit and scanned_total > scan_limit:
                        return
                    item = ScheduleBItem.model_validate(row)
                    if dedupe:
                        if item.sub_id is not None:
                            key = ("sub_id", item.sub_id)
                        else:
                            key = (
                                "fallback",
                                item.image_number,
                                float(item.disbursement_amount or 0.0),
                                item.disbursement_date,
                                item.recipient_committee_id,
                            )
                        if key in seen:
                            continue
                        seen.add(key)
                    yield item
                    yielded += 1
                    if limit and yielded >= limit:
                        return
                p = envelope.pagination
                if p and p.page is not None and p.pages is not None:
                    if p.page >= p.pages:
                        break
                    page = p.page + 1
                else:
                    if len(rows) < per_page:
                        break
                    page += 1

        # Shared base filters
        base: Dict[str, Any] = {
            "committee_id": donor_committee_id,
            "two_year_transaction_period": self._normalize_cycle(cycle),
            "sort": "-disbursement_date",
        }
        if not include_memos:
            base["memoed_subtotal"] = "false"
        if since:
            base["min_date"] = since
        if until:
            base["max_date"] = until

        # If no candidate_id: fetch all payments to candidate authorized committees (H/S/P)
        if not candidate_id:
            # The API supports filtering by recipient committee type.
            # Query each type to keep result sizes manageable.
            for r_type in ("H", "S", "P"):
                params = {**base, "recipient_committee_type": r_type}
                yield from _yield_rows(params)
            return

        # Existing candidate-specific behavior below

        # If matching "authorized" aggregate exactly, use candidate recipient_id
        if match_aggregate and cohort == "authorized":
            params = {**base, "recipient_id": candidate_id}
            yield from _yield_rows(params)
            return

        # Otherwise iterate authorized/all linked recipient committees for the candidate
        committees = self.candidate_committees(candidate_id, cycle=cycle, cohort=cohort)
        recipient_ids = [c.committee_id for c in committees if c.committee_id]
        for rcid in recipient_ids:
            params = {**base, "recipient_committee_id": rcid}
            yield from _yield_rows(params)
