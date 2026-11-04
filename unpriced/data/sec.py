"""SEC EDGAR client (free, point-in-time by construction).

Fair-access rules (sec.gov "Accessing EDGAR Data"): at most 10 requests/second and a declared
User-Agent with contact details. The contact email comes from an environment variable
(a GitHub secret), never from the repository, because the repo is public.

Timestamps: EDGAR's `acceptanceDateTime` is treated as US/Eastern. If it were really UTC the
rule below would only ever make an event *later*, never earlier, so it cannot leak the future.
"""
from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import requests

from ..config import section
from ..util import log

L = log("sec")

FACT_CONCEPTS = {
    # name -> ordered list of (taxonomy, tag) fallbacks
    "revenue": [("us-gaap", "Revenues"), ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                ("us-gaap", "SalesRevenueNet"), ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax")],
    "cost_of_revenue": [("us-gaap", "CostOfRevenue"), ("us-gaap", "CostOfGoodsAndServicesSold")],
    "gross_profit": [("us-gaap", "GrossProfit")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")],
    "operating_income": [("us-gaap", "OperatingIncomeLoss")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
            ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "assets": [("us-gaap", "Assets")],
    "equity": [("us-gaap", "StockholdersEquity"),
               ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")],
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasicAndDiluted"),
                    ("us-gaap", "EarningsPerShareBasic")],
    "shares_out": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
}
FLOW = {"revenue", "cost_of_revenue", "gross_profit", "net_income", "operating_income", "cfo", "eps_diluted"}


class SecUnavailable(RuntimeError):
    pass


class SecClient:
    BASE = "https://www.sec.gov"
    DATA = "https://data.sec.gov"

    def __init__(self, contact: str | None = None) -> None:
        cfg = section("sec")
        contact = contact or os.environ.get(cfg["contactEnvVar"], "").strip()
        if not contact:
            raise SecUnavailable(f"set {cfg['contactEnvVar']} to enable SEC EDGAR access")
        self.session = requests.Session()
        # SEC's documented format is "<name> <email>"; extra punctuation or URLs can be refused.
        self.session.headers.update({"User-Agent": f"{cfg['userAgentApp']} {contact}", "Accept-Encoding": "gzip, deflate",
                                     "Accept": "application/json, text/html;q=0.9, */*;q=0.8"})
        self._explained = False
        self.min_interval = 1.0 / float(cfg["maxRequestsPerSecond"])
        self._last = 0.0
        self._lock = threading.Lock()

    def _wait(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()

    def get(self, url: str, allow_404: bool = True) -> requests.Response | None:
        for attempt in range(5):
            self._wait()
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as exc:
                L.warning("SEC request error %s (%s)", url, exc)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code == 404 and allow_404:
                return None
            if r.status_code in (403, 429, 500, 502, 503):
                wait = 5 * (2 ** attempt)
                if not self._explained:   # SEC explains refusals in the body (e.g. "Undeclared Automated Tool")
                    snippet = re.sub(r"\s+", " ", html_to_text(r.text))[:240]
                    L.warning("SEC refusal body: %s", snippet)
                    self._explained = True
                L.warning("SEC %s for %s; backing off %ss", r.status_code, url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
        raise SecUnavailable(f"SEC request kept failing: {url}")

    # ---- reference data -------------------------------------------------
    def tickers(self) -> pd.DataFrame:
        r = self.get(f"{self.BASE}/files/company_tickers_exchange.json", allow_404=False)
        js = r.json()
        df = pd.DataFrame(js["data"], columns=js["fields"])
        df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
        df["exchange"] = df["exchange"].fillna("").astype(str)
        return df

    # ---- filings ----------------------------------------------------------
    @staticmethod
    def _filings_frame(block: dict, cik: int) -> pd.DataFrame:
        if not block or not block.get("accessionNumber"):
            return pd.DataFrame()
        df = pd.DataFrame({
            "accession": block.get("accessionNumber", []),
            "form": block.get("form", []),
            "filing_date": block.get("filingDate", []),
            "report_date": block.get("reportDate", [None] * len(block["accessionNumber"])),
            "acceptance": block.get("acceptanceDateTime", [None] * len(block["accessionNumber"])),
            "items": block.get("items", [""] * len(block["accessionNumber"])),
            "primary_doc": block.get("primaryDocument", [""] * len(block["accessionNumber"])),
        })
        df["cik"] = int(cik)
        return df

    def submissions(self, cik: int, include_older: bool = False) -> tuple[dict, pd.DataFrame]:
        r = self.get(f"{self.DATA}/submissions/CIK{int(cik):010d}.json")
        if r is None:
            return {}, pd.DataFrame()
        js = r.json()
        meta = {
            "cik": int(cik), "name": js.get("name"), "sic": int(js["sic"]) if str(js.get("sic") or "").isdigit() else None,
            "sicDescription": js.get("sicDescription"), "tickers": js.get("tickers", []), "exchanges": js.get("exchanges", []),
            "entityType": js.get("entityType"),
        }
        frames = [self._filings_frame(js.get("filings", {}).get("recent", {}), cik)]
        if include_older:
            for f in js.get("filings", {}).get("files", []):
                rr = self.get(f"{self.DATA}/submissions/{f['name']}")
                if rr is not None:
                    frames.append(self._filings_frame(rr.json(), cik))
        filings = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
        return meta, filings

    def daily_index(self, day: pd.Timestamp) -> pd.DataFrame:
        q = (day.month - 1) // 3 + 1
        url = f"{self.BASE}/Archives/edgar/daily-index/{day.year}/QTR{q}/master.{day:%Y%m%d}.idx"
        r = self.get(url)
        if r is None:
            return pd.DataFrame(columns=["cik", "company", "form", "filing_date", "filename"])
        rows = []
        for line in r.text.splitlines():
            parts = line.split("|")
            if len(parts) == 5 and parts[0].isdigit():
                rows.append(parts)
        df = pd.DataFrame(rows, columns=["cik", "company", "form", "filing_date", "filename"])
        df["cik"] = df["cik"].astype(int)
        return df

    def companyfacts(self, cik: int) -> dict | None:
        r = self.get(f"{self.DATA}/api/xbrl/companyfacts/CIK{int(cik):010d}.json")
        return None if r is None else r.json()

    # ---- documents --------------------------------------------------------
    def filing_files(self, cik: int, accession: str) -> list[str]:
        acc = accession.replace("-", "")
        r = self.get(f"{self.BASE}/Archives/edgar/data/{int(cik)}/{acc}/index.json")
        if r is None:
            return []
        return [it.get("name", "") for it in r.json().get("directory", {}).get("item", [])]

    def document_text(self, cik: int, accession: str, filename: str, max_chars: int = 200_000) -> str:
        acc = accession.replace("-", "")
        r = self.get(f"{self.BASE}/Archives/edgar/data/{int(cik)}/{acc}/{filename}")
        if r is None:
            return ""
        return html_to_text(r.text)[:max_chars]


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|ix:header)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</h\d>", "\n", raw)
    text = _TAG.sub(" ", raw)
    text = (text.replace("&nbsp;", " ").replace("&#160;", " ").replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&#8217;", "'").replace("&#8220;", '"').replace("&#8221;", '"'))
    text = _WS.sub(" ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def parse_acceptance(series: pd.Series) -> pd.Series:
    """EDGAR acceptance timestamps as naive US/Eastern datetimes."""
    s = series.fillna("").astype(str).str.replace("Z", "", regex=False).str.replace(".000", "", regex=False)
    return pd.to_datetime(s, errors="coerce")


def reaction_session(filing_date: pd.Series, acceptance: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """First session whose close happens after the information became public.

    Accepted before 16:00 ET on a session day -> that session; otherwise the next session.
    """
    acc = parse_acceptance(acceptance)
    fdate = pd.to_datetime(filing_date, errors="coerce")
    stamp = acc.where(acc.notna(), fdate + pd.Timedelta(hours=17))   # unknown time -> assume after close
    stamp = stamp.where(stamp.dt.normalize() >= fdate, fdate + pd.Timedelta(hours=17))
    day = stamp.dt.normalize()
    after_close = stamp.dt.time >= dtime(16, 0)
    target = day + pd.to_timedelta(after_close.astype(int), unit="D")
    pos = sessions.searchsorted(target.values.astype("datetime64[ns]"), side="left")
    pos = np.clip(pos, 0, len(sessions) - 1)
    out = pd.Series(sessions[pos], index=filing_date.index)
    out[target.isna() | (target > sessions[-1])] = pd.NaT
    return out


def facts_table(js: dict, cik: int) -> pd.DataFrame:
    """Flatten selected XBRL concepts with their first-filed date (point-in-time)."""
    rows = []
    facts = js.get("facts", {}) if js else {}
    for name, options in FACT_CONCEPTS.items():
        for prio, (tax, tag) in enumerate(options):
            node = facts.get(tax, {}).get(tag)
            if not node:
                continue
            for unit, vals in node.get("units", {}).items():
                if name == "shares_out" and unit != "shares":
                    continue
                if name == "eps_diluted" and "/shares" not in unit:
                    continue
                if name not in ("shares_out", "eps_diluted") and unit != "USD":
                    continue
                for v in vals:
                    if v.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "40-F"):
                        continue
                    rows.append((cik, name, tag, prio, v.get("start"), v.get("end"), v.get("val"), v.get("fy"), v.get("fp"),
                                 v.get("form"), v.get("filed"), v.get("frame"), v.get("accn")))
    # Companies switch tags over time (e.g. SalesRevenueNet -> ASC 606 tags); keep all and let the
    # fundamentals builder prefer the highest-priority tag per period.
    df = pd.DataFrame(rows, columns=["cik", "concept", "tag", "prio", "start", "end", "val", "fy", "fp", "form", "filed", "frame", "accn"])
    if df.empty:
        return df
    for c in ["start", "end", "filed"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    return df.dropna(subset=["end", "filed", "val"])
