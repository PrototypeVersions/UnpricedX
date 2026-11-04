"""Candidate stock list from Nasdaq Trader's public symbol directory (no key or email needed).

nasdaqlisted.txt covers Nasdaq listings; otherlisted.txt covers NYSE, NYSE American, Arca, Cboe.
Both flag ETFs and test issues, which makes it easy to keep only operating-company equity.
"""
from __future__ import annotations

import io
import re

import pandas as pd
import requests

from ..util import log

L = log("symbols")

URLS = {
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
KEEP_NAME = re.compile(r"(?i)common|ordinary share|ordinary stock|depositary share|class [a-c] |capital stock|shares of beneficial interest")
DROP_NAME = re.compile(
    r"(?i)warrant|\bunits?\b|\bright(?:s)?\b|preferred|pfd|\bnotes?\b|debenture|%|subordinated|"
    r"\betf\b|\betn\b|\bfund\b|acquisition corp|blank check|contingent value"
)
EXCHANGES = {"Q": "Nasdaq", "N": "NYSE", "A": "NYSE American"}   # Arca/Cboe/IEX are ETF-heavy: excluded


def _read(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60, headers={"User-Agent": "Unpriced research lab (github.com/PrototypeVersions/UnpricedX)"})
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("File Creation Time")]
    # keep_default_na=False: real tickers such as "NA" must not turn into missing values
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, keep_default_na=False, na_values=[])


def parse(nasdaq: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    a = pd.DataFrame({
        "ticker": nasdaq["Symbol"], "name": nasdaq["Security Name"], "exchange_code": "Q",
        "etf": nasdaq["ETF"], "test": nasdaq["Test Issue"], "status": nasdaq.get("Financial Status", ""),
    })
    b = pd.DataFrame({
        "ticker": other["ACT Symbol"], "name": other["Security Name"], "exchange_code": other["Exchange"],
        "etf": other["ETF"], "test": other["Test Issue"], "status": "",
    })
    df = pd.concat([a, b], ignore_index=True)
    df = df[(df["etf"] != "Y") & (df["test"] != "Y") & df["exchange_code"].isin(EXCHANGES)]
    df = df[df["name"].str.contains(KEEP_NAME) & ~df["name"].str.contains(DROP_NAME)]
    df = df[~df["status"].isin(["D", "E", "H"])]                     # deficient / delinquent / both
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.upper()
    df = df[(df["ticker"].str.len() > 0) & ~df["ticker"].str.contains(r"[\$\^/= ]", regex=True)]
    df["exchange"] = df["exchange_code"].map(EXCHANGES)
    return df[["ticker", "name", "exchange"]].drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def fetch_candidates() -> pd.DataFrame:
    df = parse(_read(URLS["nasdaq"]), _read(URLS["other"]))
    L.info("symbol directory: %d candidate common stocks", len(df))
    return df
