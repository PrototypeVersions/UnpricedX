"""Candidate list and point-in-time universe membership.

Candidates are US-listed common stocks from SEC's ticker file (exchange + name + SIC filters).
Membership on each date uses only information available that day: price above a floor,
enough trading history, and rank by trailing median dollar volume. No hindsight about which
companies later became large is used to pick the universe on any given date.

Known limitation (documented on the site): free data only covers companies that are still
listed today, so delisted names are missing from history (survivorship bias).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import section

_BAD_SUFFIX = re.compile(r"^[A-Z]{4}[WURZ]$")   # Nasdaq warrants/units/rights


def filter_candidates(tickers: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = section("data")
    df = tickers.copy()
    df = df[~df["exchange"].isin(cfg["excludeExchanges"])]
    df = df[~df["name"].fillna("").str.contains(cfg["excludeNamePattern"], regex=True)]
    tick = df["ticker"].astype(str)
    df = df[~tick.str.contains(r"[\^/ ]", regex=True)]
    df = df[~((df["exchange"] == "Nasdaq") & tick.str.match(_BAD_SUFFIX))]
    df = df[~tick.str.contains(r"-(?:W|WS|WT|U|UN|R|RT|P[A-Z]?)$", regex=True)]
    if meta is not None and len(meta):
        sic = meta.set_index("cik")["sic"]
        df = df[~df["cik"].map(sic).isin(cfg["excludeSic"])]
    return df.drop_duplicates("ticker").reset_index(drop=True)


def membership(mats: dict[str, pd.DataFrame], exclude: list[str]) -> dict[str, pd.DataFrame]:
    """Boolean date x ticker matrices: `research` and `tradable` universes, plus liquidity rank."""
    ucfg = section("universe")
    close = mats.get("raw_close", mats["close"])   # price floor applies to the price that printed
    adj, dv = mats["adj_close"], mats["dollar_volume"]
    cols = [c for c in close.columns if c not in set(exclude)]
    close, adj, dv = close[cols], adj[cols], dv[cols]
    win = int(ucfg["liquidityWindow"])
    med_dv = dv.rolling(win, min_periods=int(win * 0.8)).median()
    history = adj.notna().rolling(int(ucfg["minHistorySessions"]), min_periods=1).sum()
    ok = (close >= float(ucfg["minPrice"])) & (history >= int(ucfg["minHistorySessions"]) * 0.9) & med_dv.notna() & adj.notna()
    rank = med_dv.where(ok).rank(axis=1, ascending=False, method="first")
    return {
        "liq_rank": rank,
        "research": rank <= int(ucfg["researchSize"]),
        "tradable": rank <= int(ucfg["tradableSize"]),
        "median_dollar_volume": med_dv,
    }


def industry_map(meta: pd.DataFrame, tickers: pd.DataFrame) -> pd.Series:
    """Ticker -> 2-digit SIC industry code (string). Missing -> 'NA'."""
    if meta is None or meta.empty:
        return pd.Series(dtype=str)
    sic = meta.set_index("cik")["sic"]
    m = tickers[["cik", "ticker"]].copy()
    m["sic"] = m["cik"].map(sic)
    m["ind"] = (m["sic"] // 100).astype("Int64").astype(str).replace({"<NA>": "NA"})
    return m.set_index("ticker")["ind"]


def cost_bps(liq_rank: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """One-way trading cost by liquidity tier (bps), from config. Rough but conservative."""
    tiers = section("lab")["costTiersBps"]
    out = liq_rank * np.nan
    upper_prev = 0
    for upper, bps in tiers:
        out = out.where(~((liq_rank > upper_prev) & (liq_rank <= upper)), bps)
        upper_prev = upper
    return out.fillna(tiers[-1][1])
