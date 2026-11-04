"""Data refresh jobs used by the bootstrap, daily and weekly workflows."""
from __future__ import annotations

import os

import pandas as pd

from .. import store
from ..config import section
from ..util import iso, log, now_et
from . import prices as prices_mod

L = log("pipeline")


def sec_enabled() -> bool:
    return bool(section("sec")["enabled"]) and bool(os.environ.get(section("sec")["contactEnvVar"], "").strip())


def refresh_candidates() -> pd.DataFrame:
    """Merge today's symbol directory into the candidate table; tickers that disappear stay
    (their history is kept, which slowly reduces survivorship bias going forward)."""
    from .symbols import fetch_candidates
    today = now_et().date().isoformat()
    fresh = fetch_candidates()
    old = store.read_or_empty("candidates")
    if old.empty:
        fresh["firstSeen"] = today
        fresh["lastSeen"] = today
        merged = fresh
    else:
        old = old.set_index("ticker")
        f = fresh.set_index("ticker")
        merged = old.combine_first(f)
        merged.loc[f.index, ["name", "exchange"]] = f[["name", "exchange"]]
        merged.loc[f.index, "lastSeen"] = today
        merged["firstSeen"] = merged["firstSeen"].fillna(today)
        merged = merged.reset_index()
    store.write("candidates", merged)
    L.info("candidates: %d total, %d listed today", len(merged), int((merged["lastSeen"] == today).sum()))
    return merged


def active_tickers(recent_days: int = 10) -> list[str]:
    c = store.read_or_empty("candidates")
    if c.empty:
        return []
    cutoff = (now_et() - pd.Timedelta(days=recent_days)).date().isoformat()
    return sorted(t for t in c.loc[c["lastSeen"] >= cutoff, "ticker"].astype(str) if t and t != "nan")


def recent_universe(sessions_back: int = 40) -> list[str]:
    """Tickers that were in the research universe recently (cheap daily refresh set)."""
    from ..features.context import load_context  # local import: heavy
    p = store.read("prices", columns=["date", "ticker", "close", "adj_close", "volume", "open", "high", "low", "split"])
    last = p["date"].max()
    p = p[p["date"] >= last - pd.Timedelta(days=500)]
    ctx = load_context(prices=p)
    member = ctx.member["research"].iloc[-sessions_back:]
    return sorted(member.columns[member.any(axis=0)])


def ever_universe() -> list[str]:
    """Tickers that were in the research universe on any date in the stored history."""
    from ..features.context import load_context  # local import: heavy
    ctx = load_context()
    return sorted(ctx.member["research"].columns)


def update_prices_job(mode: str, provider=None) -> dict:
    ctx_syms = section("data")["contextSymbols"]
    if mode == "daily" and store.exists("prices"):
        from ..live import portfolio
        held = list(portfolio.state().get("positions", {}).keys())
        tickers = sorted(set(recent_universe()) | set(ctx_syms) | set(held))
    else:
        tickers = sorted(set(active_tickers()) | set(ctx_syms))
    return prices_mod.update_prices(tickers, provider=provider)


def update_sec_job(mode: str, session: pd.Timestamp | None = None, tickers: list[str] | None = None) -> dict:
    """Filings and fundamentals from EDGAR. Skipped (not failed) when no contact email is set."""
    if not sec_enabled():
        return {"enabled": False, "note": f"set {section('sec')['contactEnvVar']} to enable"}
    from .sec import SecClient, facts_table
    from .fundamentals import build_fundamentals
    sec = SecClient()
    tick = sec.tickers()
    store.write("sec_tickers", tick)
    if tickers is None:
        tickers = recent_universe(sessions_back=252) if mode == "daily" else ever_universe()
    universe = set(tickers)
    m = tick[tick["ticker"].isin(universe)].drop_duplicates("cik")
    ciks = list(m["cik"].astype(int))
    filings = store.read_or_empty("sec_filings")
    meta = store.read_or_empty("sec_meta")
    facts = store.read_or_empty("sec_facts")
    report = {"enabled": True, "mode": mode}

    if mode == "daily" and session is not None and len(filings):
        idx = sec.daily_index(pd.Timestamp(session))
        hit = idx[idx["cik"].isin(set(ciks)) & idx["form"].isin(["8-K", "10-Q", "10-K", "8-K/A", "10-Q/A", "10-K/A"])]
        refresh = sorted(set(hit["cik"]))
        fact_refresh = sorted(set(hit.loc[hit["form"].isin(["10-Q", "10-K", "10-Q/A", "10-K/A"]), "cik"]))
        include_older = False
    else:
        refresh = ciks
        fact_refresh = ciks
        include_older = mode != "weekly"     # bootstrap pulls full history; weekly only recent + new names
    have = set(filings["cik"]) if len(filings) else set()
    report["submissions"] = len(refresh)
    rows, metas = [], []
    for i, cik in enumerate(refresh):
        older = include_older or cik not in have
        mt, fl = sec.submissions(cik, include_older=older)
        if mt:
            metas.append(mt)
        if len(fl):
            rows.append(fl)
        if i % 200 == 0:
            L.info("SEC submissions %d/%d", i, len(refresh))
    if rows:
        new = pd.concat(rows, ignore_index=True)
        filings = pd.concat([filings, new], ignore_index=True) if len(filings) else new
        filings = filings.drop_duplicates("accession", keep="last").reset_index(drop=True)
        store.write("sec_filings", filings)
    if metas:
        mdf = pd.DataFrame(metas)[["cik", "name", "sic", "sicDescription", "entityType"]]
        meta = pd.concat([meta, mdf], ignore_index=True) if len(meta) else mdf
        store.write("sec_meta", meta.drop_duplicates("cik", keep="last").reset_index(drop=True))
    report["facts"] = len(fact_refresh)
    fparts = []
    for i, cik in enumerate(fact_refresh):
        js = sec.companyfacts(cik)
        if js:
            fparts.append(facts_table(js, cik))
        if i % 200 == 0:
            L.info("SEC company facts %d/%d", i, len(fact_refresh))
    if fparts:
        newf = pd.concat([f for f in fparts if len(f)], ignore_index=True) if any(len(f) for f in fparts) else pd.DataFrame()
        if len(newf):
            keep = facts[~facts["cik"].isin(set(newf["cik"]))] if len(facts) else facts
            facts = pd.concat([keep, newf], ignore_index=True) if len(keep) else newf
            store.write("sec_facts", facts)
            cik_t = tick.drop_duplicates("cik").set_index("cik")["ticker"].to_dict()
            fund = build_fundamentals(facts, cik_t)
            if len(fund):
                store.write("fundamentals", fund)
                report["fundamentalRows"] = int(len(fund))
    report["at"] = iso(now_et())
    return report
