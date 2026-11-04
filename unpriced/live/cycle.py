"""The daily cycle (runs before the US open, processing the previous session).

    1. find the last completed session t; stop if its snapshot already exists (idempotent)
    2. refresh prices through t; fail closed if t's data is missing or thin
    3. refresh SEC filings/fundamentals (if configured) and run the AI filing reader (if configured)
    4. fill yesterday's paper orders at t's open, mark the portfolio at t's close
    5. score matured forecasts and update the live confidence sequences / promotions
    6. predict with every live model, freeze the snapshot into the hash chain
    7. decide the next paper orders (executed at the first open after the commit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import store
from ..calendar import last_completed_session
from ..config import paths, section
from ..data import pipeline
from ..features.context import load_context
from ..features.panel import build_panel
from ..lab import ledger
from ..lab.runner import code_version
from ..models import Calibration, Fitted
from ..util import iso, log, now_et, read_json, write_json
from . import portfolio, promotion, scoring, snapshot

L = log("cycle")


class StaleData(RuntimeError):
    pass


def _check_coverage(prices: pd.DataFrame, t: pd.Timestamp, expected: list[str]) -> float:
    bench = section("data")["benchmark"]
    day = prices[prices["date"] == t]
    if bench not in set(day["ticker"]):
        raise StaleData(f"{bench} has no bar for {t.date()} yet")
    if not expected:
        return 1.0
    cov = len(set(day["ticker"]) & set(expected)) / len(expected)
    if cov < float(section("data")["minCoverageForSession"]):
        raise StaleData(f"only {cov:.0%} of the universe has a bar for {t.date()}")
    return cov


def predict_models(ctx, t: pd.Timestamp, reg: dict) -> tuple[pd.DataFrame, dict]:
    live = ledger.live_models(reg)
    feats = sorted({f for mid in live for f in reg["models"][mid].get("fit", {}).get("features", [])})
    if not feats:
        raise RuntimeError("live models have no fitted artifacts; run the weekly sweep first")
    panel = build_panel(ctx, feats, start=t, end=t, labels=False)
    if panel.empty:
        raise StaleData(f"no universe rows for {t.date()}")
    out = {}
    for mid in live:
        m = reg["models"][mid]
        path = paths().models / f"{mid}.pkl"
        if not path.exists():
            L.warning("artifact missing for %s; skipped", mid)
            continue
        fitted = Fitted.load(path)
        score = fitted.predict(panel)
        pct = pd.Series(score).rank(pct=True).to_numpy()
        cal = Calibration.from_dict(m["calibration"]) if m.get("calibration") else None
        out[mid] = {
            "role": m["role"], "name": m.get("name"), "horizon": int(m.get("horizon", 5)),
            "fitSha256": m["fit"]["sha256"], "trainedThrough": m["fit"]["trainedThrough"],
            "score": [None if not np.isfinite(v) else round(float(v), 6) for v in score],
            "probBeatMedian": None if cal is None else [round(float(v), 4) for v in cal.prob(pct)],
            "expectedExcess": None if cal is None else [round(float(v), 5) for v in cal.excess(pct)],
        }
    return panel, out


def add_llm_overlay(models: dict, panel: pd.DataFrame, t: pd.Timestamp, dates: pd.DatetimeIndex, reg: dict) -> None:
    from ..llm import reader
    if not reader.enabled():
        return
    champ = reg.get("champion")
    if champ not in models:
        return
    sig = reader.signal(t, dates).reindex(panel["ticker"]).fillna(0.0).to_numpy()
    base = pd.Series(models[champ]["score"], dtype=float)
    base_pct = base.rank(pct=True).fillna(0.5).to_numpy() - 0.5
    nz = sig != 0
    llm_pct = np.zeros_like(sig)
    if nz.sum() >= 3:
        llm_pct[nz] = pd.Series(sig[nz]).rank(pct=True).to_numpy() - 0.5
    w = float(section("llm")["overlayWeight"])
    score = base_pct + w * llm_pct
    oid = f"llm_overlay-on-{champ}"
    models[oid] = {"role": "challenger", "name": "champion + AI filing reader", "horizon": models[champ]["horizon"],
                   "fitSha256": models[champ]["fitSha256"], "trainedThrough": models[champ]["trainedThrough"],
                   "score": [round(float(v), 6) for v in score], "llmCovered": int(nz.sum()),
                   "probBeatMedian": None, "expectedExcess": None}


def run_daily(force: bool = False, skip_data: bool = False, provider=None) -> dict:
    now = now_et()
    t = last_completed_session(now, settle_minutes=int(section("live").get("settleMinutes", 90)))
    status_path = paths().live / "status.json"
    status = read_json(status_path, {})
    if snapshot.exists(t) and not force:
        L.info("snapshot for %s already frozen; nothing to do", t.date())
        return {"status": "noop", "session": t.date().isoformat()}
    reg = ledger.registry()
    if not reg.get("champion"):
        L.warning("no champion yet: run the research sweep first")
        return {"status": "awaiting-lab", "session": t.date().isoformat()}

    report = {"session": t.date().isoformat(), "startedAt": iso(now)}
    if not skip_data:
        report["prices"] = pipeline.update_prices_job("daily", provider=provider)
        try:
            report["sec"] = pipeline.update_sec_job("daily", session=t)
        except Exception as exc:          # SEC trouble must not stop forecasts
            L.warning("SEC update failed: %s", exc)
            report["sec"] = {"error": str(exc)}
    prices = store.read("prices")
    prices = prices[prices["date"] <= t]
    ctx = load_context(prices=prices[prices["date"] >= t - pd.Timedelta(days=1100)])
    if t not in ctx.dates:
        raise StaleData(f"no price data for {t.date()} yet")
    expected = list(ctx.member["research"].columns[ctx.member["research"].iloc[-2].fillna(False).astype(bool)])
    report["coverage"] = _check_coverage(prices, t, expected)

    if not skip_data:
        try:
            from ..llm import reader
            report["llm"] = reader.run(t, list(ctx.tickers), ctx.member["liq_rank"].iloc[-1])
        except Exception as exc:
            L.warning("AI filing reader failed: %s", exc)
            report["llm"] = {"error": str(exc)}

    # --- paper portfolio: fill orders decided before this session's open, mark at close
    st = portfolio.state()
    adj_open = ctx.mats["adj_open"]
    trades = portfolio.execute(st, adj_open.loc[t], t, adj_open) if t in adj_open.index else []
    if st.get("lastRebalanceSession") and not trades and st.get("sessionsSinceRebalance") is not None:
        st["sessionsSinceRebalance"] = int(st["sessionsSinceRebalance"]) + 1
    portfolio.mark(st, t, ctx.mats["adj_close"].loc[t], adj_open)
    report["trades"] = len(trades)

    # --- score matured forecasts, update live statistics and promotions
    report["newScores"] = scoring.update(adj_open)
    report["promotion"] = promotion.review(reg)
    reg = ledger.registry()

    # --- forecasts
    panel, models = predict_models(ctx, t, reg)
    add_llm_overlay(models, panel, t, ctx.dates, reg)
    decided = now_et()
    entry = snapshot.entry_session(decided, t)
    snap = {
        "schema": 2, "project": "Unpriced", "session": t.date().isoformat(), "decidedAt": iso(decided),
        "entrySession": entry.date().isoformat(), "dataThrough": t.date().isoformat(), "code": code_version(),
        "champion": reg["champion"], "championStatus": reg.get("championStatus"),
        "universe": list(panel["ticker"]), "tradable": [bool(x) for x in panel["tradable"]],
        "models": models,
        "note": "Scores rank stocks within this universe for the stated horizon, entered at the entry session's open.",
    }
    link = snapshot.write(snap)
    report["snapshot"] = link

    # --- next paper orders
    if st.get("pending") is None:
        order = portfolio.decide(st, snap, reg["champion"] if reg["champion"] in models else next(iter(models)),
                                 ctx.member["liq_rank"].iloc[-1])
        if order:
            st["pending"] = order
    portfolio.save(st)
    status.update({"lastSession": t.date().isoformat(), "lastRunAt": iso(now_et()), "lastSnapshot": link["file"],
                   "entrySession": entry.date().isoformat(), "champion": reg["champion"],
                   "championStatus": reg.get("championStatus"), "coverage": report["coverage"],
                   "secEnabled": pipeline.sec_enabled(), "llmEnabled": bool(report.get("llm", {}).get("enabled")),
                   "dataLastDate": t.date().isoformat(), "lastError": None})
    write_json(status_path, status)
    report["status"] = "published"
    write_json(paths().live / "last_cycle.json", report)
    return report
