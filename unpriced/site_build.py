"""Builds the JSON files the static Unpriced site reads (site/data/*.json)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import store
from .config import config, paths, section
from .features.library import REGISTRY, families
from .lab import ledger
from .live import scoring, snapshot
from .util import iso, now_et, read_json, read_jsonl, write_json


def _round(x, n=5):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def _downsample(dates: list, values: list, max_points: int = 400) -> dict:
    if len(dates) <= max_points:
        return {"dates": dates, "values": values}
    step = int(np.ceil(len(dates) / max_points))
    return {"dates": dates[::step] + [dates[-1]], "values": values[::step] + [values[-1]]}


def _cum(values: list) -> list:
    out, s = [], 0.0
    for v in values:
        s += v if v is not None else 0.0
        out.append(round(s, 5))
    return out


def _cumprod(values: list) -> list:
    out, s = [], 1.0
    for v in values:
        s *= 1 + (v or 0.0)
        out.append(round(s, 5))
    return out


def _null_summary() -> dict:
    rows = read_jsonl(paths().lab / "null_controls.jsonl")
    return {"runs": len(rows), "passed": sum(1 for r in rows if r.get("passed")),
            "recent": rows[-6:]}


def lab_json() -> dict:
    ent = [e for e in ledger.entries() if e.get("status") == "evaluated"]
    hold = {e["id"]: e for e in ent if e["era"] == "holdout"}
    reg = ledger.registry()
    rows = []
    for e in ent:
        if e["era"] != "dev":
            continue
        m, ls, lo = e["metrics"], e["metrics"].get("longShort", {}), e["metrics"].get("longOnly", {})
        h = hold.get(e["id"])
        rows.append({
            "id": e["id"], "name": e.get("name"), "kind": e["spec"]["kind"], "horizon": e["spec"].get("horizon", 5),
            "reason": e.get("reason"), "at": e["at"][:10], "dataSig": e.get("dataSig"),
            "icMean": _round(m["ic"].get("mean")), "icT": _round(m["ic"].get("tNW"), 2), "icHit": _round(m["ic"].get("hitRate"), 3),
            "yearsPositive": _round(m["ic"].get("yearsPositive"), 2),
            "lsNet": _round(ls.get("netAnnual"), 4), "lsGross": _round(ls.get("grossAnnual"), 4), "lsSharpe": _round(ls.get("sharpeNet"), 2),
            "dsr": _round(ls.get("deflated", {}).get("dsr"), 3), "psr": _round(ls.get("deflated", {}).get("psr"), 3),
            "turnover": _round(ls.get("turnover"), 3), "loExcess": _round(lo.get("excessAnnual"), 4),
            "passedDev": e["gates"]["passed"], "icBar": e["gates"].get("icBar"),
            "holdout": None if not h else {"icMean": _round(h["metrics"]["ic"].get("mean")), "icT": _round(h["metrics"]["ic"].get("tNW"), 2),
                                            "lsNet": _round(h["metrics"].get("longShort", {}).get("netAnnual"), 4),
                                            "passed": h["gates"]["passed"]},
            "role": reg["models"].get(e["id"], {}).get("role"),
            "byYear": m["ic"].get("byYear", {}),
        })
    rows.sort(key=lambda r: (r["lsSharpe"] is None, -(r["lsSharpe"] or 0)))
    charts = {}
    for mid in ledger.live_models(reg)[:6]:
        for era in ("dev", "holdout"):
            s = ledger.load_series(mid, era)
            if not s:
                continue
            c = {}
            if "ic" in s:
                c["cumIC"] = _downsample(s["ic"]["dates"], _cum(s["ic"]["values"]))
            if "longShort" in s:
                c["lsNet"] = _downsample(s["longShort"]["dates"], _cumprod(s["longShort"]["net"]))
            if "longOnly" in s:
                c["loExcess"] = _downsample(s["longOnly"]["dates"], _cumprod(s["longOnly"]["excess"]))
            charts.setdefault(mid, {})[era] = c
    n_trials, _ = ledger.dev_trials()
    fam = families()
    from .lab.runner import ic_threshold
    return {
        "nullControls": _null_summary(),
        "updatedAt": iso(now_et()), "trials": n_trials, "icBarNow": round(ic_threshold(n_trials + 1), 2), "experiments": rows, "charts": charts,
        "eras": {"devEnd": section("lab")["devEnd"], "holdoutStart": section("lab")["holdoutStart"], "trainStart": section("lab")["trainStart"]},
        "gates": section("lab")["gates"], "lastSweep": read_json(paths().lab / "last_sweep.json", {}),
        "features": [{"name": f.name, "family": f.family, "doc": f.doc, "ref": f.ref, "sign": f.sign} for f in REGISTRY.values()],
        "families": fam,
    }


_SUFFIX = re.compile(r"(?i)\s*[-,]?\s*(class [a-z] )?(common stock|ordinary shares?|american depositary shares?.*|"
                     r"depositary shares?.*|shares of beneficial interest|capital stock).*$")


def company_names() -> dict:
    c = store.read_or_empty("candidates")
    if c.empty:
        return {}
    names = c.set_index("ticker")["name"].astype(str)
    return {t: _SUFFIX.sub("", n.split(" - ")[0]).strip().rstrip(",") for t, n in names.items()}


def forecasts_json(reg: dict) -> dict:
    links = snapshot.chain()
    if not links:
        return {"status": "pre-live"}
    snap = snapshot.load(pd.Timestamp(links[-1]["session"]))
    champ = snap.get("champion")
    names = company_names()
    models = {}
    for mid, m in snap["models"].items():
        df = pd.DataFrame({"ticker": snap["universe"], "score": m["score"], "tradable": snap["tradable"]})
        df["name"] = df["ticker"].map(names).fillna("")
        if m.get("probBeatMedian"):
            df["prob"] = m["probBeatMedian"]
            df["excess"] = m["expectedExcess"]
        df = df.dropna(subset=["score"]).sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["pct"] = 1 - df.index / max(1, len(df) - 1)
        models[mid] = {"name": m.get("name"), "role": m.get("role"), "horizon": m.get("horizon"),
                       "trainedThrough": m.get("trainedThrough"), "n": int(len(df)), "llmCovered": m.get("llmCovered"),
                       "rows": df.round(5).to_dict(orient="records") if mid == champ else df.head(25).round(5).to_dict(orient="records")}
        if mid != champ:
            models[mid]["bottom"] = df.tail(25).round(5).to_dict(orient="records")
    return {"status": "live", "session": snap["session"], "decidedAt": snap["decidedAt"], "entrySession": snap["entrySession"],
            "champion": champ, "championStatus": snap.get("championStatus"), "models": models,
            "file": links[-1]["file"], "sha256": links[-1]["sha256"]}


def live_json(reg: dict) -> dict:
    df = scoring.scores()
    seq = read_json(paths().live / "sequential.json", {})
    out = {"updatedAt": iso(now_et()), "chain": snapshot.verify_chain(), "snapshots": len(snapshot.chain()),
           "models": {}, "sequential": seq, "registryHistory": reg.get("history", [])[-40:]}
    if df.empty:
        return out
    for mid, g in df.groupby("model"):
        per_h = {}
        for h, gh in g.groupby("horizon"):
            gh = gh.sort_values("session")
            per_h[int(h)] = {"n": int(len(gh)), "icMean": _round(gh["ic"].mean()), "spreadMean": _round(gh["spread"].mean()),
                             "topExcessMean": _round(gh["topExcess"].mean()), "hit": _round((gh["ic"] > 0).mean(), 3),
                             "series": {"dates": list(gh["session"]), "ic": [round(v, 4) for v in gh["ic"]],
                                        "cumIC": _cum(list(gh["ic"]))}}
        info = reg["models"].get(mid, {})
        out["models"][mid] = {"name": info.get("name") or mid, "role": info.get("role", "overlay"), "horizons": per_h,
                              "firstSession": str(g["session"].min())}
    return out


def portfolio_json() -> dict:
    st = read_json(paths().live / "portfolio.json", None)
    trades = read_jsonl(paths().live / "trades.jsonl")
    if not st:
        return {"status": "pre-live", "startingCash": section("portfolio")["startingCash"], "rules": section("portfolio")}
    eq = st.get("equity", st["startingCash"])
    curve = st.get("curve", [])
    spy = next((c["spy"] for c in reversed(curve) if c.get("spy")), None)
    names = company_names()
    pos = [{"ticker": t, "name": names.get(t, ""), "value": _round(p.get("lastValue", p["dollars"]), 2), "weight": _round(p.get("lastValue", p["dollars"]) / eq, 4),
            "since": p["entrySession"], "return": _round(p.get("lastValue", p["dollars"]) / p["dollars"] - 1, 4)} for t, p in st["positions"].items()]
    pos.sort(key=lambda r: -(r["value"] or 0))
    return {"status": "live" if st.get("startSession") else "awaiting-first-fill", "equity": _round(eq, 2), "cash": _round(st["cash"], 2),
            "startingCash": st["startingCash"], "startSession": st.get("startSession"), "spyValue": _round(spy, 2),
            "return": _round(eq / st["startingCash"] - 1, 4), "spyReturn": None if spy is None else _round(spy / st["startingCash"] - 1, 4),
            "positions": pos, "pending": st.get("pending"), "curve": curve[-600:], "trades": trades[-60:][::-1],
            "tradeCount": len(trades), "rebalances": st.get("rebalances", 0), "rules": section("portfolio")}


def filings_json() -> dict:
    from .llm import reader
    rec = reader.records()
    out = {"enabled": reader.enabled(), "model": section("llm")["model"], "count": int(len(rec)), "recent": [], "evaluation": None}
    if rec.empty:
        return out
    rec = rec.sort_values("readAt", ascending=False)
    cols = ["ticker", "form", "items", "filingDate", "reactionSession", "event_type", "direction", "surprise", "guidance",
            "materiality", "confidence", "summary"]
    out["recent"] = rec[[c for c in cols if c in rec]].head(150).to_dict(orient="records")
    try:  # event-level check: realized 5-session excess return after the reaction session, by direction
        p = store.read("prices", columns=["date", "ticker", "open", "close", "adj_close"])
        p = p[p["date"] >= pd.to_datetime(rec["reactionSession"]).min() - pd.Timedelta(days=5)]
        adj_open = (p.pivot(index="date", columns="ticker", values="open") * (p.pivot(index="date", columns="ticker", values="adj_close")
                    / p.pivot(index="date", columns="ticker", values="close")))
        bench = section("data")["benchmark"]
        rows = []
        for r in rec.itertuples():
            s = pd.Timestamp(r.reactionSession)
            if s not in adj_open.index or r.ticker not in adj_open:
                continue
            i = adj_open.index.get_loc(s)
            if i + 6 >= len(adj_open):
                continue
            ret = adj_open[r.ticker].iloc[i + 6] / adj_open[r.ticker].iloc[i + 1] - 1
            bret = adj_open[bench].iloc[i + 6] / adj_open[bench].iloc[i + 1] - 1
            rows.append({"direction": int(r.direction), "excess": float(ret - bret)})
        if rows:
            ev = pd.DataFrame(rows).groupby("direction")["excess"].agg(["mean", "count"]).reset_index()
            out["evaluation"] = {"note": "Excess return vs SPY from the open after the reaction session, over 5 sessions.",
                                 "byDirection": ev.round(5).to_dict(orient="records")}
    except Exception:
        pass
    return out


def status_json(reg: dict) -> dict:
    st = read_json(paths().live / "status.json", {})
    champ = reg.get("champion")
    info = reg["models"].get(champ, {}) if champ else {}
    from .data.pipeline import sec_enabled
    from .llm import reader
    m = store.meta()
    return {
        "project": "Unpriced", "updatedAt": iso(now_et()), "phase": "live" if snapshot.chain() else ("lab" if champ else "bootstrapping"),
        "lastSession": st.get("lastSession"), "entrySession": st.get("entrySession"), "lastRunAt": st.get("lastRunAt"),
        "champion": {"id": champ, "name": info.get("name"), "status": reg.get("championStatus"), "note": info.get("note"),
                     "horizon": info.get("horizon")},
        "challengers": [{"id": k, "name": v.get("name"), "note": v.get("note")} for k, v in reg["models"].items() if v.get("role") == "challenger"],
        "sources": {"prices": section("data")["priceProvider"], "sec": sec_enabled(), "llm": reader.enabled(),
                    "survivorshipBias": True},
        "trials": ledger.dev_trials()[0], "snapshots": len(snapshot.chain()),
        "schedule": config().get("schedule", {}), "storeUpdated": m.get("updatedAt"),
    }


def export_json() -> dict | None:
    """UnpricedHub export (contract unpriced-export/1) built only from the latest frozen snapshot."""
    links = snapshot.chain()
    if not links:
        return None
    link = links[-1]
    snap = snapshot.load(pd.Timestamp(link["session"]))
    from .calendar import sessions as cal
    entry = pd.Timestamp(snap["entrySession"])
    future = cal(entry, entry + pd.Timedelta(days=20))
    h = int(section("live")["primaryHorizon"])
    exit_session = future[h] if len(future) > h else None
    models = [{"id": mid, "role": "champion" if mid == snap.get("champion") else "challenger",
               "artifactSha256": m.get("fitSha256"), "trainedThrough": m.get("trainedThrough")}
              for mid, m in snap["models"].items() if int(m.get("horizon", h)) == h]
    if exit_session is None or not models:
        return None
    from .lab.runner import code_version
    task = "stock-rank-5d-open-open"
    return {
        "schema": "unpriced-export/1",
        "lab": {"id": "stock-lab", "name": "Unpriced Stock Lab", "repo": config()["repo"],
                "experiment": f"unpricedx-{snap.get('championStatus') or 'live'}", "codeCommit": snap.get("code") or code_version()},
        "forecastId": f"stock-lab/{snap['session']}/{task}",
        "decisionSession": snap["session"],
        "informationCutoff": pd.Timestamp(snap["session"]).tz_localize("America/New_York").replace(hour=17, minute=30).isoformat(),
        "issuedAt": snap["decidedAt"],
        "entry": {"session": snap["entrySession"], "price": "open"},
        "exit": {"session": exit_session.date().isoformat(), "price": "open"},
        "task": {"id": task, "forecastType": "rank-score", "event": None,
                 "description": "Rank of each stock's total return from the entry session's open to the open five sessions later; higher score means expected to do better than the others.",
                 "target": {"quantity": "total-return", "entryPrice": "open", "exitPrice": "open", "horizonSessions": h, "relativeTo": "none"}},
        "universe": {"definition": "Up to 1,000 most-traded US common stocks priced at $5 or more, chosen each day from that day's data.",
                     "size": len(snap["universe"])},
        "models": models,
        "forecasts": {"assets": snap["universe"], "values": {m["id"]: snap["models"][m["id"]]["score"] for m in models}},
        "abstained": [],
        "provenance": {"snapshotPath": link["file"], "snapshotSha256": link["sha256"], "chainPrevHash": link.get("prevHash"),
                       "inputFingerprint": snap.get("dataFingerprint")},
        "limitations": ["Historical research uses free data that omits delisted companies (survivorship bias); live forecasts are unaffected.",
                        "Information cutoff is set to 17:30 ET of the decision session: SEC filings accepted before then may be used."],
    }


def build() -> None:
    reg = ledger.registry()
    out = paths().site_data
    exp = export_json()
    if exp:
        write_json(out / "export" / "latest.json", exp, indent=None)
    write_json(out / "status.json", status_json(reg), indent=None)
    write_json(out / "lab.json", lab_json(), indent=None)
    write_json(out / "forecasts.json", forecasts_json(reg), indent=None)
    write_json(out / "live.json", live_json(reg), indent=None)
    write_json(out / "portfolio.json", portfolio_json(), indent=None)
    write_json(out / "filings.json", filings_json(), indent=None)
