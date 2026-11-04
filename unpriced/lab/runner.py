"""The weekly research sweep: evaluate, gate, select, refit.

Order of work inside the time budget:
  1. pre-registered initial experiments not yet evaluated on the current data
  2. queued experiments (lab/queue/*.json, written by people or the AI research session)
  3. automatic neighbors of the current best learned model
  4. holdout tests for development-era passers (capped per week)
  5. registry update (champion/challengers) and live refits

The champion is never replaced here once it has earned its place through the gates; after
that, only the live sequential test (live/promotion.py) can replace it.
"""
from __future__ import annotations

import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import paths, section
from ..features.context import Context, load_context
from ..features.library import REGISTRY, families
from ..features.panel import build_panel
from ..models import calibrate, fit as fit_model, resolve_features, spec_id
from ..util import iso, log, now_et, sha256_file, write_json
from . import ledger
from .metrics import summarize
from .specs import DEFAULT_PROVISIONAL, initial_specs, neighbors, queued_specs
from .walkforward import PanelIndex, era_bounds, training_dates, walk_forward

L = log("lab")


def code_version() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=paths().root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def available_families(idx: PanelIndex) -> list[str]:
    p = idx.panel
    out = []
    for fam, names in families().items():
        if any(n in p and p[n].notna().mean() > 0.05 for n in names):
            out.append(fam)
    return sorted(out)


def spec_runnable(spec: dict, fams: list[str]) -> bool:
    feats = resolve_features(spec)
    return any(REGISTRY[f].family in fams for f in feats)


def ic_threshold(n_trials: int) -> float:
    """IC t-stat bar that rises with the number of ideas tried (Bonferroni, two-sided 5%),
    never below the t >= 3 suggested by Harvey, Liu & Zhu (2016)."""
    base = float(section("lab")["gates"]["devIcTstatMin"])
    return max(base, float(norm.ppf(1 - 0.05 / (2 * max(1, n_trials)))))


def gates(metrics: dict, era: str, n_trials: int = 1) -> dict:
    g = section("lab")["gates"]
    ic = metrics.get("ic", {})
    ls = metrics.get("longShort", {})
    if era == "dev":
        bar = ic_threshold(n_trials)
        checks = {
            "icTstat": bool(ic.get("tNW", 0) >= bar),
            "netSharpeSignificant": bool(ls.get("deflated", {}).get("psr", 0) >= g["devNetPsrMin"]),
            "positiveNet": bool(ls.get("netAnnual", -1) > 0),
        }
        return {"checks": checks, "passed": all(checks.values()), "icBar": round(bar, 3), "trials": n_trials}
    else:
        checks = {
            "icTstat": bool(ic.get("tNW", 0) >= g["holdoutIcTstatMin"]),
            "positiveNet": bool(ls.get("netAnnual", -1) > g["holdoutNetSpreadMin"]),
        }
    return {"checks": checks, "passed": all(checks.values())}


def evaluate(idx: PanelIndex, spec: dict, era: str, data_sig: str, reason: str) -> dict | None:
    sid = spec_id(spec)
    if ledger.has(sid, era, data_sig):
        return None
    h = int(spec.get("horizon", 5))
    t0 = time.time()
    n_trials, sr_var = ledger.dev_trials()
    oos, blocks, _ = walk_forward(idx, spec, era)
    if oos.empty:
        L.warning("%s produced no out-of-sample predictions in %s", sid, era)
        return None
    m = summarize(oos, h, n_trials=n_trials + (1 if era == "dev" else 0), sr_var_trials=sr_var)
    series = m.pop("_series")
    cal = calibrate(oos, h).to_dict()
    entry = ledger.record({
        "id": sid, "name": spec.get("name"), "era": era, "status": "evaluated", "reason": reason,
        "spec": spec, "dataSig": data_sig, "metrics": m, "gates": gates(m, era, n_trials + 1), "calibration": cal,
        "blocks": len(blocks), "seconds": round(time.time() - t0, 1), "code": code_version(),
        "oosStart": oos["date"].min().date().isoformat(), "oosEnd": oos["date"].max().date().isoformat(),
    })
    ledger.save_series(sid, era, series)
    ls = m.get("longShort", {})
    L.info("%-40s %s IC=%+.4f t=%+.2f net=%+.3f DSR=%s pass=%s (%.0fs)", sid, era, m["ic"].get("mean", np.nan),
           m["ic"].get("tNW", np.nan), ls.get("netAnnual", np.nan), round(ls.get("deflated", {}).get("dsr", np.nan) or np.nan, 3),
           entry["gates"]["passed"], time.time() - t0)
    return entry


def _best_learned() -> dict | None:
    best, score = None, -np.inf
    for e in ledger.entries():
        if e["era"] != "dev" or e.get("status") != "evaluated" or e["spec"]["kind"] == "factor":
            continue
        s = e["metrics"].get("longShort", {}).get("sharpeNet")
        if s is not None and np.isfinite(s) and s > score:
            best, score = e, s
    return best


def update_registry(data_sig: str) -> dict:
    lcfg = section("live")
    reg = ledger.registry()
    dev = {}
    for e in ledger.entries():
        if e.get("status") == "evaluated" and e["era"] == "dev" and e.get("dataSig") == data_sig:
            dev[e["id"]] = e
    hold = {e["id"]: e for e in ledger.entries() if e.get("status") == "evaluated" and e["era"] == "holdout"}
    learned = [e for e in dev.values() if e["spec"]["kind"] != "factor"]

    def sharpe(e):
        v = e["metrics"].get("longShort", {}).get("sharpeNet")
        return v if v is not None and np.isfinite(v) else -np.inf

    gated = sorted([e for e in learned if e["gates"]["passed"] and e["id"] in hold and hold[e["id"]]["gates"]["passed"]],
                   key=sharpe, reverse=True)
    dev_pass = sorted([e for e in learned if e["gates"]["passed"]], key=sharpe, reverse=True)

    def add(e, role, note):
        m = reg["models"].setdefault(e["id"], {"spec": e["spec"], "addedAt": iso(now_et())})
        m.update({"role": role, "name": e.get("name"), "note": note, "dev": e["metrics"],
                  "devGates": e["gates"], "holdout": hold.get(e["id"], {}).get("metrics"),
                  "holdoutGates": hold.get(e["id"], {}).get("gates"),
                  "calibration": (hold.get(e["id"]) or e)["calibration"], "horizon": int(e["spec"].get("horizon", 5))})

    current = reg.get("champion")
    status = reg.get("championStatus")
    if gated and (current is None or status == "provisional"):
        if current and current in reg["models"] and current != gated[0]["id"]:
            reg["models"][current]["role"] = "challenger"
            reg["history"].append({"at": iso(now_et()), "event": "provisional champion replaced by gated model",
                                   "from": current, "to": gated[0]["id"]})
        add(gated[0], "champion", "passed development and holdout gates")
        reg["champion"], reg["championStatus"] = gated[0]["id"], "gated"
    elif current is None:
        prov = next((e for e in learned if e.get("name") == DEFAULT_PROVISIONAL), None) or (dev_pass[0] if dev_pass else None) \
            or (max(learned, key=lambda e: e["metrics"]["ic"].get("tNW", -9)) if learned else None)
        if prov:
            add(prov, "champion", "provisional: pre-registered default; no model has passed all gates yet")
            reg["champion"], reg["championStatus"] = prov["id"], "provisional"
            reg["history"].append({"at": iso(now_et()), "event": "provisional champion set", "to": prov["id"]})

    # challengers: gated first, then development passers, then the best remaining learned models (exploratory)
    challengers = [k for k, v in reg["models"].items() if v.get("role") == "challenger"]
    pool = [e for e in gated + dev_pass if e["id"] != reg.get("champion")]
    pool += sorted([e for e in learned if e not in pool and e["id"] != reg.get("champion")],
                   key=lambda e: e["metrics"]["ic"].get("tNW", -9), reverse=True)
    for e in pool:
        if len(challengers) >= int(lcfg["maxChallengers"]):
            break
        if e["id"] in challengers:
            continue
        note = "passed all gates" if e in gated else "passed development gates" if e in dev_pass else "exploratory: did not pass gates"
        add(e, "challenger", note)
        challengers.append(e["id"])
        reg["history"].append({"at": iso(now_et()), "event": "challenger added", "to": e["id"], "note": note})
    ledger.save_registry(reg)
    return reg


def refit_live(idx: PanelIndex, reg: dict | None = None) -> dict:
    """Fit every live model on all data whose labels are known; save artifacts to the store."""
    reg = reg or ledger.registry()
    lab_cfg = section("lab")
    for mid in ledger.live_models(reg):
        m = reg["models"][mid]
        h = int(m["spec"].get("horizon", 5))
        before = pd.Timestamp(idx.dates[-1]) + pd.Timedelta(days=1)
        tdates = training_dates(idx, before, h, int(lab_cfg["trainEveryNthDate"]))
        fitted = fit_model(m["spec"], idx.rows(tdates))
        path = paths().models / f"{mid}.pkl"
        fitted.save(path)
        m["fit"] = {"trainedAt": iso(now_et()), "trainedThrough": pd.Timestamp(tdates[-1]).date().isoformat(),
                    "rows": int(fitted.info.get("rows", 0)), "sha256": sha256_file(path), "features": fitted.features}
        L.info("refit %s through %s (%s rows)", mid, m["fit"]["trainedThrough"], m["fit"]["rows"])
    ledger.save_registry(reg)
    return reg


def run_sweep(budget_minutes: float | None = None, ctx: Context | None = None, max_new: int | None = None) -> dict:
    lab = section("lab")
    budget = 60 * float(budget_minutes if budget_minutes is not None else lab["weeklyBudgetMinutes"])
    t0 = time.time()
    ctx = ctx or load_context()
    panel = build_panel(ctx, list(REGISTRY))
    idx = PanelIndex(panel)
    fams = available_families(idx)
    data_sig = ",".join(fams)
    L.info("panel %s rows, %d dates; available families: %s", len(panel), len(idx.dates), data_sig)
    report = {"startedAt": iso(now_et()), "dataSig": data_sig, "evaluated": [], "holdout": [], "skipped": []}

    def over() -> bool:
        return time.time() - t0 > budget or (max_new is not None and len(report["evaluated"]) >= max_new)

    plan: list[tuple[dict, str]] = [(s, "pre-registered") for s in initial_specs()]
    queue = queued_specs()
    plan += [(s, f"queued:{f.name}") for f, s in queue]
    for spec, reason in plan:
        if over():
            break
        if not spec_runnable(spec, fams):
            report["skipped"].append({"name": spec.get("name"), "reason": "required data unavailable"})
            continue
        e = evaluate(idx, spec, "dev", data_sig, reason)
        if e:
            report["evaluated"].append(e["id"])
    done_dir = paths().lab / "queue" / "done"
    for f, spec in queue:
        if ledger.has(spec_id(spec), "dev", data_sig):
            done_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), done_dir / f.name)

    while not over():
        best = _best_learned()
        if not best:
            break
        todo = [s for s in neighbors(best["spec"]) if spec_runnable(s, fams) and not ledger.has(spec_id(s), "dev", data_sig)]
        if not todo:
            break
        e = evaluate(idx, todo[0], "dev", data_sig, f"neighbor of {best['id']}")
        if e:
            report["evaluated"].append(e["id"])

    # holdout: development passers that have never been tested there, best first, capped per week
    cap = int(lab["maxHoldoutTestsPerWeek"]) - ledger.holdout_tests_this_week()
    tested = {e["id"] for e in ledger.entries() if e["era"] == "holdout"}
    cands = [e for e in ledger.entries() if e["era"] == "dev" and e.get("status") == "evaluated" and e["dataSig"] == data_sig
             and e["gates"]["passed"] and e["spec"]["kind"] != "factor" and e["id"] not in tested]
    cands.sort(key=lambda e: e["metrics"].get("longShort", {}).get("sharpeNet", -9), reverse=True)
    reg0 = ledger.registry()
    if reg0.get("champion") is None:   # the provisional default always gets one holdout look for the record
        prov = [e for e in ledger.entries() if e.get("name") == DEFAULT_PROVISIONAL and e["era"] == "dev"
                and e.get("dataSig") == data_sig and e["id"] not in tested and e not in cands]
        cands = prov[-1:] + cands
    for e in cands[: max(0, cap)]:
        h = evaluate(idx, e["spec"], "holdout", data_sig, "holdout test")
        if h:
            report["holdout"].append(h["id"])

    reg = update_registry(data_sig)
    refit_live(idx, reg)
    report["finishedAt"] = iso(now_et())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["champion"] = reg.get("champion")
    write_json(paths().lab / "last_sweep.json", report)
    return report
