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
from ..models import calibrate, effective_signature, fit as fit_model, resolve_features, spec_id
from ..util import append_jsonl, iso, log, now_et, sha256_file, write_json
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
    looks = max(1, ledger.holdout_looks() + 1)
    bar = max(float(g["holdoutIcTstatMin"]), float(norm.ppf(1 - 0.05 / (2 * looks))))
    checks = {
        "icTstat": bool(ic.get("tNW", 0) >= bar),
        "positiveNet": bool(ls.get("netAnnual", -1) > g["holdoutNetSpreadMin"]),
    }
    return {"checks": checks, "passed": all(checks.values()), "icBar": round(bar, 3), "holdoutLooks": looks}


def evaluate(idx: PanelIndex, spec: dict, era: str, data_sig: str, reason: str) -> dict | None:
    sid = spec_id(spec)
    eff = effective_signature(spec, data_sig)
    if ledger.has(sid, era, data_sig, eff):
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
        "spec": spec, "dataSig": data_sig, "effSig": eff, "metrics": m, "gates": gates(m, era, n_trials + 1), "calibration": cal,
        "blocks": len(blocks), "seconds": round(time.time() - t0, 1), "code": code_version(),
        "oosStart": oos["date"].min().date().isoformat(), "oosEnd": oos["date"].max().date().isoformat(),
    })
    ledger.save_series(sid, era, series)
    ls = m.get("longShort", {})
    L.info("%-40s %s IC=%+.4f t=%+.2f net=%+.3f DSR=%s pass=%s (%.0fs)", sid, era, m["ic"].get("mean", np.nan),
           m["ic"].get("tNW", np.nan), ls.get("netAnnual", np.nan), round(ls.get("deflated", {}).get("dsr", np.nan) or np.nan, 3),
           entry["gates"]["passed"], time.time() - t0)
    return entry


def _score(e: dict) -> tuple:
    """Ranking of learned models: after-cost Sharpe of the hysteresis long-short, then IC t-stat."""
    s = e["metrics"].get("longShort", {}).get("sharpeNet")
    t = e["metrics"].get("ic", {}).get("tNW")
    return (s if s is not None and np.isfinite(s) else -np.inf, t if t is not None and np.isfinite(t) else -np.inf)


def top_learned(data_sig: str, k: int = 3) -> list[dict]:
    """Up to k learned models worth exploring around: some real evidence of ranking skill and
    not already deep in a chain of tweaks (deep chains mostly fit noise and inflate the trial count)."""
    lab = section("lab")
    learned = [e for e in ledger.current("dev", data_sig).values() if e["spec"]["kind"] != "factor"
               and e["metrics"].get("ic", {}).get("tNW", 0) >= float(lab["minIcTstatToExplore"])
               and int(e["spec"].get("depth", 0)) < int(lab["maxNeighborDepth"])]
    return sorted(learned, key=_score, reverse=True)[:k]


def update_registry(data_sig: str) -> dict:
    lcfg = section("live")
    reg = ledger.registry()
    dev = ledger.current("dev", data_sig)
    hold = ledger.current("holdout", data_sig)
    learned = [e for e in dev.values() if e["spec"]["kind"] != "factor"]
    sharpe = _score

    gated = sorted([e for e in learned if e["gates"]["passed"] and e["id"] in hold and hold[e["id"]]["gates"]["passed"]],
                   key=sharpe, reverse=True)
    dev_pass = sorted([e for e in learned if e["gates"]["passed"]], key=sharpe, reverse=True)

    def add(e, role, note):
        m = reg["models"].setdefault(e["id"], {"spec": e["spec"], "addedAt": iso(now_et())})
        m.update({"role": role, "name": e.get("name"), "note": note, "dev": e["metrics"],
                  "devGates": e["gates"], "holdout": hold.get(e["id"], {}).get("metrics"),
                  "holdoutGates": hold.get(e["id"], {}).get("gates"),
                  "calibration": (hold.get(e["id"]) or e)["calibration"], "horizon": int(e["spec"].get("horizon", 5))})

    for mid, info in reg["models"].items():          # keep live models' lab evidence current
        if mid in dev and info.get("role") in ("champion", "challenger"):
            add(dev[mid], info["role"], info.get("note", ""))

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

    # one live slot per distinct fitted model: retire challengers that duplicate the champion or each other
    seen = {ledger.identity(dev[reg["champion"]])} if reg.get("champion") in dev else set()
    for k, v in reg["models"].items():
        if v.get("role") == "challenger" and k in dev:
            ident = ledger.identity(dev[k])
            if ident in seen:
                v["role"] = "retired"
                v["note"] = "retired: same fitted model as another live model on the current data"
                reg["history"].append({"at": iso(now_et()), "event": "challenger retired (duplicate)", "model": k})
            seen.add(ident)
    # challengers: gated first, then development passers, then the best remaining learned models (exploratory)
    challengers = [k for k, v in reg["models"].items() if v.get("role") == "challenger"]
    pool = [e for e in gated + dev_pass if e["id"] != reg.get("champion")]
    pool += sorted([e for e in learned if e not in pool and e["id"] != reg.get("champion") and e.get("reason") == "pre-registered"],
                   key=_score, reverse=True)
    for e in pool:
        if len(challengers) >= int(lcfg["maxChallengers"]):
            break
        if e["id"] in challengers or ledger.identity(e) in seen:
            continue
        seen.add(ledger.identity(e))
        note = "passed all gates" if e in gated else "passed development gates" if e in dev_pass else "exploratory: did not pass gates"
        add(e, "challenger", note)
        challengers.append(e["id"])
        reg["history"].append({"at": iso(now_et()), "event": "challenger added", "to": e["id"], "note": note})
    ledger.save_registry(reg)
    return reg


def null_controls(idx: PanelIndex, data_sig: str, n: int) -> list[dict]:
    """Run pre-registered learned models on a copy of the panel whose outcomes are shuffled among
    the stocks of each day (destroys any real cross-sectional signal, keeps everything else).
    The share that passes the development gates estimates the gates' false-pass rate. These runs
    are diagnostics, logged in lab/null_controls.jsonl, never ledger trials."""
    names = ["ens_all", "gbm_all", "ridge_all"]
    specs = [s for s in initial_specs() if s["name"] in names]
    seed = int(time.time()) % 2**31
    rng = np.random.default_rng(seed)
    p = idx.panel
    order = np.lexsort((rng.random(len(p)), p["date"].to_numpy()))     # p is date-sorted: permutes within each date
    labels = [c for c in p.columns if c.startswith(("fwd_", "target_"))]
    q = p.copy()
    q[labels] = p[labels].to_numpy()[order]
    nidx = PanelIndex(q)
    n_trials, sr_var = ledger.dev_trials()
    out = []
    for spec in specs[:n]:
        h = int(spec.get("horizon", 5))
        oos, _, _ = walk_forward(nidx, spec, "dev")
        if oos.empty:
            continue
        m = summarize(oos, h, n_trials=n_trials, sr_var_trials=sr_var)
        m.pop("_series", None)
        g = gates(m, "dev", n_trials)
        row = {"at": iso(now_et()), "spec": spec["name"], "seed": seed, "dataSig": data_sig, "icMean": m["ic"].get("mean"),
               "icT": m["ic"].get("tNW"), "netAnnual": m.get("longShort", {}).get("netAnnual"), "passed": g["passed"]}
        append_jsonl(paths().lab / "null_controls.jsonl", row)
        out.append(row)
        L.info("null control %s: IC=%+.4f t=%+.2f passed=%s", spec["name"], row["icMean"] or 0, row["icT"] or 0, row["passed"])
    return out


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
        if ledger.has(spec_id(spec), "dev", data_sig, effective_signature(spec, data_sig)):
            done_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), done_dir / f.name)

    if not over() and budget > 0:
        report["nullControls"] = null_controls(idx, data_sig, int(lab["nullControlsPerSweep"]))

    auto = 0
    while not over() and auto < int(lab["maxAutoTrialsPerSweep"]):
        nxt = None
        for best in top_learned(data_sig, k=3):           # explore around the three best models, best first
            todo = [s for s in neighbors(best["spec"], fams) if spec_runnable(s, fams)
                    and not ledger.has(spec_id(s), "dev", data_sig, effective_signature(s, data_sig))]
            if todo:
                nxt = (todo[0], best)
                break
        if nxt is None:
            break
        e = evaluate(idx, nxt[0], "dev", data_sig, f"neighbor of {nxt[1]['id']}")
        auto += 1
        if e:
            report["evaluated"].append(e["id"])

    # holdout: development passers that have never been tested there, best first, capped per week
    cap = min(int(lab["maxHoldoutTestsPerWeek"]) - ledger.holdout_tests_this_week(),
              int(lab["maxHoldoutTestsPerYear"]) - ledger.holdout_tests_this_week(days=365))
    tested = set(ledger.current("holdout", data_sig))
    dev_now = ledger.current("dev", data_sig)
    cands = [e for e in dev_now.values() if e["gates"]["passed"] and e["spec"]["kind"] != "factor" and e["id"] not in tested]
    cands.sort(key=_score, reverse=True)
    reg0 = ledger.registry()
    if reg0.get("champion") is None or reg0.get("championStatus") == "provisional":
        # the provisional champion always gets a holdout look on the current data, for the record
        cid = reg0.get("champion")
        prov = [e for e in dev_now.values() if (e["id"] == cid or (cid is None and e.get("name") == DEFAULT_PROVISIONAL))
                and e["id"] not in tested and e not in cands]
        cands = prov[:1] + cands
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
