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
                  "calibration": (hold.get(e["id"]) or e).get("calibration"), "horizon": int(e["spec"].get("horizon", 5))})

    for mid, info in reg["models"].items():          # keep live models' lab evidence current
        if mid in dev and info.get("role") in ("champion", "challenger", "benchmark"):
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

    # one live slot per distinct fitted model: retire challengers that duplicate the champion or each other.
    # Models without an evaluation under the current accounting are identified from their spec, so an
    # older duplicate (e.g. a neighbour whose tweak recreated an existing model) cannot keep a slot.
    def live_identity(k: str, v: dict) -> str:
        return ledger.identity(dev[k]) if k in dev else effective_signature(v["spec"], data_sig)

    champ = reg.get("champion")
    seen = {live_identity(champ, reg["models"][champ])} if champ in reg["models"] else set()
    for k, v in reg["models"].items():
        if v.get("role") == "challenger":
            ident = live_identity(k, v)
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
    # a simple rule forecasts live beside the models, so "better than chance" is never the only bar:
    # favoring last week's losers (short-term reversal) nearly matched the champion on history
    bench = lcfg.get("benchmarkModel")
    if bench and not any(v.get("role") == "benchmark" for v in reg["models"].values()):
        b = next((e for e in dev.values() if e.get("name") == bench), None)
        if b:
            add(b, "benchmark", "simple rule the models must beat: favor last week's losers")
            reg["history"].append({"at": iso(now_et()), "event": "benchmark added", "to": b["id"]})
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


# ---------------------------------------------------------------- trading-rule trials
# Pre-registered on 2026-11-23 (docs/DECISIONS.md): rules for *trading* the champion's ranks, each a
# trial gated on after-cost excess over SPY. They never change a forecast, a model or the live portfolio.

def rule_specs() -> list[dict]:
    from ..util import read_json
    d = paths().lab / "rules"
    out = []
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        if f.name == "last_run.json":
            continue
        spec = read_json(f)
        if isinstance(spec, dict) and spec.get("kind") == "rule":
            out.append(spec)
    return out


RULE_PARAMS = {"holdings": "n", "exitRank": "exit_rank", "maxPerSector": "max_per_sector", "tradeEvery": "trade_every"}


def rule_kwargs(rule: dict) -> dict:
    p = rule.get("params", {})
    unknown = set(p) - set(RULE_PARAMS)
    if unknown:
        raise ValueError(f"unknown rule parameters {sorted(unknown)}")
    return {RULE_PARAMS[k]: v for k, v in p.items()}


def rule_id(rule: dict, base_id: str) -> str:
    import hashlib
    import json
    key = json.dumps({"params": rule.get("params", {}), "base": base_id}, sort_keys=True)
    return f"{rule['name']}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def rule_metrics(lo: pd.DataFrame, h: int, n_trials: int, sr_var: float | None) -> dict:
    from .metrics import METRICS_VERSION, deflated_sharpe, max_drawdown, sharpe
    if lo.empty:
        return {"metricsVersion": METRICS_VERSION, "longOnly": {}}
    ppy = 252 / h
    return {"metricsVersion": METRICS_VERSION, "longOnly": {
        "periods": int(len(lo)), "grossAnnual": float(lo["gross"].mean() * ppy), "netAnnual": float(lo["net"].mean() * ppy),
        "benchAnnual": float(lo["bench"].mean() * ppy), "excessAnnual": float(lo["excess"].mean() * ppy),
        "sharpeExcess": sharpe(lo["excess"], ppy), "turnover": float(lo["turnover"].mean()),
        "costAnnual": float((lo["gross"] - lo["net"]).mean() * ppy), "maxDrawdownExcess": max_drawdown(lo["excess"]),
        "deflated": deflated_sharpe(lo["excess"], n_trials, sr_var)}}


def rule_gates(m: dict, era: str) -> dict:
    """The existing thresholds applied to the rule's after-cost excess over SPY (no new numbers)."""
    g = section("lab")["gates"]
    lo = m.get("longOnly", {})
    if era == "dev":
        checks = {"excessSharpeSignificant": bool((lo.get("deflated", {}).get("psr") or 0) >= g["devNetPsrMin"]),
                  "positiveExcess": bool(lo.get("excessAnnual", -1) > 0)}
    else:
        checks = {"positiveExcess": bool(lo.get("excessAnnual", -1) > g["holdoutNetSpreadMin"])}
    return {"checks": checks, "passed": all(checks.values())}


def _rule_done(rid: str, era: str, data_sig: str) -> dict | None:
    from .metrics import METRICS_VERSION
    found = [e for e in ledger.entries() if e["id"] == rid and e["era"] == era and e.get("status") == ledger.RULE_STATUS
             and e.get("dataSig") == data_sig and int(e["metrics"].get("metricsVersion", 1)) == METRICS_VERSION]
    return found[-1] if found else None


def run_rules(ctx: Context | None = None) -> dict:
    """Evaluate the pre-registered trading rules on the champion's out-of-sample predictions.

    Development first (each rule one logged trial), then the sealed holdout for development passers,
    best development excess Sharpe first, within the weekly and yearly holdout caps. Each rule is
    evaluated once per data signature. Writes lab/rules/last_run.json and returns the report."""
    from .metrics import long_only
    t0 = time.time()
    rules = rule_specs()
    reg = ledger.registry()
    champ = reg.get("champion")
    report = {"startedAt": iso(now_et()), "champion": champ, "rules": [r["name"] for r in rules], "dev": [], "holdout": [],
              "baseline": {}, "selected": None, "decisionRule": "pass both gates; if several, highest development sharpeExcess"}
    if not rules or not champ or champ not in reg["models"]:
        report["note"] = "nothing to do (no rules or no champion)"
        write_json(paths().lab / "rules" / "last_run.json", report)
        return report
    base = reg["models"][champ]["spec"]
    h = int(base.get("horizon", 5))
    ctx = ctx or load_context()
    idx = PanelIndex(build_panel(ctx, list(REGISTRY)))
    data_sig = ",".join(available_families(idx))
    base_eff = effective_signature(base, data_sig)
    report["dataSig"] = data_sig
    oos_cache: dict[str, pd.DataFrame] = {}

    def oos(era: str) -> pd.DataFrame:
        if era not in oos_cache:
            oos_cache[era], _, _ = walk_forward(idx, base, era)
        return oos_cache[era]

    def evaluate_rule(rule: dict, era: str) -> dict | None:
        rid = rule_id(rule, champ)
        prior = _rule_done(rid, era, data_sig)
        if prior:
            return prior
        o = oos(era)
        if o.empty:
            return None
        n_trials, sr_var = ledger.dev_trials()
        lo = long_only(o, "pred", h, **rule_kwargs(rule))
        m = rule_metrics(lo, h, n_trials + (1 if era == "dev" else 0), sr_var)
        entry = ledger.record({
            "id": rid, "name": rule["name"], "era": era, "status": ledger.RULE_STATUS,
            "reason": "pre-registered trading rule (docs/DECISIONS.md 2026-11-23)" if era == "dev" else "holdout test",
            "spec": {**rule, "base": champ, "baseSpec": base}, "dataSig": data_sig,
            "effSig": f"rule|{base_eff}|{sorted(rule.get('params', {}).items())}", "metrics": m, "gates": rule_gates(m, era),
            "code": code_version(), "oosStart": o["date"].min().date().isoformat(), "oosEnd": o["date"].max().date().isoformat(),
            "seconds": round(time.time() - t0, 1)})
        ledger.save_series(rid, era, {"longOnly": lo})
        lom = m.get("longOnly", {})
        L.info("rule %-22s %s excess=%+.4f sharpe=%+.3f psr=%.3f turnover=%.3f pass=%s", rule["name"], era,
               lom.get("excessAnnual", np.nan), lom.get("sharpeExcess", np.nan),
               lom.get("deflated", {}).get("psr", np.nan) or np.nan, lom.get("turnover", np.nan), entry["gates"]["passed"])
        return entry

    def row(e: dict) -> dict:
        lo = e["metrics"].get("longOnly", {})
        return {"id": e["id"], "name": e["name"], "excessAnnual": lo.get("excessAnnual"), "sharpeExcess": lo.get("sharpeExcess"),
                "psr": lo.get("deflated", {}).get("psr"), "turnover": lo.get("turnover"), "costAnnual": lo.get("costAnnual"),
                "gates": e["gates"]}

    # the current live rule on the same predictions, for comparison only (not a trial, not logged)
    for era in ("dev",):
        o = oos(era)
        if not o.empty:
            report["baseline"][era] = rule_metrics(long_only(o, "pred", h), h, 1, None)["longOnly"]

    dev = [e for e in (evaluate_rule(r, "dev") for r in rules) if e]
    report["dev"] = [row(e) for e in dev]
    passers = sorted([e for e in dev if e["gates"]["passed"]],
                     key=lambda e: e["metrics"]["longOnly"].get("sharpeExcess") or -np.inf, reverse=True)
    lab = section("lab")
    cap = min(int(lab["maxHoldoutTestsPerWeek"]) - ledger.holdout_tests_this_week(),
              int(lab["maxHoldoutTestsPerYear"]) - ledger.holdout_tests_this_week(days=365))
    by_name = {r["name"]: r for r in rules}
    held = []
    for e in passers:
        rid = e["id"]
        prior = _rule_done(rid, "holdout", data_sig)
        if prior is None:
            if cap <= 0:
                report.setdefault("waitingForHoldout", []).append(e["name"])
                continue
            cap -= 1
        hh = evaluate_rule(by_name[e["name"]], "holdout")
        if hh:
            held.append(hh)
    report["holdout"] = [row(e) for e in held]
    winners = [d for d in passers if any(x["id"] == d["id"] and x["gates"]["passed"] for x in held)]
    if winners:
        w = winners[0]
        report["selected"] = {"id": w["id"], "name": w["name"], "params": w["spec"].get("params", {})}
    report["finishedAt"] = iso(now_et())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    write_json(paths().lab / "rules" / "last_run.json", report)
    return report
