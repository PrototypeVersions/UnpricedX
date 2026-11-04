"""Pre-registered experiments and the automatic proposal policy.

The initial set below was written before any real-data result was seen (Nov 2026). It includes
single-factor baselines from the literature so that every learned model has to beat simple,
documented signals, not just zero.

`DEFAULT_PROVISIONAL` is the model that runs live if nothing passes the gates. Choosing it in
advance keeps the first live champion from being cherry-picked after looking at results.
"""
from __future__ import annotations

import copy
from pathlib import Path

from ..config import paths
from ..features.library import REGISTRY
from ..util import read_json

PRICE = ["momentum", "reversal", "risk", "liquidity"]
ALL = PRICE + ["events", "fundamentals", "valuation"]

FACTORS = ["mom_12_1", "mom_6_1", "high_52w", "ind_mom_6", "ret_1m", "ret_1w", "ind_rel_1m", "overnight_1m",
           "intraday_1m", "vol_3m", "ivol_3m", "beta_1y", "max_1m", "ear", "earn_soon", "gp_assets",
           "asset_growth", "accruals", "issuance", "sue", "earnings_yield", "book_to_market"]


def initial_specs() -> list[dict]:
    factors = [{"name": f"factor_{f}", "kind": "factor", "factor": f, "sign": REGISTRY[f].sign, "horizon": 5}
               for f in FACTORS]
    specs = []
    ridge_price = {"name": "ridge_price", "kind": "ridge", "families": PRICE, "horizon": 5}
    ridge_all = {"name": "ridge_all", "kind": "ridge", "families": ALL, "horizon": 5}
    gbm_price = {"name": "gbm_price", "kind": "gbm", "families": PRICE, "horizon": 5}
    gbm_all = {"name": "gbm_all", "kind": "gbm", "families": ALL, "horizon": 5}
    ens_all = {"name": "ens_all", "kind": "ensemble", "horizon": 5,
               "members": [copy.deepcopy(ridge_all), copy.deepcopy(gbm_all)]}
    specs += [ridge_all, gbm_all, ens_all, ridge_price, gbm_price]     # learned models first (budget safety)
    specs += factors
    specs += [{**copy.deepcopy(gbm_all), "name": "gbm_all_h21", "horizon": 21},
              {**copy.deepcopy(ridge_all), "name": "ridge_all_h21", "horizon": 21}]
    return specs


DEFAULT_PROVISIONAL = "ens_all"


def queued_specs() -> list[tuple[Path, dict]]:
    q = paths().lab / "queue"
    out = []
    for f in sorted(q.glob("*.json")) if q.exists() else []:
        spec = read_json(f)
        if isinstance(spec, dict) and "kind" in spec:
            out.append((f, spec))
    return out


def neighbors(best: dict) -> list[dict]:
    """Small, interpretable perturbations of the current best learned model."""
    out = []
    fams = best.get("families", [])
    if best["kind"] in ("ridge", "gbm") and len(fams) > 2:
        for f in fams:   # what does each family contribute?
            out.append({**copy.deepcopy(best), "name": f"{best['name']}-no_{f}", "families": [x for x in fams if x != f]})
    if best["kind"] == "gbm":
        base = best.get("params", {})
        for k, vals in {"learning_rate": [0.02, 0.08], "max_leaf_nodes": [15, 63], "l2_regularization": [0.1, 10.0],
                        "max_features": [0.4, 1.0]}.items():
            for v in vals:
                out.append({**copy.deepcopy(best), "name": f"{best['name']}-{k}={v}", "params": {**base, k: v}})
    if best["kind"] == "ridge":
        for a in [5.0, 500.0]:
            out.append({**copy.deepcopy(best), "name": f"{best['name']}-alpha={a}", "params": {"alpha": a}})
    for h in (5, 21):
        if best.get("horizon") != h:
            s = copy.deepcopy(best)
            s["horizon"] = h
            for m in s.get("members", []):
                m["horizon"] = h
            s["name"] = f"{best['name']}-h{h}"
            out.append(s)
    return out
