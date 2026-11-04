"""Champion/challenger decisions from the live record only.

Each day, for every challenger, an anytime-valid confidence sequence for
E[IC_challenger - IC_champion] is updated on paired, non-overlapping blocks at the primary
horizon. After at least `minBlocks` blocks:
  * band entirely above 0 -> the challenger is promoted to champion (old champion stays as a challenger)
  * band entirely below 0 -> the challenger is retired, freeing a slot for the lab
Because the band is valid under continuous monitoring, checking daily does not inflate errors.
Models never get credit for any period before their first frozen forecast.
"""
from __future__ import annotations

import numpy as np

from ..config import section
from ..lab import ledger
from ..util import iso, now_et, write_json
from ..config import paths
from . import scoring
from .sequential import compare, skill


def review(reg: dict | None = None) -> dict:
    reg = reg or ledger.registry()
    cfg = section("live")["promotion"]
    h = int(section("live")["primaryHorizon"])
    df = scoring.scores()
    out = {"comparisons": {}, "skill": {}, "events": []}
    champ = reg.get("champion")
    if df.empty or not champ:
        write_json(paths().live / "sequential.json", out)
        return {"events": []}
    for mid in sorted(set(df["model"])):
        b = scoring.model_blocks(df, mid, h)
        cs = skill(b["ic"].to_numpy(), alpha=float(cfg["alpha"]))
        out["skill"][mid] = {"n": cs["n"], "mean": cs["mean"][-1] if cs["n"] else None,
                             "lower": cs["lower"][-1] if cs["n"] else None, "upper": cs["upper"][-1] if cs["n"] else None,
                             "verdict": cs["verdict"], "sessions": list(b["session"]),
                             "path": {"mean": cs["mean"], "lower": cs["lower"], "upper": cs["upper"]}}
    for mid in sorted(set(df["model"]) - {champ}):
        pairs = scoring.paired_blocks(df, mid, champ, h)
        cs = compare(pairs["a"].to_numpy(), pairs["b"].to_numpy(), alpha=float(cfg["alpha"]))
        out["comparisons"][mid] = {"vs": champ, "n": cs["n"], "verdict": cs["verdict"],
                                   "mean": cs["mean"][-1] if cs["n"] else None,
                                   "lower": cs["lower"][-1] if cs["n"] else None, "upper": cs["upper"][-1] if cs["n"] else None,
                                   "path": {"mean": cs["mean"], "lower": cs["lower"], "upper": cs["upper"]},
                                   "sessions": list(pairs["session"])}
        if cs["n"] < int(cfg["minBlocks"]) or mid not in reg["models"]:
            continue
        if cs["verdict"] == "challenger better" and reg["models"][mid].get("role") == "challenger":
            reg["models"][champ]["role"] = "challenger"
            reg["models"][mid]["role"] = "champion"
            reg["champion"], reg["championStatus"] = mid, "promoted-live"
            ev = {"at": iso(now_et()), "event": "promoted on live evidence", "from": champ, "to": mid, "blocks": cs["n"],
                  "lower": cs["lower"][-1]}
            reg["history"].append(ev)
            out["events"].append(ev)
            champ = mid
        elif cs["verdict"] == "challenger worse" and reg["models"][mid].get("role") == "challenger":
            reg["models"][mid]["role"] = "retired"
            ev = {"at": iso(now_et()), "event": "retired on live evidence", "model": mid, "blocks": cs["n"],
                  "upper": cs["upper"][-1]}
            reg["history"].append(ev)
            out["events"].append(ev)
    if out["events"]:
        ledger.save_registry(reg)
    write_json(paths().live / "sequential.json", out)
    return {"events": out["events"]}
