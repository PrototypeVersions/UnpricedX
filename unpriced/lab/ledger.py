"""Append-only experiment ledger and the live-model registry.

Every evaluation ever run (pass or fail) is a ledger line. The number of distinct development
trials feeds the deflated Sharpe ratio, so trying more ideas automatically raises the bar each
new idea has to clear. Ledger lines are never edited or deleted.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from ..config import paths
from ..util import append_jsonl, iso, now_et, read_json, read_jsonl, write_json, dumps


def ledger_path():
    return paths().lab / "ledger.jsonl"


def entries() -> list[dict]:
    return read_jsonl(ledger_path())


def has(spec_id: str, era: str, data_sig: str | None = None) -> bool:
    """True if this spec was already evaluated in this era on the same data availability."""
    return any(e["id"] == spec_id and e["era"] == era and e.get("status") == "evaluated"
               and (data_sig is None or e.get("dataSig") == data_sig) for e in entries())


def latest(spec_id: str, era: str) -> dict | None:
    found = [e for e in entries() if e["id"] == spec_id and e["era"] == era and e.get("status") == "evaluated"]
    return found[-1] if found else None


def dev_trials() -> tuple[int, float | None]:
    """Distinct development trials so far and the variance of their per-period net Sharpe ratios."""
    ent = [e for e in entries() if e["era"] == "dev" and e.get("status") == "evaluated"]
    ids = {e["id"] for e in ent}
    srs = [e["metrics"].get("longShort", {}).get("deflated", {}).get("srPerPeriod") for e in ent]
    srs = [s for s in srs if s is not None and np.isfinite(s)]
    var = float(np.var(srs, ddof=1)) if len(srs) >= 5 else None
    return len(ids), var


def holdout_tests_this_week() -> int:
    week_ago = pd.Timestamp(now_et()).tz_localize(None) - pd.Timedelta(days=7)
    n = 0
    for e in entries():
        if e["era"] == "holdout" and pd.Timestamp(e["at"][:19]) >= week_ago:
            n += 1
    return n


def record(entry: dict) -> dict:
    entry = {"at": iso(now_et()), **entry}
    append_jsonl(ledger_path(), entry)
    return entry


def save_series(spec_id: str, era: str, series: dict) -> None:
    """Daily IC and portfolio series for charts (small, gzipped, deterministic)."""
    out = {}
    ic = series.get("ic")
    if ic is not None and len(ic):
        out["ic"] = {"dates": [d.date().isoformat() for d in ic.index], "values": [round(float(v), 5) for v in ic.values]}
    for k in ("longShort", "longOnly"):
        df = series.get(k)
        if df is not None and len(df):
            out[k] = {"dates": [pd.Timestamp(d).date().isoformat() for d in df["date"]],
                      **{c: [round(float(v), 6) for v in df[c]] for c in df.columns if c != "date"}}
    p = paths().lab / "results" / f"{spec_id}.{era}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(dumps(out, indent=None).encode(), mtime=0))


def load_series(spec_id: str, era: str) -> dict:
    p = paths().lab / "results" / f"{spec_id}.{era}.json.gz"
    return json.loads(gzip.decompress(p.read_bytes())) if p.exists() else {}


# ---------------------------------------------------------------- registry
def registry_path():
    return paths().lab / "registry.json"


def registry() -> dict:
    return read_json(registry_path(), {"champion": None, "championStatus": None, "models": {}, "history": []})


def save_registry(reg: dict) -> None:
    reg["updatedAt"] = iso(now_et())
    write_json(registry_path(), reg)


def live_models(reg: dict | None = None) -> list[str]:
    reg = reg or registry()
    return [k for k, v in reg["models"].items() if v.get("role") in ("champion", "challenger")]
