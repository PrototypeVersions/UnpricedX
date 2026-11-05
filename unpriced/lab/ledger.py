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


def _version(e: dict) -> int:
    return int(e.get("metrics", {}).get("metricsVersion", 1))


def has(spec_id: str, era: str, data_sig: str | None = None, eff_sig: str | None = None) -> bool:
    """True if this model (by spec id, or by effective signature: the same features actually
    available, kind, horizon and parameters) was already evaluated in this era on the same data
    with the current accounting. Re-evaluating the same model is not a new trial."""
    from .metrics import METRICS_VERSION
    for e in entries():
        if e["era"] != era or e.get("status") != "evaluated" or _version(e) != METRICS_VERSION:
            continue
        if data_sig is not None and e.get("dataSig") != data_sig:
            continue
        if e["id"] == spec_id or (eff_sig is not None and e.get("effSig") == eff_sig):
            return True
    return False


def latest(spec_id: str, era: str) -> dict | None:
    found = [e for e in entries() if e["id"] == spec_id and e["era"] == era and e.get("status") == "evaluated"]
    return found[-1] if found else None


def current(era: str, data_sig: str | None = None) -> dict[str, dict]:
    """Latest evaluation per spec id in an era, current accounting only."""
    from .metrics import METRICS_VERSION
    out = {}
    for e in entries():
        if e["era"] == era and e.get("status") == "evaluated" and _version(e) == METRICS_VERSION \
                and (data_sig is None or e.get("dataSig") == data_sig):
            out[e["id"]] = e
    return out


def identity(e: dict) -> str:
    if e.get("effSig"):
        return e["effSig"]
    from ..models import effective_signature
    try:
        return effective_signature(e["spec"], e.get("dataSig", ""))
    except Exception:
        return e["id"]


def dev_trials() -> tuple[int, float | None]:
    """Distinct development trials so far, and the variance of their latest per-period net Sharpe.

    A trial is a distinct model actually fit (see models.effective_signature): re-scoring the same
    model with new accounting, or a spec that collapses to an existing one because some data is
    unavailable, does not count again.
    """
    latest: dict[str, dict] = {}
    for e in entries():
        if e["era"] == "dev" and e.get("status") == "evaluated":
            latest[identity(e)] = e
    srs = [e["metrics"].get("longShort", {}).get("deflated", {}).get("srPerPeriod") for e in latest.values()]
    srs = [s for s in srs if s is not None and np.isfinite(s)]
    var = float(np.var(srs, ddof=1)) if len(srs) >= 5 else None
    return len(latest), var


def holdout_tests_this_week(days: int = 7) -> int:
    since = pd.Timestamp(now_et()).tz_localize(None) - pd.Timedelta(days=days)
    return sum(1 for e in entries() if e["era"] == "holdout" and pd.Timestamp(e["at"][:19]) >= since)


def holdout_looks() -> int:
    """Distinct models ever scored on the sealed holdout. Each look leaks a little information,
    so the holdout gate's bar rises with this count (Bonferroni), like the development gate's."""
    return len({identity(e) for e in entries() if e["era"] == "holdout" and e.get("status") == "evaluated"})


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
    return [k for k, v in reg["models"].items() if v.get("role") in ("champion", "challenger", "benchmark")]
