"""Frozen daily forecasts with a tamper-evident hash chain.

Each snapshot is written once, as deterministic gzip, and its SHA-256 is appended to
live/chain.jsonl together with the previous snapshot's hash. Editing any past snapshot breaks
every later link, which `verify_chain` (run in CI) detects. Git history and the Actions run log
provide independent timestamps for when each forecast existed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..calendar import next_session, session_open
from ..config import paths
from ..util import append_jsonl, read_json, read_jsonl, sha256_bytes, sha256_file, write_json_gz, ET


def snapshot_path(session: pd.Timestamp) -> Path:
    s = pd.Timestamp(session)
    return paths().live / "snapshots" / f"{s:%Y}" / f"{s:%Y-%m-%d}.json.gz"


def chain() -> list[dict]:
    return read_jsonl(paths().live / "chain.jsonl")


def exists(session) -> bool:
    return snapshot_path(session).exists()


class PublicationWindowMissed(RuntimeError):
    """The registered next-open target can no longer be forecast in advance."""


def entry_session(decided_at, after_session) -> pd.Timestamp:
    """Keep the next-session target fixed; never shift a late forecast to another open."""
    entry = next_session(after_session)
    op = session_open(entry.date().isoformat())
    decided = pd.Timestamp(decided_at)
    if decided.tzinfo is None:
        raise ValueError("forecast issue time must include a timezone")
    if op is None:
        raise RuntimeError(f"no opening time for {entry.date()}")
    if decided >= op:
        raise PublicationWindowMissed(
            f"forecast for {pd.Timestamp(after_session).date()} missed its fixed entry "
            f"at {op.isoformat()}; no forecast or new order will be issued"
        )
    return entry


def write(snapshot: dict) -> dict:
    links = chain()
    prev = links[-1]["sha256"] if links else None
    snapshot = {**snapshot, "prevHash": prev}
    path = snapshot_path(pd.Timestamp(snapshot["session"]))
    if path.exists():
        raise FileExistsError(f"snapshot already frozen: {path}")
    data = write_json_gz(path, snapshot)
    link = {"session": snapshot["session"], "file": str(path.relative_to(paths().root)), "sha256": sha256_bytes(data),
            "prevHash": prev, "decidedAt": snapshot["decidedAt"], "entrySession": snapshot["entrySession"]}
    append_jsonl(paths().live / "chain.jsonl", link)
    return link


def load(session) -> dict:
    return read_json(snapshot_path(pd.Timestamp(session)))


def verify_chain() -> dict:
    prev, problems = None, []
    links = chain()
    for link in links:
        p = paths().root / link["file"]
        if not p.exists():
            problems.append(f"missing {link['file']}")
            continue
        h = sha256_file(p)
        if h != link["sha256"]:
            problems.append(f"hash mismatch {link['file']}")
        if link["prevHash"] != prev:
            problems.append(f"broken link at {link['session']}")
        snap = read_json(p)
        if snap.get("prevHash") != prev:
            problems.append(f"embedded prevHash mismatch at {link['session']}")
        prev = link["sha256"]
    return {"links": len(links), "ok": not problems, "problems": problems}


def frame(snap: dict, model_id: str) -> pd.DataFrame:
    m = snap["models"][model_id]
    return pd.DataFrame({"ticker": snap["universe"], "score": m["score"], "tradable": snap["tradable"]})
