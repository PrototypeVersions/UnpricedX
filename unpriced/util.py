"""Small shared helpers: JSON/IO, hashing, logging, time."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=os.environ.get("UNPRICED_LOG", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)


def log(name: str) -> logging.Logger:
    return logging.getLogger(f"unpriced.{name}")


def now_et() -> datetime:
    override = os.environ.get("UNPRICED_NOW")
    if override:
        dt = datetime.fromisoformat(override)
        return dt if dt.tzinfo else dt.replace(tzinfo=ET)
    return datetime.now(ET)


def iso(dt: datetime | pd.Timestamp | None) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ET)
    return dt.astimezone(ET).isoformat(timespec="seconds")


def _clean(obj: Any) -> Any:
    """Make numpy/pandas values JSON-safe; NaN/inf become null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        if obj.tzinfo is None and obj == obj.normalize():
            return obj.date().isoformat()
        return iso(obj)
    if isinstance(obj, datetime):
        return iso(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def dumps(obj: Any, indent: int | None = 2) -> str:
    return json.dumps(_clean(obj), indent=indent, sort_keys=False, allow_nan=False)


def write_json(path: Path, obj: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(obj, indent) + "\n")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    return json.loads(path.read_text())


def write_json_gz(path: Path, obj: Any) -> bytes:
    """Write deterministic gzip (mtime=0) so the file hash depends only on content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (dumps(obj, indent=None) + "\n").encode()
    data = gzip.compress(raw, compresslevel=9, mtime=0)
    path.write_bytes(data)
    return data


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(dumps(obj, indent=None) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj: Any, n: int = 10) -> str:
    return sha256_bytes(dumps(obj, indent=None).encode())[:n]
