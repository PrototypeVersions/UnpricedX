"""Columnar storage for large, regenerable data.

Parquet (pyarrow) is used whenever available, which is always the case in CI and Actions.
Offline test environments without pyarrow fall back to pickle with the same API.
The store directory is synced to a GitHub release (see .github/workflows), never to git.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import paths
from .util import read_json, sha256_file, write_json, now_et, iso

try:
    import pyarrow  # noqa: F401
    HAS_PARQUET = True
except Exception:  # pragma: no cover
    HAS_PARQUET = False

EXT = ".parquet" if HAS_PARQUET else ".pkl"


def _path(name: str) -> Path:
    return paths().store / f"{name}{EXT}"


def exists(name: str) -> bool:
    return _path(name).exists()


def read(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = _path(name)
    if not p.exists():
        raise FileNotFoundError(f"store table missing: {name} ({p})")
    if HAS_PARQUET:
        return pd.read_parquet(p, columns=columns)
    df = pd.read_pickle(p)
    return df[columns] if columns else df


def read_or_empty(name: str) -> pd.DataFrame:
    return read(name) if exists(name) else pd.DataFrame()


def write(name: str, df: pd.DataFrame) -> Path:
    p = _path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if HAS_PARQUET:
        df.to_parquet(tmp, index=False, compression="zstd")
    else:
        df.to_pickle(tmp)
    tmp.replace(p)
    _touch_manifest(name, p)
    return p


def _touch_manifest(name: str, p: Path) -> None:
    mpath = paths().store / "manifest.json"
    manifest = read_json(mpath, {"tables": {}})
    manifest["tables"][name] = {"file": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size, "updatedAt": iso(now_et())}
    write_json(mpath, manifest)


def meta() -> dict:
    return read_json(paths().store / "meta.json", {})


def save_meta(m: dict) -> None:
    write_json(paths().store / "meta.json", m)
