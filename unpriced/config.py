"""Configuration and repository paths.

Everything that changes how the experiment behaves lives in config/unpriced.json so that
it is versioned in git. Paths can be redirected with UNPRICED_ROOT / UNPRICED_STORE for
tests and local runs.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def store(self) -> Path:
        """Large regenerable data (prices, filings, features). Synced to a GitHub release, not git."""
        override = os.environ.get("UNPRICED_STORE")
        return Path(override) if override else self.root / "store"

    @property
    def lab(self) -> Path:
        return self.root / "lab"

    @property
    def live(self) -> Path:
        return self.root / "live"

    @property
    def site(self) -> Path:
        return self.root / "site"

    @property
    def site_data(self) -> Path:
        return self.root / "site" / "data"

    @property
    def models(self) -> Path:
        return self.store / "models"


def paths() -> Paths:
    return Paths(Path(os.environ.get("UNPRICED_ROOT", REPO_ROOT)))


@lru_cache(maxsize=1)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def config() -> dict:
    path = os.environ.get("UNPRICED_CONFIG", str(REPO_ROOT / "config" / "unpriced.json"))
    return _load(path)


def section(name: str) -> dict:
    return config()[name]
