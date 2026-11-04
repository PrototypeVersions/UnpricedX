import gzip
import json

import numpy as np
import pandas as pd
import pytest

from unpriced.live import sequential
from unpriced.live import snapshot as snap


@pytest.mark.parametrize("sd", [0.05, 0.16])
def test_confidence_sequence_coverage_under_null(sd):
    """Under no skill, the band may exclude zero at *any* time after the minimum block count
    in at most ~alpha of histories, even though we look every week."""
    rng = np.random.default_rng(1)
    misses, sims = 0, 400
    for _ in range(sims):
        x = np.clip(rng.normal(0.0, sd, 520), -1, 1)
        cs = sequential.skill(x, alpha=0.05)
        if any(lo > 0 or up < 0 for lo, up in list(zip(cs["lower"], cs["upper"]))[26:]):
            misses += 1
    assert misses / sims <= 0.06


@pytest.mark.parametrize("rho", [0.2, 0.4])
def test_confidence_sequence_coverage_with_persistent_regimes(rho):
    """Weekly results are serially correlated in real markets. Under no skill with AR(1)
    correlation, the band still rarely excludes zero after the minimum block count."""
    rng = np.random.default_rng(3)
    misses, sims = 0, 300
    for _ in range(sims):
        e = rng.normal(0, 0.08 * np.sqrt(1 - rho**2), 520)
        x = np.empty(520)
        x[0] = rng.normal(0, 0.08)
        for i in range(1, 520):
            x[i] = rho * x[i - 1] + e[i]
        cs = sequential.skill(x, alpha=0.05)
        if any(lo > 0 or up < 0 for lo, up in list(zip(cs["lower"], cs["upper"]))[26:]):
            misses += 1
    assert misses / sims <= 0.08


def test_confidence_sequence_detects_real_edge():
    rng = np.random.default_rng(2)
    a = np.clip(rng.normal(0.12, 0.15, 400), -1, 1)
    b = np.clip(rng.normal(0.0, 0.15, 400), -1, 1)
    assert sequential.compare(a, b)["verdict"] == "challenger better"


def test_hash_chain_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setenv("UNPRICED_ROOT", str(tmp_path))
    for i, d in enumerate(["2026-01-05", "2026-01-06", "2026-01-07"]):
        snap.write({"session": d, "decidedAt": f"{d}T06:10:00-05:00", "entrySession": d, "models": {}, "universe": [], "tradable": []})
    assert snap.verify_chain()["ok"]
    p = snap.snapshot_path(pd.Timestamp("2026-01-06"))
    data = json.loads(gzip.decompress(p.read_bytes()))
    data["models"] = {"x": {"score": [1.0]}}
    p.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
    v = snap.verify_chain()
    assert not v["ok"] and any("hash mismatch" in x for x in v["problems"])
    with pytest.raises(FileExistsError):
        snap.write({"session": "2026-01-07", "decidedAt": "x", "entrySession": "x", "models": {}, "universe": [], "tradable": []})


def test_entry_session_is_first_open_after_decision():
    from unpriced.util import ET
    from datetime import datetime
    early = datetime(2026, 1, 6, 6, 10, tzinfo=ET)
    late = datetime(2026, 1, 6, 11, 0, tzinfo=ET)
    assert snap.entry_session(early, "2026-01-05") == pd.Timestamp("2026-01-06")
    assert snap.entry_session(late, "2026-01-05") == pd.Timestamp("2026-01-07")


def test_portfolio_fills_at_open_and_includes_dividends(tmp_path, monkeypatch):
    monkeypatch.setenv("UNPRICED_ROOT", str(tmp_path))
    from unpriced.live import portfolio
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    adj_open = pd.DataFrame({"AAA": [10.0, 10.0, 11.0], "SPY": [100.0, 100.0, 101.0]}, index=dates)
    st = portfolio.state()
    st["pending"] = {"model": "m", "decidedFrom": "2026-01-05", "decidedAt": "x", "executeSession": "2026-01-06",
                     "targets": ["AAA"], "costBps": {"AAA": 0.0}}
    trades = portfolio.execute(st, adj_open.loc[dates[1]], dates[1], adj_open)
    assert len(trades) == 1 and st["benchmark"]["anchor"] == 100.0
    # later a dividend re-bases history: the entry open is scaled down, current price unchanged
    rebased = adj_open.copy()
    rebased.loc[dates[:2], "AAA"] *= 0.98
    portfolio.mark(st, dates[2], pd.Series({"AAA": 11.0, "SPY": 101.0}), rebased)
    assert st["equity"] == pytest.approx(1000 * 11.0 / 9.8, rel=1e-6)
