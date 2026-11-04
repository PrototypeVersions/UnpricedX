"""Score matured live forecasts against what actually happened.

For each frozen snapshot, model and horizon h, once the market has traded h sessions past the
snapshot's entry session, we record:
  ic        rank correlation of score vs realized open-to-open return
  spread    top-decile minus bottom-decile mean return
  topExcess top-decile mean return minus the universe mean
Results are appended to live/scores.jsonl exactly once per (session, model, horizon).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import paths, section
from ..util import append_jsonl, read_jsonl
from . import snapshot as snap_mod


def scores() -> pd.DataFrame:
    rows = read_jsonl(paths().live / "scores.jsonl")
    return pd.DataFrame(rows)


def realized(adj_open: pd.DataFrame, entry: pd.Timestamp, h: int, tickers: list[str]) -> pd.Series | None:
    dates = adj_open.index
    if entry not in dates:
        return None
    i = dates.get_loc(entry)
    if i + h >= len(dates):
        return None
    o0 = adj_open.iloc[i].reindex(tickers)
    o1 = adj_open.iloc[i + h].reindex(tickers)
    return o1 / o0 - 1


def score_snapshot(snap: dict, adj_open: pd.DataFrame, done: set) -> list[dict]:
    out = []
    entry = pd.Timestamp(snap["entrySession"])
    for h in section("labels")["horizons"]:
        r = realized(adj_open, entry, h, snap["universe"])
        if r is None:
            continue
        for mid, m in snap["models"].items():
            key = (snap["session"], mid, h)
            if key in done:
                continue
            s = pd.Series(m["score"], index=snap["universe"], dtype=float)
            ok = s.notna() & r.notna()
            if ok.sum() < 30:
                continue
            ss, rr = s[ok], r[ok]
            ic = float(ss.rank().corr(rr.rank()))
            k = max(5, int(ok.sum() * 0.1))
            top, bot = rr[ss.nlargest(k).index], rr[ss.nsmallest(k).index]
            out.append({"session": snap["session"], "entry": entry.date().isoformat(), "model": mid, "role": m.get("role"),
                        "horizon": h, "n": int(ok.sum()), "ic": round(ic, 6), "spread": round(float(top.mean() - bot.mean()), 6),
                        "topExcess": round(float(top.mean() - rr.mean()), 6), "universeMean": round(float(rr.mean()), 6)})
    return out


def update(adj_open: pd.DataFrame) -> int:
    existing = read_jsonl(paths().live / "scores.jsonl")
    done = {(r["session"], r["model"], r["horizon"]) for r in existing}
    new = 0
    horizons = section("labels")["horizons"]
    last = adj_open.index[-1]
    for link in snap_mod.chain():
        if link.get("entrySession") and pd.Timestamp(link["entrySession"]) > last:
            continue
        snap = snap_mod.load(pd.Timestamp(link["session"]))
        if all((snap["session"], m, h) in done for h in horizons for m in snap.get("models", {})):
            continue
        for row in score_snapshot(snap, adj_open, done):
            append_jsonl(paths().live / "scores.jsonl", row)
            done.add((row["session"], row["model"], row["horizon"]))
            new += 1
    return new


def paired_blocks(df: pd.DataFrame, a: str, b: str, h: int) -> pd.DataFrame:
    """Non-overlapping paired observations: sessions where both models were scored at horizon h,
    taking every h-th one so that no two observations share a holding period."""
    if df.empty:
        return pd.DataFrame(columns=["session", "a", "b"])
    x = df[(df["horizon"] == h)]
    pa = x[x["model"] == a].set_index("session")["ic"]
    pb = x[x["model"] == b].set_index("session")["ic"]
    both = pd.concat([pa.rename("a"), pb.rename("b")], axis=1, join="inner").sort_index()
    return both.iloc[::h].reset_index()


def model_blocks(df: pd.DataFrame, m: str, h: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["session", "ic"])
    x = df[(df["horizon"] == h) & (df["model"] == m)].sort_values("session")
    return x.iloc[::h]
