"""The $1,000 paper portfolio.

Decisions are taken from the frozen snapshot and executed at the *next session's open* (the
first open after the snapshot was committed). Fills are computed the following morning from the
official daily open, so a late-running job can never trade on information it did not have.

Positions are anchored in dollars at their last rebalance open and marked with total-return
adjusted prices, so dividends and splits are handled without tracking share counts.

Rules (config.portfolio): long-only, equal weight, `holdings` names from the tradable universe
ranked by the champion's score; a held name is kept until it falls below `exitRank`
(hysteresis to cut turnover); rebalance every `rebalanceEverySessions`; costs per side are a base
plus a liquidity-tier spread estimate. SPY bought at the first fill is the benchmark.
"""
from __future__ import annotations

import pandas as pd

from ..config import paths, section
from ..data.universe import cost_bps
from ..util import append_jsonl, iso, now_et, read_json, write_json


def path():
    return paths().live / "portfolio.json"


def state() -> dict:
    cfg = section("portfolio")
    return read_json(path(), {"startingCash": cfg["startingCash"], "cash": cfg["startingCash"], "positions": {},
                              "pending": None, "equity": cfg["startingCash"], "benchmark": None, "curve": [],
                              "rebalances": 0, "lastRebalanceSession": None, "sessionsSinceRebalance": None})


def _anchor(p: dict, ticker: str, adj_open: pd.DataFrame | None) -> float:
    """Entry price on the *current* adjustment basis, so later dividends/splits are included."""
    if adj_open is not None and ticker in adj_open.columns:
        d = pd.Timestamp(p["entrySession"])
        if d in adj_open.index and pd.notna(adj_open.at[d, ticker]):
            return float(adj_open.at[d, ticker])
    return float(p["anchor"])


def value(st: dict, adj_close: pd.Series, adj_open: pd.DataFrame | None = None) -> float:
    total = float(st["cash"])
    for t, p in st["positions"].items():
        px = adj_close.get(t)
        if px is not None and pd.notna(px):
            total += p["dollars"] * float(px) / _anchor(p, t, adj_open)
        else:
            total += p.get("lastValue", p["dollars"])
    return total


def mark(st: dict, session: pd.Timestamp, adj_close: pd.Series, adj_open: pd.DataFrame | None = None) -> None:
    for t, p in st["positions"].items():
        px = adj_close.get(t)
        if px is not None and pd.notna(px):
            p["lastValue"] = round(p["dollars"] * float(px) / _anchor(p, t, adj_open), 6)
            p["lastPrice"] = round(float(px), 6)
    st["equity"] = round(value(st, adj_close, adj_open), 6)
    b = st.get("benchmark")
    bench_val = None
    sym = section("data")["benchmark"]
    if b and adj_close.get(sym) is not None and pd.notna(adj_close.get(sym)):
        bench_val = round(b["dollars"] * float(adj_close[sym]) / _anchor({"entrySession": b["startSession"], "anchor": b["anchor"]}, sym, adj_open), 6)
        b["value"] = bench_val
    day = pd.Timestamp(session).date().isoformat()
    st["curve"] = [c for c in st["curve"] if c["session"] != day] + [{"session": day, "equity": st["equity"], "spy": bench_val}]
    st["markedAt"] = iso(now_et())


def decide(st: dict, snap: dict, model_id: str, liq_rank: pd.Series) -> dict | None:
    """Target holdings for the next open, or None if no rebalance is due."""
    cfg = section("portfolio")
    # counter is the number of sessions held since the last fill; the next fill happens one session after deciding
    due = st["lastRebalanceSession"] is None or (st.get("sessionsSinceRebalance") or 0) >= int(cfg["rebalanceEverySessions"]) - 1
    if not due:
        return None
    df = pd.DataFrame({"ticker": snap["universe"], "score": snap["models"][model_id]["score"], "tradable": snap["tradable"]})
    df = df[df["tradable"] & df["score"].notna()].sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    n, exit_rank = int(cfg["holdings"]), int(cfg["exitRank"])
    keep = [t for t in st["positions"] if t in set(df.loc[df["rank"] <= exit_rank, "ticker"])]
    fill = [t for t in df["ticker"] if t not in keep][: max(0, n - len(keep))]
    target = (keep + fill)[:n]
    return {"model": model_id, "decidedFrom": snap["session"], "decidedAt": snap["decidedAt"],
            "executeSession": snap["entrySession"], "targets": target,
            "costBps": {t: float(cost_bps(pd.Series([liq_rank.get(t, 1e6)])).iloc[0]) + float(cfg["baseCostBps"]) for t in target}}


def execute(st: dict, adj_open: pd.Series, session: pd.Timestamp, adj_open_hist: pd.DataFrame | None = None) -> list[dict]:
    """Fill pending orders at this session's open: sell everything not targeted, rebalance to equal weight."""
    order = st.get("pending")
    if not order or pd.Timestamp(order["executeSession"]) != pd.Timestamp(session):
        return []
    cfg = section("portfolio")
    day = pd.Timestamp(session).date().isoformat()
    trades = []
    # value current holdings at the open
    holdings = {}
    for t, p in st["positions"].items():
        px = adj_open.get(t)
        holdings[t] = p["dollars"] * float(px) / _anchor(p, t, adj_open_hist) if px is not None and pd.notna(px) else p.get("lastValue", p["dollars"])
    equity = float(st["cash"]) + sum(holdings.values())
    targets = [t for t in order["targets"] if adj_open.get(t) is not None and pd.notna(adj_open.get(t))]
    costs = order.get("costBps", {})
    band = float(cfg.get("noTradeBand", 0.02)) * equity     # let small drifts ride instead of churning

    def plan(tgt_val: float) -> tuple[dict, float]:
        want, fees = {}, 0.0
        for t in set(holdings) | set(targets):
            cur = holdings.get(t, 0.0)
            w = tgt_val if t in targets else 0.0
            if t in targets and cur > 0 and abs(w - cur) < band:
                w = cur
            want[t] = w
            fees += abs(w - cur) * costs.get(t, float(cfg["baseCostBps"]) + 15.0) / 1e4
        return want, fees

    tgt_val = equity / len(targets) if targets else 0.0
    for _ in range(3):                                       # size positions so cash never goes negative after fees
        want, fees = plan(tgt_val)
        invested = sum(want.values())
        if targets and invested + fees > equity:
            tgt_val *= (equity - fees) / max(invested, 1e-9)
        else:
            break
    new_positions, cash = {}, equity
    for t, w in want.items():
        cur = holdings.get(t, 0.0)
        traded = abs(w - cur)
        fee = traded * costs.get(t, float(cfg["baseCostBps"]) + 15.0) / 1e4
        cash -= fee
        if traded > 0.005:
            px = adj_open.get(t)
            trades.append({"session": day, "ticker": t, "side": "BUY" if w > cur else "SELL", "dollars": round(traded, 4),
                           "price": round(float(px), 6) if px is not None and pd.notna(px) else None,
                           "costDollars": round(fee, 4), "model": order["model"], "decidedFrom": order["decidedFrom"]})
        if w > 0:
            if t in st["positions"] and abs(w - cur) < 1e-9:
                new_positions[t] = st["positions"][t]           # untouched holding keeps its anchor
            else:
                new_positions[t] = {"dollars": w, "anchor": float(adj_open[t]), "entrySession": day}
    invested = sum(want[t] for t in new_positions)
    st["cash"] = round(cash - invested, 6)
    st["positions"] = new_positions
    st["pending"] = None
    st["rebalances"] += 1
    st["lastRebalanceSession"] = day
    st["sessionsSinceRebalance"] = 0
    bench_open = adj_open.get(section("data")["benchmark"])
    if st.get("benchmark") is None and bench_open is not None and pd.notna(bench_open):
        st["benchmark"] = {"symbol": section("data")["benchmark"], "dollars": float(st["startingCash"]), "anchor": float(bench_open),
                           "startSession": day}
        st["startSession"] = day
    for tr in trades:
        append_jsonl(paths().live / "trades.jsonl", tr)
    return trades


def save(st: dict) -> None:
    write_json(path(), st)
