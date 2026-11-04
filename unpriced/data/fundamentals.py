"""Point-in-time fundamentals from EDGAR XBRL company facts.

Each value is stamped with the date it was *first filed*; restated numbers never overwrite
what was knowable earlier. Flow items are converted to fiscal-quarter values (direct 3-month
facts, or differences of year-to-date facts) and then to trailing-twelve-month sums.

Output table `fundamentals`: one row per (ticker, fiscal quarter) with the date the row
became public (`available`) and ratio inputs. Features apply a further 1-session lag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sec import FLOW
from ..util import log

L = log("fundamentals")

QMIN, QMAX = 75, 105   # days in a fiscal quarter


def _first_reported(f: pd.DataFrame) -> pd.DataFrame:
    """One value per period: best tag priority, then the earliest filing that reported it."""
    f = f.sort_values(["prio", "filed"])
    return f.drop_duplicates(["start", "end"], keep="first")


def quarterly_flow(f: pd.DataFrame) -> pd.DataFrame:
    """Fiscal-quarter values for one flow concept of one company: columns end, val, avail."""
    f = _first_reported(f.dropna(subset=["start"]))
    if f.empty:
        return pd.DataFrame(columns=["end", "val", "avail"])
    f = f.assign(days=(f["end"] - f["start"]).dt.days)
    q: dict[pd.Timestamp, tuple[float, pd.Timestamp]] = {}
    for r in f[(f["days"] >= QMIN) & (f["days"] <= QMAX)].itertuples():
        q[r.end] = (r.val, r.filed)
    # Year-to-date chains: same fiscal-year start, consecutive ends one quarter apart.
    for _, grp in f[f["days"] >= QMIN].groupby("start"):
        g = grp.sort_values("end")
        ends, vals, filed = list(g["end"]), list(g["val"]), list(g["filed"])
        for i in range(1, len(g)):
            gap = (ends[i] - ends[i - 1]).days
            if QMIN <= gap <= QMAX and ends[i] not in q:
                q[ends[i]] = (vals[i] - vals[i - 1], max(filed[i], filed[i - 1]))
    out = pd.DataFrame([(e, v, a) for e, (v, a) in q.items()], columns=["end", "val", "avail"])
    return out.sort_values("end").reset_index(drop=True)


def ttm(qdf: pd.DataFrame) -> pd.DataFrame:
    """Trailing-twelve-month sums where four consecutive quarters exist."""
    if len(qdf) < 4:
        return pd.DataFrame(columns=["end", "ttm", "avail"])
    q = qdf.sort_values("end").reset_index(drop=True)
    rows = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3 : i + 1]
        span = (w["end"].iloc[-1] - w["end"].iloc[0]).days
        if 240 <= span <= 300:
            rows.append((w["end"].iloc[-1], float(w["val"].sum()), w["avail"].max()))
    return pd.DataFrame(rows, columns=["end", "ttm", "avail"])


def instant(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["prio", "filed"]).drop_duplicates(["end"], keep="first")
    return f[["end", "val", "filed"]].rename(columns={"filed": "avail"}).sort_values("end").reset_index(drop=True)


def _asof(left: pd.DataFrame, right: pd.DataFrame, col: str, tolerance_days: int = 120) -> pd.Series:
    if right.empty:
        return pd.Series(np.nan, index=left.index)
    m = pd.merge_asof(left[["end"]].reset_index(), right[["end", col]].sort_values("end"), on="end",
                      direction="backward", tolerance=pd.Timedelta(days=tolerance_days))
    return m.set_index("index")[col]


def company_quarters(facts: pd.DataFrame) -> pd.DataFrame:
    """Quarterly fundamentals for one company, each row stamped with its availability date."""
    by = {c: g for c, g in facts.groupby("concept")}
    flows = {c: quarterly_flow(by[c]) for c in FLOW if c in by}
    if "gross_profit" not in flows and "revenue" in flows and "cost_of_revenue" in flows:
        rv, cr = flows["revenue"], flows["cost_of_revenue"]
        m = rv.merge(cr, on="end", suffixes=("_r", "_c"))
        flows["gross_profit"] = pd.DataFrame({"end": m["end"], "val": m["val_r"] - m["val_c"],
                                              "avail": m[["avail_r", "avail_c"]].max(axis=1)})
    anchor = flows.get("revenue")
    if anchor is None or anchor.empty:
        anchor = flows.get("net_income", pd.DataFrame(columns=["end", "val", "avail"]))
    if anchor.empty:
        return pd.DataFrame()
    base = anchor[["end"]].drop_duplicates().sort_values("end").reset_index(drop=True)
    avail_cols = []
    for c in ["revenue", "gross_profit", "net_income", "operating_income", "cfo"]:
        if c in flows and len(flows[c]):
            t = ttm(flows[c])
            if len(t):
                m = base.merge(t, on="end", how="left")
                base[f"{c}_ttm"] = m["ttm"].values
                base[f"avail_{c}"] = m["avail"].values
                avail_cols.append(f"avail_{c}")
    if "eps_diluted" in flows:
        m = base.merge(flows["eps_diluted"], on="end", how="left")
        base["eps_q"] = m["val"].values
        base["avail_eps"] = m["avail"].values
        avail_cols.append("avail_eps")
    for c in ["assets", "equity", "shares_out"]:
        if c in by:
            inst = instant(by[c])
            base[c] = _asof(base, inst.rename(columns={"val": c}), c).values
    if not avail_cols:
        return pd.DataFrame()
    base["available"] = base[avail_cols].max(axis=1)
    base = base.drop(columns=avail_cols).dropna(subset=["available"])
    base = base.sort_values("end").reset_index(drop=True)

    def lag(col: str, k: int = 4) -> pd.Series:
        s = base[col] if col in base else pd.Series(np.nan, index=base.index)
        prev_end = base["end"].shift(k)
        ok = ((base["end"] - prev_end).dt.days.between(330, 400))
        return s.shift(k).where(ok)

    def col(c: str) -> pd.Series:
        return base[c] if c in base else pd.Series(np.nan, index=base.index)

    assets, assets_ly = col("assets"), lag("assets")
    base["asset_growth"] = assets / assets_ly - 1
    base["sales_growth"] = col("revenue_ttm") / lag("revenue_ttm") - 1
    avg_assets = (assets + assets_ly.fillna(assets)) / 2
    base["gp_assets"] = col("gross_profit_ttm") / assets
    base["accruals"] = (col("net_income_ttm") - col("cfo_ttm")) / avg_assets
    base["roe"] = col("net_income_ttm") / col("equity").where(col("equity") > 0)
    base["issuance"] = col("shares_out") / lag("shares_out") - 1
    if "eps_q" in base:
        diff = base["eps_q"] - lag("eps_q")
        sd = diff.rolling(8, min_periods=4).std().shift(1)
        base["sue"] = diff / sd.where(sd > 1e-6)
    for c in ["asset_growth", "sales_growth", "issuance"]:
        base[c] = base[c].where(base[c].abs() < 10)
    return base


def build_fundamentals(facts: pd.DataFrame, cik_to_ticker: dict[int, str]) -> pd.DataFrame:
    parts = []
    for cik, grp in facts.groupby("cik"):
        t = cik_to_ticker.get(int(cik))
        if not t:
            continue
        try:
            q = company_quarters(grp)
        except Exception as exc:  # one bad filer must not sink the run
            L.warning("fundamentals failed for CIK %s: %s", cik, exc)
            continue
        if len(q):
            q.insert(0, "ticker", t)
            parts.append(q)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["available"] = pd.to_datetime(out["available"])
    return out
