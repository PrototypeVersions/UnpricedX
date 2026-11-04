"""Long-format modeling panel: cross-sectionally ranked features + forward labels.

Features are converted to within-date percentile ranks in [-0.5, 0.5] across the universe,
which removes market-wide level shifts and makes every feature comparable over 20 years.
Missing values stay NaN: the boosted model handles them natively, linear models treat them as
the cross-sectional median (0).

Labels assume decisions at the close of day t and entry at the *next open*:
    fwd_h = open[t+1+h] / open[t+1] - 1      (total-return adjusted)
which is exactly what a paper trader acting on a frozen forecast could earn.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import section
from .context import Context
from .library import compute


def rank_cs(mat: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    x = mat.where(mask)
    return (x.rank(axis=1, pct=True) - 0.5).astype("float32")


def forward_returns(ctx: Context, h: int, cols=None) -> pd.DataFrame:
    o = ctx.mats["adj_open"]
    if cols is not None:
        o = o[cols]
    return o.shift(-(h + 1)) / o.shift(-1) - 1


def build_panel(ctx: Context, names: list[str], horizons: list[int] | None = None, mask_name: str = "research",
                start=None, end=None, labels: bool = True) -> pd.DataFrame:
    horizons = horizons or section("labels")["horizons"]
    mask = ctx.member[mask_name].fillna(False).astype(bool)
    if start is not None:
        mask = mask & (mask.index >= pd.Timestamp(start))[:, None]
    if end is not None:
        mask = mask & (mask.index <= pd.Timestamp(end))[:, None]
    rows, cols = np.nonzero(mask.to_numpy())
    dates, tick = mask.index, mask.columns
    data: dict[str, np.ndarray] = {
        "date": dates[rows].values,
        "ticker": tick[cols].values,
    }
    feats = compute(ctx, names)
    for n, m in feats.items():
        r = rank_cs(m.reindex(index=dates, columns=tick), mask)
        data[n] = r.to_numpy()[rows, cols]
    data["liq_rank"] = ctx.member["liq_rank"].reindex(index=dates, columns=tick).to_numpy()[rows, cols].astype("float32")
    data["tradable"] = ctx.member["tradable"].reindex(index=dates, columns=tick).fillna(False).to_numpy()[rows, cols]
    data["industry"] = ctx.industry.reindex(index=dates, columns=tick).to_numpy()[rows, cols]
    if ctx.sector is not None:
        data["sector"] = ctx.sector.reindex(index=dates, columns=tick).to_numpy()[rows, cols]
    if labels:
        bench = section("data")["benchmark"]
        for h in horizons:
            fwd = forward_returns(ctx, h, tick)
            vals = fwd.to_numpy()[rows, cols]
            data[f"fwd_{h}"] = vals.astype("float32")
            data[f"target_{h}"] = rank_cs(fwd, mask & fwd.notna()).to_numpy()[rows, cols]
            bfwd = forward_returns(ctx, h, [bench])[bench].reindex(dates)
            data[f"bench_{h}"] = bfwd.to_numpy()[rows].astype("float32")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df
