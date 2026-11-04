"""No feature may change its past values when future data is removed (no look-ahead)."""
import numpy as np
import pandas as pd

from unpriced.features.context import load_context
from unpriced.features.library import REGISTRY, compute


def test_features_do_not_peek_into_the_future(market, ctx):
    names = list(REGISTRY)
    full = compute(ctx, names)
    cut = ctx.dates[int(len(ctx.dates) * 0.7)]
    prices = market.tables["prices"]
    trunc_ctx = load_context(prices=prices[prices["date"] <= cut])
    trunc = compute(trunc_ctx, names)
    check_from = ctx.dates[300]
    for n in names:
        a = full[n].loc[check_from:cut]
        b = trunc[n].reindex(index=a.index, columns=a.columns)
        both = a.notna() & b.notna()
        diff = (a[both] - b[both]).abs().max().max()
        assert not (diff > 1e-5), f"{n} changes when future data is removed (max diff {diff})"
        # values present with full data must also be computable without the future
        only_full = (a.notna() & b.isna()).to_numpy().sum()
        assert only_full <= 0.001 * a.size, f"{n} needs future data for {only_full} cells"


def test_labels_start_at_next_open(ctx):
    from unpriced.features.panel import forward_returns
    fwd = forward_returns(ctx, 5)
    o = ctx.mats["adj_open"]
    t = ctx.dates[500]
    i = ctx.dates.get_loc(t)
    tick = fwd.columns[0]
    expected = o[tick].iloc[i + 6] / o[tick].iloc[i + 1] - 1
    assert np.isclose(fwd.at[t, tick], expected, equal_nan=True)


def test_industries_use_only_past_returns(market, ctx):
    cut = ctx.dates[int(len(ctx.dates) * 0.8)]
    prices = market.tables["prices"]
    trunc = load_context(prices=prices[prices["date"] <= cut])
    for name in ("industry", "sector"):        # sectors need >= 200 names, so they are empty in this small market
        a = getattr(ctx, name).loc[:cut]
        b = getattr(trunc, name).reindex_like(a)
        both = a.notna() & b.notna()
        assert name == "sector" or both.to_numpy().any()
        assert ((a[both] - b[both]).abs().fillna(0).to_numpy() == 0).all()
