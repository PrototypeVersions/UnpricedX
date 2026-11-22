"""Tests for the tug-of-war (overnight vs. daytime) and intra-industry lead-lag features (research 2026-11-22)."""
import numpy as np
import pandas as pd

from unpriced.features.context import Context
from unpriced.features.library import REGISTRY, compute, ind_lead_1w, nr_freq_1m, pr_freq_1m


def _ctx(close: pd.DataFrame, open_: pd.DataFrame, market: pd.Series, industry: pd.DataFrame | None = None,
         dollar_volume: pd.DataFrame | None = None) -> Context:
    member = pd.DataFrame(True, index=close.index, columns=close.columns)
    empty = pd.DataFrame(columns=["ticker", "session", "form", "items"])
    mats = {"adj_close": close, "adj_open": open_, "raw_close": close,
            "dollar_volume": dollar_volume if dollar_volume is not None else close * 1e6}
    ind = industry if industry is not None else pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    return Context(mats=mats, member={"research": member}, market_ret=market, industry=ind,
                   events=empty, filings=empty, fundamentals=pd.DataFrame())


def test_new_families_leave_existing_models_unchanged():
    from unpriced.lab.specs import ALL
    for n, fam in (("nr_freq_1m", "tugofwar"), ("pr_freq_1m", "tugofwar"), ("ind_lead_1w", "leadlag")):
        assert REGISTRY[n].family == fam
        assert REGISTRY[n].doc and REGISTRY[n].ref
    assert "tugofwar" not in ALL and "leadlag" not in ALL


def test_tug_of_war_counts_overnight_up_daytime_down_sessions():
    dates = pd.bdate_range("2020-01-01", periods=60)
    n = len(dates)
    # A: every session gaps up 1% overnight and falls 1.5% during the day (tug of war every day)
    # B: every session gaps down 1% and rises 1.5% during the day (mirror pattern)
    # C: overnight and daytime both up (no pattern)
    close = pd.DataFrame(index=dates, columns=["A", "B", "C"], dtype=float)
    open_ = close.copy()
    pairs = {"A": (0.01, -0.015), "B": (-0.01, 0.015), "C": (0.004, 0.004)}
    for t, (on, dy) in pairs.items():
        c, o = [100.0], [100.0]
        for _ in range(1, n):
            o.append(c[-1] * np.exp(on))
            c.append(o[-1] * np.exp(dy))
        close[t], open_[t] = c, o
    m = pd.Series(0.0, index=dates)
    ctx = _ctx(close, open_, m)
    nr, pr = nr_freq_1m(ctx).iloc[-1], pr_freq_1m(ctx).iloc[-1]
    assert np.isclose(nr["A"], 1.0) and np.isclose(nr["B"], 0.0) and np.isclose(nr["C"], 0.0)
    assert np.isclose(pr["B"], 1.0) and np.isclose(pr["A"], 0.0) and np.isclose(pr["C"], 0.0)
    assert nr_freq_1m(ctx).iloc[:10].isna().all().all()     # needs >= 15 sessions


def _lead_lag_market(n_days: int = 120):
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    rng = np.random.default_rng(5)
    tick = [f"S{i}" for i in range(12)]
    r = pd.DataFrame(rng.normal(0, 0.01, (n_days, 12)), index=dates, columns=tick)
    close = 100 * (1 + r).cumprod()
    dv = pd.DataFrame(np.tile(np.arange(12, 0, -1) * 1e7, (n_days, 1)), index=dates, columns=tick)  # S0 biggest
    industry = pd.DataFrame(np.where(np.arange(12) < 6, 0.0, 1.0)[None, :].repeat(n_days, 0), index=dates, columns=tick)
    return dates, tick, r, close, dv, industry


def test_lead_lag_uses_big_peers_in_the_same_industry_excluding_self():
    dates, tick, r, close, dv, industry = _lead_lag_market()
    industry.iloc[:, :] = 0.0                            # one industry of 12, so the big third has four names
    m = pd.Series(0.0, index=dates)
    ctx = _ctx(close, close, m, industry=industry, dollar_volume=dv)
    f = ind_lead_1w(ctx).iloc[-1]
    r5 = close.iloc[-1] / close.iloc[-6] - 1
    big = ["S0", "S1", "S2", "S3"]                       # top third of 12 by dollar volume
    assert np.isclose(f["S11"], r5[big].mean())          # small firm: mean of the big four
    assert np.isclose(f["S0"], r5[["S1", "S2", "S3"]].mean())   # big firm: its own return is left out


def test_lead_lag_is_market_adjusted_and_needs_enough_peers():
    dates, tick, r, close, dv, industry = _lead_lag_market()
    m = pd.Series(0.002, index=dates)
    small = _ctx(close, close, m, industry=industry, dollar_volume=dv)   # two industries of six: two big names each
    assert ind_lead_1w(small).iloc[-1].isna().all()                       # fewer than three big peers
    industry.iloc[:, :] = 0.0
    ctx = _ctx(close, close, m, industry=industry, dollar_volume=dv)
    f = ind_lead_1w(ctx).iloc[-1]
    r5 = close.iloc[-1] / close.iloc[-6] - 1
    assert np.isclose(f["S11"], r5[["S0", "S1", "S2", "S3"]].mean() - (1.002 ** 5 - 1))
    no_ind = _ctx(close, close, m, industry=None, dollar_volume=dv)
    assert ind_lead_1w(no_ind).isna().all().all()


def test_lead_lag_does_not_peek():
    dates, tick, r, close, dv, industry = _lead_lag_market()
    m = pd.Series(0.0, index=dates)
    industry.iloc[:, :] = 0.0
    full = ind_lead_1w(_ctx(close, close, m, industry=industry, dollar_volume=dv))
    cut = 90
    part = ind_lead_1w(_ctx(close.iloc[:cut], close.iloc[:cut], m.iloc[:cut], industry=industry.iloc[:cut],
                            dollar_volume=dv.iloc[:cut]))
    a, b = full.iloc[:cut], part
    both = a.notna() & b.notna()
    assert both.to_numpy().any()
    assert np.allclose(a[both].fillna(0), b[both].fillna(0))


def test_compute_returns_float32_and_finite():
    dates, tick, r, close, dv, industry = _lead_lag_market()
    industry.iloc[:, :] = 0.0
    m = pd.Series(0.0, index=dates)
    out = compute(_ctx(close, close * 0.999, m, industry=industry, dollar_volume=dv),
                  ["nr_freq_1m", "pr_freq_1m", "ind_lead_1w"])
    for mat in out.values():
        assert mat.dtypes.eq("float32").all()
        assert not np.isinf(mat.to_numpy()).any()
