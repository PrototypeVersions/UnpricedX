"""Tests for residual reversal and turnover / short-term momentum (research 2026-11-15)."""
import numpy as np
import pandas as pd

from unpriced.features.context import Context
from unpriced.features.library import REGISTRY, compute, resid_rev_1m, stmom_1m, turnover_1m


def _ctx(prices: pd.DataFrame, market: pd.Series, industry: pd.DataFrame | None = None,
         dollar_volume: pd.DataFrame | None = None, fundamentals: pd.DataFrame | None = None) -> Context:
    member = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    empty = pd.DataFrame(columns=["ticker", "session", "form", "items"])
    mats = {"adj_close": prices, "adj_open": prices, "raw_close": prices,
            "dollar_volume": dollar_volume if dollar_volume is not None else prices * 1e6}
    ind = industry if industry is not None else pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    return Context(mats=mats, member={"research": member}, market_ret=market, industry=ind,
                   events=empty, filings=empty, fundamentals=fundamentals if fundamentals is not None else pd.DataFrame())


def test_new_families_leave_existing_models_unchanged():
    assert REGISTRY["resid_rev_1m"].family == "residual"
    assert REGISTRY["turnover_1m"].family == "turnover"
    assert REGISTRY["stmom_1m"].family == "turnover"
    from unpriced.lab.specs import ALL
    assert "residual" not in ALL and "turnover" not in ALL
    for n in ("resid_rev_1m", "turnover_1m", "stmom_1m"):
        assert REGISTRY[n].doc and REGISTRY[n].ref


def test_residual_reversal_removes_market_exposure():
    dates = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(3)
    m = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    idio = rng.normal(0, 0.01, (len(dates), 3))
    betas = np.array([0.5, 1.0, 2.0])
    r = pd.DataFrame(m.to_numpy()[:, None] * betas + idio, index=dates, columns=["LOW", "MID", "HIGH"])
    m.iloc[-21:] = 0.01                       # the market rallies 1% a day in the last month
    r.iloc[-21:] = (m.iloc[-21:].to_numpy()[:, None] * betas)
    r.iloc[-21:, 1] += 0.005                  # MID also has a positive stock-specific month
    prices = 100 * (1 + r).cumprod()
    f = resid_rev_1m(_ctx(prices, m)).iloc[-1]
    raw = (prices.iloc[-1] / prices.iloc[-22] - 1)
    assert raw["HIGH"] > raw["MID"] > raw["LOW"]          # raw returns ranked by beta
    assert abs(f["HIGH"]) < 1.0 and abs(f["LOW"]) < 1.0    # residuals near zero once beta is removed
    assert f["MID"] > 1.5 and f["MID"] == f.max()           # only the stock-specific move remains


def test_residual_reversal_removes_industry_moves():
    dates = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(4)
    n = 12
    cols = [f"S{i:02d}" for i in range(n)]
    m = pd.Series(rng.normal(0, 0.008, len(dates)), index=dates)
    ind_f = rng.normal(0, 0.01, (len(dates), 2))
    grp = np.array([0] * 6 + [1] * 6)
    ind_f[-21:, 0] = 0.01                         # industry 0 rallies in the last month, industry 1 does not
    ind_f[-21:, 1] = 0.0
    idio = rng.normal(0, 0.006, (len(dates), n))
    idio[-21:] = 0.0
    r = pd.DataFrame(m.to_numpy()[:, None] + ind_f[:, grp] + idio, index=dates, columns=cols)
    m.iloc[-21:] = 0.0
    r.iloc[-21:] = ind_f[-21:, grp]
    prices = 100 * (1 + r).cumprod()
    industry = pd.DataFrame(np.tile(grp.astype(float), (len(dates), 1)), index=dates, columns=cols)
    f = resid_rev_1m(_ctx(prices, m, industry)).iloc[-1]
    raw = prices.iloc[-1] / prices.iloc[-22] - 1
    assert raw[cols[:6]].min() > 0.15 and raw[cols[6:]].abs().max() < 1e-9
    assert f.abs().max() < 1.0                    # the whole industry move is explained, nothing stock-specific left


def test_residual_reversal_uses_no_future_data():
    dates = pd.bdate_range("2015-01-01", periods=420)
    rng = np.random.default_rng(5)
    m = pd.Series(rng.normal(0, 0.01, len(dates)), index=dates)
    r = pd.DataFrame(m.to_numpy()[:, None] + rng.normal(0, 0.01, (len(dates), 4)), index=dates, columns=list("ABCD"))
    prices = 100 * (1 + r).cumprod()
    full = resid_rev_1m(_ctx(prices, m))
    cut = 380
    part = resid_rev_1m(_ctx(prices.iloc[:cut], m.iloc[:cut]))
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def _turnover_ctx():
    dates = pd.bdate_range("2019-01-01", periods=60)
    cols = ["HOT_UP", "HOT_DN", "COLD_UP", "COLD_DN"]
    daily = pd.Series({"HOT_UP": 0.004, "HOT_DN": -0.004, "COLD_UP": 0.004, "COLD_DN": -0.004})
    r = pd.DataFrame(np.tile(daily.to_numpy(), (len(dates), 1)), index=dates, columns=cols)
    prices = 50 * (1 + r).cumprod()
    shares = pd.Series({"HOT_UP": 1e6, "HOT_DN": 1e6, "COLD_UP": 1e6, "COLD_DN": 1e6})
    turn = pd.Series({"HOT_UP": 0.05, "HOT_DN": 0.04, "COLD_UP": 0.002, "COLD_DN": 0.001})  # daily turnover
    dv = prices * shares * turn
    fund = pd.DataFrame({"ticker": cols, "available": [dates[0] - pd.Timedelta(days=30)] * 4,
                         "shares_out": shares.to_numpy()})
    return _ctx(prices, pd.Series(0.0, index=dates), dollar_volume=dv, fundamentals=fund), dates


def test_turnover_is_dollar_volume_over_market_value():
    ctx, dates = _turnover_ctx()
    t = turnover_1m(ctx).iloc[-1]
    assert np.isclose(t["HOT_UP"], 0.05, rtol=0.06) and np.isclose(t["COLD_DN"], 0.001, rtol=0.06)
    assert t["HOT_UP"] > t["HOT_DN"] > t["COLD_UP"] > t["COLD_DN"]


def test_short_term_momentum_sign():
    ctx, dates = _turnover_ctx()
    s = stmom_1m(ctx).iloc[-1]
    # high-turnover winners and low-turnover losers score highest; high-turnover losers lowest
    assert s["HOT_UP"] > 0 and s["COLD_DN"] > 0
    assert s["HOT_DN"] < 0 and s["COLD_UP"] < 0
    assert s["HOT_UP"] == s.max()


def test_turnover_family_needs_share_counts():
    dates = pd.bdate_range("2019-01-01", periods=60)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    ctx = _ctx(prices, pd.Series(0.0, index=dates))
    out = compute(ctx, ["turnover_1m", "stmom_1m"])
    assert out["turnover_1m"].isna().all().all() and out["stmom_1m"].isna().all().all()
