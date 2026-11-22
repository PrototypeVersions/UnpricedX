"""Feature library. Each feature is a backward-looking function of the Context.

`sign` is the direction documented in the literature (+1: higher value -> higher future return).
It is used only by single-factor baseline models, which are pre-registered benchmarks; the
learned models ignore it.

To add a feature: write a function, register it with @feature, add a test. New features are
evaluated by the lab like any other experiment and count toward the trial total.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from .context import Context


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    fn: Callable[[Context], pd.DataFrame]
    sign: int
    doc: str
    ref: str = ""


REGISTRY: dict[str, Feature] = {}


def feature(name: str, family: str, sign: int, doc: str, ref: str = ""):
    def wrap(fn):
        REGISTRY[name] = Feature(name, family, fn, sign, doc, ref)
        return fn
    return wrap


def families() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in REGISTRY.values():
        out.setdefault(f.family, []).append(f.name)
    return out


# ---------------------------------------------------------------- helpers
def _px(ctx: Context) -> pd.DataFrame:
    return ctx.mats["adj_close"][ctx.tickers]


def _group_mean(mat: pd.DataFrame, groups: pd.DataFrame, mask: pd.DataFrame, min_members: int = 5) -> pd.DataFrame:
    """Per-date mean of `mat` within each group (members only), broadcast back to tickers."""
    x = mat.where(mask & groups.notna())
    g = groups.reindex_like(x)
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    values = x.to_numpy(dtype="float64")
    gv = g.to_numpy()
    res = np.full_like(values, np.nan)
    ids = np.unique(gv[~np.isnan(gv)]) if np.isfinite(gv).any() else []
    for gid in ids:
        m = gv == gid
        v = np.where(m, values, np.nan)
        cnt = np.sum(~np.isnan(v), axis=1)
        mean = np.where(cnt >= min_members, np.nansum(v, axis=1) / np.maximum(cnt, 1), np.nan)
        res = np.where(m, mean[:, None], res)
    out[:] = res
    return out


def _ret(ctx: Context) -> pd.DataFrame:
    return ctx.returns


# ---------------------------------------------------------------- price: momentum / reversal
@feature("mom_12_1", "momentum", +1, "Return from 12 months to 1 month ago.", "Jegadeesh & Titman (1993)")
def mom_12_1(ctx):
    p = _px(ctx)
    return p.shift(21) / p.shift(252) - 1


@feature("mom_6_1", "momentum", +1, "Return from 6 months to 1 month ago.", "Jegadeesh & Titman (1993)")
def mom_6_1(ctx):
    p = _px(ctx)
    return p.shift(21) / p.shift(126) - 1


@feature("high_52w", "momentum", +1, "Price relative to its 52-week high.", "George & Hwang (2004)")
def high_52w(ctx):
    p = _px(ctx)
    return p / p.rolling(252, min_periods=200).max()


@feature("fip", "momentum", +1, "Frog-in-the-pan: momentum built from many small moves rather than a few jumps.",
         "Da, Gurun & Warachka (2014)")
def fip(ctx):
    r = _ret(ctx).shift(21)
    pos = (r > 0).astype(float).rolling(231, min_periods=180).mean()
    neg = (r < 0).astype(float).rolling(231, min_periods=180).mean()
    return -np.sign(mom_12_1(ctx)) * (neg - pos)


@feature("ind_mom_6", "momentum", +1, "Six-month momentum of the stock's statistical industry.", "Moskowitz & Grinblatt (1999)")
def ind_mom_6(ctx):
    p = _px(ctx)
    m = p / p.shift(126) - 1
    return _group_mean(m, ctx.industry, ctx.member["research"])


@feature("ret_1m", "reversal", -1, "Return over the last month (short-term reversal).", "Jegadeesh (1990); Lehmann (1990)")
def ret_1m(ctx):
    p = _px(ctx)
    return p / p.shift(21) - 1


@feature("ret_1w", "reversal", -1, "Return over the last week.", "Lehmann (1990)")
def ret_1w(ctx):
    p = _px(ctx)
    return p / p.shift(5) - 1


@feature("ind_rel_1m", "reversal", -1, "Last-month return minus its industry's (industry-adjusted reversal).",
         "Da, Liu & Schaumburg (2014)")
def ind_rel_1m(ctx):
    m = ret_1m(ctx)
    return m - _group_mean(m, ctx.industry, ctx.member["research"])


@feature("overnight_1m", "reversal", +1, "Sum of overnight (close-to-open) returns over the last month.",
         "Lou, Polk & Skouras (2019)")
def overnight_1m(ctx):
    o = ctx.mats["adj_open"][ctx.tickers] / _px(ctx).shift(1)
    return np.log(o).rolling(21, min_periods=15).sum()


@feature("intraday_1m", "reversal", -1, "Sum of intraday (open-to-close) returns over the last month.",
         "Lou, Polk & Skouras (2019)")
def intraday_1m(ctx):
    i = _px(ctx) / ctx.mats["adj_open"][ctx.tickers]
    return np.log(i).rolling(21, min_periods=15).sum()


# ---------------------------------------------------------------- price: risk / lottery / liquidity
@feature("vol_3m", "risk", -1, "Realized daily volatility over three months (low-volatility effect).", "Ang et al. (2006)")
def vol_3m(ctx):
    return _ret(ctx).rolling(63, min_periods=50).std()


def _beta_ivol(ctx: Context, win: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"beta_ivol_{win}"
    if key not in ctx.extras:
        r = _ret(ctx)
        m = ctx.market_ret.reindex(r.index)
        mp = int(win * 0.8)
        mean_r = r.rolling(win, min_periods=mp).mean()
        mean_m = m.rolling(win, min_periods=mp).mean()
        cov = r.mul(m, axis=0).rolling(win, min_periods=mp).mean() - mean_r.mul(mean_m, axis=0)
        var = (m * m).rolling(win, min_periods=mp).mean() - mean_m ** 2
        beta = cov.div(var, axis=0)
        resid = r - beta.shift(1).mul(m, axis=0)
        ivol = resid.rolling(63, min_periods=50).std()
        ctx.extras[key] = (beta, ivol)
    return ctx.extras[key]


@feature("beta_1y", "risk", -1, "Market beta over one year (betting-against-beta).", "Frazzini & Pedersen (2014)")
def beta_1y(ctx):
    return _beta_ivol(ctx, 252)[0]


@feature("ivol_3m", "risk", -1, "Idiosyncratic volatility relative to the market.", "Ang et al. (2006)")
def ivol_3m(ctx):
    return _beta_ivol(ctx, 252)[1]


@feature("max_1m", "risk", -1, "Largest daily return in the last month (lottery demand).", "Bali, Cakici & Whitelaw (2011)")
def max_1m(ctx):
    return _ret(ctx).rolling(21, min_periods=15).max()


@feature("skew_3m", "risk", -1, "Skewness of daily returns over three months.", "Boyer, Mitton & Vorkink (2010)")
def skew_3m(ctx):
    return _ret(ctx).rolling(63, min_periods=50).skew()


@feature("log_dvol", "liquidity", -1, "Log of average daily dollar volume over a month (size/liquidity).", "Amihud (2002)")
def log_dvol(ctx):
    return np.log(ctx.mats["dollar_volume"][ctx.tickers].rolling(21, min_periods=15).mean().replace(0, np.nan))


@feature("amihud_1m", "liquidity", +1, "Amihud illiquidity: price impact per dollar traded.", "Amihud (2002)")
def amihud_1m(ctx):
    dv = ctx.mats["dollar_volume"][ctx.tickers].replace(0, np.nan)
    return np.log((_ret(ctx).abs() / dv).rolling(21, min_periods=15).mean() * 1e9 + 1e-9)


@feature("volume_trend", "liquidity", -1, "Recent month's volume relative to the past year (attention).",
         "Gervais, Kaniel & Mingelgrin (2001)")
def volume_trend(ctx):
    v = ctx.mats["volume"][ctx.tickers].replace(0, np.nan)
    return np.log(v.rolling(21, min_periods=15).mean() / v.rolling(252, min_periods=200).mean())


# ---------------------------------------------------------------- events (SEC 8-K)
def _event_matrix(ctx: Context) -> pd.DataFrame:
    key = "event_matrix"
    if key not in ctx.extras:
        arr = np.zeros((len(ctx.dates), len(ctx.tickers)))
        ev = ctx.events[ctx.events["ticker"].isin(set(ctx.tickers))]
        if len(ev):
            rows = ctx.dates.get_indexer(pd.to_datetime(ev["session"]))
            cols = ctx.tickers.get_indexer(ev["ticker"])
            ok = (rows >= 0) & (cols >= 0)
            arr[rows[ok], cols[ok]] = 1.0
        ctx.extras[key] = pd.DataFrame(arr, index=ctx.dates, columns=ctx.tickers)
    return ctx.extras[key]


def _has_events(ctx: Context) -> bool:
    return len(ctx.events) > 0


def _sessions_since(ind: pd.DataFrame) -> pd.DataFrame:
    idx = np.arange(len(ind))[:, None] * np.ones((1, ind.shape[1]))
    last = pd.DataFrame(np.where(ind.to_numpy() > 0, idx, np.nan), index=ind.index, columns=ind.columns).ffill()
    return pd.DataFrame(idx, index=ind.index, columns=ind.columns) - last


@feature("ear", "events", +1, "Market-adjusted return over the 3 days around the latest earnings release (if within 63 sessions).",
         "Brandt, Kishore, Santa-Clara & Venkatachalam (2008)")
def ear(ctx):
    ev = _event_matrix(ctx)
    if not _has_events(ctx):
        return ev * np.nan
    ab = _ret(ctx).sub(ctx.market_ret.reindex(ctx.dates), axis=0)
    car = ab.shift(1) + ab + ab.shift(-1)                 # window [s-1, s+1] around reaction session s
    known = (car.where(ev > 0)).shift(1)                  # only known after the close of s+1
    return known.ffill(limit=62)


@feature("days_since_earn", "events", 0, "Sessions since the latest earnings release (capped).", "Frazzini & Lamont (2007)")
def days_since_earn(ctx):
    if not _has_events(ctx):
        return _event_matrix(ctx) * np.nan
    return _sessions_since(_event_matrix(ctx)).clip(upper=130)


@feature("earn_soon", "events", +1, "Next earnings release is likely within the next ~2 weeks (announcement premium).",
         "Barber, De George, Lehavy & Trueman (2013)")
def earn_soon(ctx):
    if not _has_events(ctx):
        return _event_matrix(ctx) * np.nan
    d = _sessions_since(_event_matrix(ctx))
    return ((d >= 52) & (d <= 66)).astype(float).where(d.notna())


@feature("filing_rate_3m", "events", 0, "Number of SEC filings with market-moving forms in the last 63 sessions.", "")
def filing_rate_3m(ctx):
    if ctx.filings.empty:
        return _event_matrix(ctx) * np.nan
    f = ctx.filings[ctx.filings["form"].isin(["8-K", "10-Q", "10-K"]) & ctx.filings["ticker"].isin(set(ctx.tickers))]
    m = pd.DataFrame(0.0, index=ctx.dates, columns=ctx.tickers)
    if len(f):
        counts = f.groupby(["session", "ticker"]).size().unstack(fill_value=0)
        counts.index = pd.to_datetime(counts.index)
        m = m.add(counts.reindex(index=m.index, columns=m.columns, fill_value=0), fill_value=0)
    return m.rolling(63, min_periods=1).sum()


# ---------------------------------------------------------------- fundamentals (SEC XBRL)
FUND_FIELDS = ["gp_assets", "asset_growth", "sales_growth", "accruals", "roe", "issuance", "sue",
               "net_income_ttm", "equity", "revenue_ttm", "shares_out"]


def _fund_matrix(ctx: Context, col: str) -> pd.DataFrame:
    key = f"fund_{col}"
    if key in ctx.extras:
        return ctx.extras[key]
    f = ctx.fundamentals
    out = pd.DataFrame(np.nan, index=ctx.dates, columns=ctx.tickers)
    if f is None or f.empty or col not in f:
        ctx.extras[key] = out
        return out
    lag = 1
    sub = f[f["ticker"].isin(set(ctx.tickers))][["ticker", "available", col]].dropna()
    if len(sub):
        # effective from the first session strictly after the filing date, plus `lag` sessions
        pos = ctx.dates.searchsorted(pd.to_datetime(sub["available"]).values, side="right") + (lag - 1)
        ok = pos < len(ctx.dates)
        sub = sub[ok].assign(session=ctx.dates[pos[ok]])
        piv = sub.sort_values("available").drop_duplicates(["session", "ticker"], keep="last").pivot(
            index="session", columns="ticker", values=col)
        out = piv.reindex(index=ctx.dates, columns=ctx.tickers).ffill(limit=320)
    ctx.extras[key] = out
    return out


def _mcap(ctx: Context) -> pd.DataFrame:
    shares = _fund_matrix(ctx, "shares_out")
    return ctx.mats["raw_close"][ctx.tickers] * shares


def _fund_feature(col: str):
    return lambda ctx: _fund_matrix(ctx, col)


for _name, _col, _sign, _doc, _ref in [
    ("gp_assets", "gp_assets", +1, "Gross profit over total assets (profitability).", "Novy-Marx (2013)"),
    ("asset_growth", "asset_growth", -1, "Year-over-year growth in total assets (investment).", "Cooper, Gulen & Schill (2008)"),
    ("sales_growth", "sales_growth", -1, "Year-over-year growth in trailing revenue.", "Lakonishok, Shleifer & Vishny (1994)"),
    ("accruals", "accruals", -1, "Net income minus operating cash flow, scaled by assets.", "Sloan (1996)"),
    ("roe", "roe", +1, "Trailing return on equity.", "Haugen & Baker (1996)"),
    ("issuance", "issuance", -1, "Year-over-year change in shares outstanding (net issuance).", "Pontiff & Woodgate (2008)"),
    ("sue", "sue", +1, "Standardized unexpected earnings (seasonal random walk).", "Bernard & Thomas (1989)"),
]:
    REGISTRY[_name] = Feature(_name, "fundamentals", _fund_feature(_col), _sign, _doc, _ref)


@feature("earnings_yield", "valuation", +1, "Trailing net income over market capitalization.", "Basu (1977)")
def earnings_yield(ctx):
    return _fund_matrix(ctx, "net_income_ttm") / _mcap(ctx)


@feature("book_to_market", "valuation", +1, "Book equity over market capitalization.", "Fama & French (1992)")
def book_to_market(ctx):
    eq = _fund_matrix(ctx, "equity")
    return eq.where(eq > 0) / _mcap(ctx)


@feature("sales_to_price", "valuation", +1, "Trailing revenue over market capitalization.", "Barbee, Mukherji & Raines (1996)")
def sales_to_price(ctx):
    return _fund_matrix(ctx, "revenue_ttm") / _mcap(ctx)


@feature("log_mcap", "valuation", -1, "Log market capitalization (size).", "Banz (1981)")
def log_mcap(ctx):
    return np.log(_mcap(ctx).where(lambda x: x > 0))


# ---------------------------------------------------------------- news vs. no-news returns (SEC filings)
# Own family ("news"), so that adding it does not change the feature set of any existing model.
NEWS_FORMS = ["8-K", "8-K/A", "10-Q", "10-K"]


def _news_days(ctx: Context) -> pd.DataFrame | None:
    """1 on a stock's filing reaction session and the session after it (delayed reactions), else 0.
    NaN for stocks with no filing in the past 252 sessions (no evidence the stock is covered)."""
    key = "news_days"
    if key in ctx.extras:
        return ctx.extras[key]
    out = None
    if not ctx.filings.empty:
        f = ctx.filings[ctx.filings["form"].isin(NEWS_FORMS) & ctx.filings["ticker"].isin(set(ctx.tickers))]
        arr = np.zeros((len(ctx.dates), len(ctx.tickers)))
        if len(f):
            rows = ctx.dates.get_indexer(pd.to_datetime(f["session"]))
            cols = ctx.tickers.get_indexer(f["ticker"])
            ok = (rows >= 0) & (cols >= 0)
            arr[rows[ok], cols[ok]] = 1.0
        ind = pd.DataFrame(arr, index=ctx.dates, columns=ctx.tickers)
        covered = ind.rolling(252, min_periods=1).sum() > 0
        out = ((ind + ind.shift(1).fillna(0)) > 0).astype(float).where(covered)
    ctx.extras[key] = out
    return out


@feature("ret_1w_nonews", "news", -1,
         "Return over the last week counting only sessions without an SEC filing (no-news reversal).",
         "Chan (2003); Jiang, Li & Wang (2021)")
def ret_1w_nonews(ctx):
    nd = _news_days(ctx)
    if nd is None:
        return _px(ctx) * np.nan
    lr = np.log1p(_ret(ctx))
    quiet = lr.where(nd == 0, 0.0).where(lr.notna() & nd.notna())
    return quiet.rolling(5, min_periods=4).sum()


@feature("news_ret_1m", "news", +1,
         "Market-adjusted return on filing sessions over the last month (drift after news).",
         "Chan (2003); Jiang, Li & Wang (2021)")
def news_ret_1m(ctx):
    nd = _news_days(ctx)
    if nd is None:
        return _px(ctx) * np.nan
    ab = np.log1p(_ret(ctx)).sub(np.log1p(ctx.market_ret.reindex(ctx.dates)), axis=0)
    loud = ab.where(nd == 1, 0.0).where(ab.notna() & nd.notna())
    return loud.rolling(21, min_periods=15).sum()


# ---------------------------------------------------------------- return seasonality (prices)
@feature("season_amn", "seasonal", +1,
         "Average return in the coming calendar month over past years minus the average in other months "
         "(years 1-10, cross-sectionally demeaned).",
         "Heston & Sadka (2008); Keloharju, Linnainmaa & Nyberg (2021)")
def season_amn(ctx):
    p = _px(ctx)
    # monthly returns of completed months, demeaned across stocks each month
    mret = p.resample("ME").last().pct_change(fill_method=None)
    inuni = ctx.member["research"].reindex(columns=mret.columns).astype("float64").resample("ME").last() > 0
    mret = mret.sub(mret.where(inuni).mean(axis=1), axis=0)     # point-in-time universe, so no look-ahead
    mv = mret.to_numpy(dtype="float64")
    months = mret.index.to_period("M")
    n = len(months)
    amn = np.full((n + 1, mv.shape[1]), np.nan)
    for j in range(n + 1):                       # j = index of the target month (n = month after the data)
        ann = [j - 12 * k for k in range(1, 11) if j - 12 * k >= 1]
        if len(ann) < 3:
            continue
        other = [i for i in range(max(1, j - 120), j - 12) if (j - i) % 12 != 0]
        a = mv[ann]
        o = mv[other]
        na, no = np.sum(~np.isnan(a), axis=0), np.sum(~np.isnan(o), axis=0)
        if not len(other):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            val = np.where((na >= 3) & (no >= 24), np.nanmean(a, axis=0) - np.nanmean(o, axis=0), np.nan)
        amn[j] = val
    target = (ctx.dates + pd.Timedelta(days=4)).to_period("M")   # month in which most of the 5-day holding falls
    first = months[0]
    pos = np.array([(t - first).n for t in target])
    out = np.full((len(ctx.dates), mv.shape[1]), np.nan)
    ok = (pos >= 0) & (pos < amn.shape[0])
    out[ok] = amn[pos[ok]]
    return pd.DataFrame(out, index=ctx.dates, columns=mret.columns)


# ---------------------------------------------------------------- residual reversal (prices; research 2026-11-15)
# Own family ("residual") so that no existing model changes its features at the next refit.
def _resid_moments(ctx: Context, win: int = 252) -> dict[str, pd.DataFrame] | None:
    """Rolling two-factor regression of each stock's daily return on the market and on its statistical
    industry's return in excess of the market. Uses only returns up to and including each date."""
    key = f"resid_moments_{win}"
    if key in ctx.extras:
        return ctx.extras[key]
    r = _ret(ctx)
    m = ctx.market_ret.reindex(r.index).astype("float64")
    ind = _group_mean(r, ctx.industry, ctx.member["research"])
    x = ind.sub(m, axis=0)
    mp = int(win * 0.8)
    roll = lambda d: d.rolling(win, min_periods=mp).mean()          # noqa: E731
    mr, mm = roll(r), m.rolling(win, min_periods=mp).mean()
    c_rm = roll(r.mul(m, axis=0)) - mr.mul(mm, axis=0)
    c_mm = (m * m).rolling(win, min_periods=mp).mean() - mm ** 2
    v_r = roll(r * r) - mr ** 2
    # market-only fit (used where the stock has no industry yet)
    b1 = c_rm.div(c_mm, axis=0)
    ve1 = v_r - b1 * c_rm
    # two-factor fit on the dates where the industry series is available over the whole window
    xr = x.where(r.notna())
    mx = roll(xr)
    c_rx = roll(r * xr) - mr * mx
    c_xx = roll(xr * xr) - mx ** 2
    c_mx = roll(xr.mul(m, axis=0)) - mx.mul(mm, axis=0)
    det = c_xx.mul(c_mm, axis=0) - c_mx ** 2
    ok = (det > 1e-12) & x.notna()
    bm = ((c_rm * c_xx - c_rx * c_mx) / det).where(ok)
    bx = (c_rx.mul(c_mm, axis=0) - c_rm * c_mx).div(det).where(ok)
    ve2 = v_r - bm * c_rm - bx * c_rx
    out = {"bm": bm.fillna(b1), "bx": bx.fillna(0.0).where(b1.notna()), "var_e": ve2.where(ok).fillna(ve1),
           "x": x.where(ok, 0.0), "m": m}
    ctx.extras[key] = out
    return out


@feature("resid_rev_1m", "residual", -1,
         "Last month's residual return (after market and statistical-industry exposure, betas from the year "
         "before the month) divided by residual volatility: short-term residual reversal.",
         "Blitz, Huij, Lansdorp & Verbeek (2013); Da, Liu & Schaumburg (2014)")
def resid_rev_1m(ctx):
    mo = _resid_moments(ctx)
    r = _ret(ctx)
    n = 21
    rsum = r.rolling(n, min_periods=18).sum()
    msum = mo["m"].rolling(n, min_periods=18).sum()
    xsum = mo["x"].where(r.notna()).rolling(n, min_periods=18).sum()
    bm, bx, ve = mo["bm"].shift(n), mo["bx"].shift(n), mo["var_e"].shift(n)   # estimated before the month
    e = rsum - bm.mul(msum, axis=0) - bx * xsum
    sd = np.sqrt(ve.where(ve > 0) * n)
    return e / sd


# ---------------------------------------------------------------- turnover and short-term momentum (research 2026-11-15)
# Own family ("turnover"); needs SEC share counts, so it is unavailable without SEC data.
def _turnover(ctx: Context, n: int = 21) -> pd.DataFrame:
    key = f"turnover_{n}"
    if key not in ctx.extras:
        dv = ctx.mats["dollar_volume"][ctx.tickers].replace(0, np.nan)   # printed price x printed volume
        mcap = _mcap(ctx)
        ctx.extras[key] = dv.rolling(n, min_periods=15).mean() / mcap.where(mcap > 0)
    return ctx.extras[key]


def _rank_within(mat: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    """Within-date rank among research-universe members, centred symmetrically on zero."""
    mask = ctx.member["research"].reindex_like(mat).fillna(False).astype(bool)
    r = mat.where(mask).rank(axis=1)
    n = r.notna().sum(axis=1)
    return (r - 0.5).div(n.where(n > 0), axis=0) - 0.5


@feature("turnover_1m", "turnover", -1, "Share turnover: last month's average daily dollar volume over market value.",
         "Datar, Naik & Radcliffe (1998); Medhat & Schmeling (2022)")
def turnover_1m(ctx):
    return _turnover(ctx)


@feature("stmom_1m", "turnover", +1,
         "Short-term momentum: last month's return (rank) times share turnover (rank); positive for high-turnover "
         "winners and low-turnover losers.",
         "Medhat & Schmeling (2022)")
def stmom_1m(ctx):
    rr = _rank_within(ret_1m(ctx), ctx)
    tr = _rank_within(_turnover(ctx), ctx)
    return rr * tr


# ---------------------------------------------------------------- tug of war: overnight vs. daytime (prices; research 2026-11-22)
# Own family ("tugofwar") so that no existing model changes its features at the next refit.
def _night_day(ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overnight (previous close to open) and daytime (open to close) log returns, total-return adjusted."""
    key = "night_day"
    if key not in ctx.extras:
        o = ctx.mats["adj_open"][ctx.tickers].where(lambda x: x > 0)
        c = _px(ctx).where(lambda x: x > 0)
        ctx.extras[key] = (np.log(o / c.shift(1)), np.log(c / o))
    return ctx.extras[key]


def _pattern_freq(night: pd.DataFrame, day: pd.DataFrame, pattern: pd.DataFrame, n: int = 21) -> pd.DataFrame:
    valid = (night.notna() & day.notna()).astype(float)
    hits = pattern.astype(float).where(valid > 0, 0.0)
    cnt = valid.rolling(n, min_periods=1).sum()
    return (hits.rolling(n, min_periods=1).sum() / cnt).where(cnt >= 15)


@feature("nr_freq_1m", "tugofwar", +1,
         "Share of the last month's sessions with a positive overnight return followed by a negative daytime "
         "return (a 'negative daytime reversal': the daily tug of war between overnight and daytime clienteles).",
         "Akbas, Boehmer, Jiang & Koch (2022, JFE 145(3):850-875)")
def nr_freq_1m(ctx):
    night, day = _night_day(ctx)
    return _pattern_freq(night, day, (night > 0) & (day < 0))


@feature("pr_freq_1m", "tugofwar", 0,
         "Share of the last month's sessions with a negative overnight return followed by a positive daytime "
         "return (the mirror pattern; used as a control for nr_freq_1m).",
         "Akbas, Boehmer, Jiang & Koch (2022, JFE 145(3):850-875)")
def pr_freq_1m(ctx):
    night, day = _night_day(ctx)
    return _pattern_freq(night, day, (night < 0) & (day > 0))


# ---------------------------------------------------------------- intra-industry lead-lag (prices; research 2026-11-22)
# Own family ("leadlag") so that no existing model changes its features at the next refit.
def _big_in_industry(ctx: Context, frac: float = 1 / 3) -> pd.DataFrame:
    """True for research-universe members in the top `frac` of their statistical industry by trailing
    63-session median dollar volume on each date (uses only volume up to that date)."""
    key = f"big_in_industry_{frac:.3f}"
    if key in ctx.extras:
        return ctx.extras[key]
    dv = ctx.mats["dollar_volume"][ctx.tickers].replace(0, np.nan)
    med = dv.rolling(63, min_periods=50).median()
    mask = ctx.member["research"].reindex_like(med).fillna(False).astype(bool)
    g = ctx.industry.reindex_like(med)
    gv = g.to_numpy(dtype="float64")
    mv = med.where(mask & g.notna()).to_numpy(dtype="float64")
    big = np.zeros_like(mv, dtype=bool)
    ids = np.unique(gv[~np.isnan(gv)]) if np.isfinite(gv).any() else []
    for gid in ids:
        v = np.where(gv == gid, mv, np.nan)
        r = pd.DataFrame(v).rank(axis=1, pct=True, ascending=False).to_numpy()
        big |= (r <= frac) & ~np.isnan(v)
    out = pd.DataFrame(big, index=med.index, columns=med.columns)
    ctx.extras[key] = out
    return out


@feature("ind_lead_1w", "leadlag", +1,
         "Last week's market-adjusted return of the largest third (by dollar volume) of the stock's statistical "
         "industry, excluding the stock itself: slow diffusion of industry news from big to small firms.",
         "Hou (2007, RFS 20(4):1113-1138); Lo & MacKinlay (1990)")
def ind_lead_1w(ctx):
    p = _px(ctx)
    r5 = (p / p.shift(5) - 1)
    m5 = np.expm1(np.log1p(ctx.market_ret.reindex(ctx.dates).astype("float64")).rolling(5, min_periods=5).sum())
    ab = r5.sub(m5, axis=0)
    big = _big_in_industry(ctx)
    g = ctx.industry.reindex_like(ab)
    gv = g.to_numpy(dtype="float64")
    x = ab.where(big).to_numpy(dtype="float64")
    own = np.where(big.to_numpy() & ~np.isnan(x), x, 0.0)
    own_n = (big.to_numpy() & ~np.isnan(x)).astype(float)
    res = np.full_like(x, np.nan)
    ids = np.unique(gv[~np.isnan(gv)]) if np.isfinite(gv).any() else []
    for gid in ids:
        m = gv == gid
        v = np.where(m, x, np.nan)
        cnt = np.sum(~np.isnan(v), axis=1)[:, None]
        tot = np.nansum(v, axis=1)[:, None]
        loo_n = cnt - own_n
        val = np.where(loo_n >= 3, (tot - own) / np.maximum(loo_n, 1), np.nan)
        res = np.where(m, val, res)
    return pd.DataFrame(res, index=ab.index, columns=ab.columns)


def compute(ctx: Context, names: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for n in names:
        mat = REGISTRY[n].fn(ctx)
        mat = mat.replace([np.inf, -np.inf], np.nan)
        out[n] = mat.astype("float32")
    return out
