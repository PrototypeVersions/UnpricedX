"""Evaluation statistics.

Forecast skill is judged on the cross-section *by date*: one information coefficient (rank
correlation between prediction and realized return) per day. Days, not stock-days, are the
unit of evidence, because stocks on the same day share the same shocks.

Overlapping horizons make consecutive daily ICs dependent, so t-statistics use Newey-West
standard errors with lag = horizon. Portfolio statistics use non-overlapping rebalances.
The deflated Sharpe ratio (Bailey & Lopez de Prado, 2014) corrects for the number of
experiments ever tried, read from the append-only ledger.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_GAMMA = 0.5772156649


def daily_ic(df: pd.DataFrame, pred: str, target: str, min_names: int = 30) -> pd.Series:
    d = df[["date", pred, target]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    g = d.groupby("date")
    pr = g[pred].rank()
    tr = g[target].rank()
    pr = pr - pr.groupby(d["date"]).transform("mean")
    tr = tr - tr.groupby(d["date"]).transform("mean")
    num = (pr * tr).groupby(d["date"]).sum()
    den = np.sqrt((pr * pr).groupby(d["date"]).sum() * (tr * tr).groupby(d["date"]).sum())
    n = g.size()
    ic = (num / den.replace(0, np.nan)).where(n >= min_names)
    return ic.dropna()


def newey_west_t(x: pd.Series | np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(max(s, 1e-18) / n))


def ic_summary(ic: pd.Series, h: int) -> dict:
    if ic.empty:
        return {"days": 0}
    yearly = ic.groupby(ic.index.year).mean()
    return {
        "days": int(len(ic)),
        "mean": float(ic.mean()),
        "std": float(ic.std()),
        "tNW": newey_west_t(ic.values, lags=max(1, h)),
        "hitRate": float((ic > 0).mean()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan"),
        "yearsPositive": float((yearly > 0).mean()),
        "byYear": {int(y): round(float(v), 5) for y, v in yearly.items()},
        "worstYear": float(yearly.min()),
    }


def _rebalance_dates(dates: np.ndarray, h: int) -> np.ndarray:
    u = np.sort(np.unique(dates))
    return u[::h]


def long_short(df: pd.DataFrame, pred: str, h: int, cost_bps: pd.Series | None = None, frac: float = 0.1) -> pd.DataFrame:
    """Equal-weight top-minus-bottom fraction, rebalanced every h sessions, net of trading costs.

    Returns one row per holding period: gross, cost, net, turnover.
    """
    d = df[["date", "ticker", pred, f"fwd_{h}", "liq_rank"]].dropna(subset=[pred, f"fwd_{h}"])
    reb = _rebalance_dates(d["date"].to_numpy(), h)
    rows, prev_long, prev_short = [], {}, {}
    for date, day in d[d["date"].isin(reb)].groupby("date"):
        if len(day) < 50:
            continue
        k = max(5, int(len(day) * frac))
        srt = day.sort_values(pred)
        short, long_ = srt.head(k), srt.tail(k)
        wl = dict.fromkeys(long_["ticker"], 1.0 / k)
        ws = dict.fromkeys(short["ticker"], 1.0 / k)
        traded_l = {t: abs(wl.get(t, 0) - prev_long.get(t, 0)) for t in set(wl) | set(prev_long)}
        traded_s = {t: abs(ws.get(t, 0) - prev_short.get(t, 0)) for t in set(ws) | set(prev_short)}
        costs = _cost_lookup(day, cost_bps)
        cost = sum(w * costs.get(t, 30.0) for t, w in traded_l.items()) + sum(w * costs.get(t, 30.0) for t, w in traded_s.items())
        gross = float(long_[f"fwd_{h}"].mean() - short[f"fwd_{h}"].mean())
        rows.append({"date": date, "gross": gross, "cost": cost / 1e4, "net": gross - cost / 1e4,
                     "turnover": (sum(traded_l.values()) + sum(traded_s.values())) / 2,
                     "long": float(long_[f"fwd_{h}"].mean()), "short": float(short[f"fwd_{h}"].mean())})
        prev_long, prev_short = wl, ws
    return pd.DataFrame(rows)


def _cost_lookup(day: pd.DataFrame, cost_bps) -> dict:
    if cost_bps is None:
        from ..data.universe import cost_bps as tiers
        return dict(zip(day["ticker"], tiers(day["liq_rank"]).to_numpy()))
    return dict(zip(day["ticker"], cost_bps.reindex(day["ticker"]).fillna(30.0).to_numpy()))


def long_only(df: pd.DataFrame, pred: str, h: int, n: int = 20) -> pd.DataFrame:
    """Top-n tradable names, equal weight, every h sessions; excess return over the benchmark."""
    d = df[df["tradable"].astype(bool)][["date", "ticker", pred, f"fwd_{h}", f"bench_{h}", "liq_rank"]].dropna(subset=[pred, f"fwd_{h}"])
    reb = _rebalance_dates(d["date"].to_numpy(), h)
    rows, prev = [], {}
    for date, day in d[d["date"].isin(reb)].groupby("date"):
        if len(day) < n * 3:
            continue
        top = day.nlargest(n, pred)
        w = dict.fromkeys(top["ticker"], 1.0 / n)
        traded = sum(abs(w.get(t, 0) - prev.get(t, 0)) for t in set(w) | set(prev))
        costs = _cost_lookup(day, None)
        cost = sum(abs(w.get(t, 0) - prev.get(t, 0)) * costs.get(t, 30.0) for t in set(w) | set(prev)) / 1e4
        ret = float(top[f"fwd_{h}"].mean())
        bench = float(top[f"bench_{h}"].iloc[0])
        rows.append({"date": date, "gross": ret, "net": ret - cost, "bench": bench, "excess": ret - cost - bench,
                     "turnover": traded / 2})
        prev = w
    return pd.DataFrame(rows)


def sharpe(r: pd.Series, periods_per_year: float) -> float:
    r = pd.Series(r).dropna()
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(periods_per_year))


def max_drawdown(r: pd.Series) -> float:
    w = (1 + pd.Series(r).fillna(0)).cumprod()
    return float((w / w.cummax() - 1).min()) if len(w) else float("nan")


def deflated_sharpe(r: pd.Series, n_trials: int, sr_var_trials: float | None = None) -> dict:
    """Probability that the true (per-period) Sharpe ratio exceeds what the best of `n_trials`
    unskilled strategies would show by luck."""
    r = pd.Series(r).dropna()
    T = len(r)
    if T < 10 or r.std() == 0:
        return {"dsr": float("nan"), "psr": float("nan"), "sr0": float("nan")}
    sr = r.mean() / r.std()
    g3 = float(((r - r.mean()) ** 3).mean() / r.std(ddof=0) ** 3)
    g4 = float(((r - r.mean()) ** 4).mean() / r.std(ddof=0) ** 4)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    theo_var = (1 + 0.5 * sr**2) / T
    v = max(sr_var_trials or 0.0, theo_var)
    n = max(1, int(n_trials))
    sr0 = 0.0 if n == 1 else math.sqrt(v) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * math.e)))
    psr = float(norm.cdf(sr * math.sqrt(T - 1) / denom))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))
    return {"dsr": dsr, "psr": psr, "sr0": float(sr0), "srPerPeriod": float(sr), "periods": T, "nTrials": n}


def summarize(oos: pd.DataFrame, h: int, n_trials: int = 1, sr_var_trials: float | None = None) -> dict:
    """Full evaluation block for one model over one era."""
    ic = daily_ic(oos, "pred", f"fwd_{h}")
    ls = long_short(oos, "pred", h)
    lo = long_only(oos, "pred", h)
    ppy = 252 / h
    out = {"ic": ic_summary(ic, h)}
    if len(ls):
        out["longShort"] = {
            "periods": int(len(ls)), "grossAnnual": float(ls["gross"].mean() * ppy), "netAnnual": float(ls["net"].mean() * ppy),
            "sharpeGross": sharpe(ls["gross"], ppy), "sharpeNet": sharpe(ls["net"], ppy), "turnover": float(ls["turnover"].mean()),
            "maxDrawdownNet": max_drawdown(ls["net"]), "deflated": deflated_sharpe(ls["net"], n_trials, sr_var_trials),
            "longLeg": float(ls["long"].mean() * ppy), "shortLeg": float(ls["short"].mean() * ppy),
        }
    if len(lo):
        out["longOnly"] = {
            "periods": int(len(lo)), "netAnnual": float(lo["net"].mean() * ppy), "benchAnnual": float(lo["bench"].mean() * ppy),
            "excessAnnual": float(lo["excess"].mean() * ppy), "sharpeExcess": sharpe(lo["excess"], ppy),
            "turnover": float(lo["turnover"].mean()),
        }
    out["_series"] = {"ic": ic, "longShort": ls, "longOnly": lo}
    return out
