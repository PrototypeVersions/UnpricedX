"""Synthetic market with planted signals of known strength.

Used for offline tests, for checking that the pipeline can recover real effects, and for
power analysis ("how many live weeks would it take to see an edge this size?"). Every data
table the real pipeline uses (prices, SEC filings and metadata, fundamentals) is generated with
the same schema, so the full system can run end-to-end without network access.

Planted effects (each scaled by `strength`):
  * momentum   - slowly varying per-stock drift (AR(1)), recoverable by 6-12 month momentum
  * reversal   - part of each day's idiosyncratic shock reverses over the next week
  * PEAD       - earnings jumps are followed by partial drift in the same direction
  * quality    - a persistent per-stock drift that shows up in gross profitability
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from ..calendar import sessions
from .prices import COLS

INDUSTRIES = ["28", "35", "36", "38", "48", "49", "50", "58", "60", "63", "73", "80"]


@dataclass
class SyntheticMarket:
    n: int = 300
    start: str = "2004-01-01"
    end: str = "2026-10-30"
    seed: int = 7
    strength: float = 1.0
    tables: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._generate()

    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        dates = sessions(self.start, self.end)
        T, N, k = len(dates), self.n, self.strength
        tickers = [f"S{i:03d}" for i in range(N)]

        vol_state = np.empty(T)
        vol_state[0] = 0.010
        z = rng.standard_normal(T)
        for t in range(1, T):
            vol_state[t] = np.sqrt(0.000002 + 0.08 * (vol_state[t - 1] * z[t - 1]) ** 2 + 0.90 * vol_state[t - 1] ** 2)
        mkt = 0.0003 + vol_state * z

        beta = rng.uniform(0.6, 1.6, N)
        sig = rng.uniform(0.012, 0.030, N)
        quality = rng.standard_normal(N)
        eps = rng.standard_normal((T, N)) * sig

        mu = np.zeros((T, N))                      # momentum: persistent drift
        rho, s_mu = 0.995, 0.00045 * k
        shocks = rng.standard_normal((T, N)) * s_mu * np.sqrt(1 - rho**2)
        for t in range(1, T):
            mu[t] = rho * mu[t - 1] + shocks[t]

        rev = np.zeros((T, N))                     # reversal of recent idiosyncratic shocks
        lam = 0.06 * k
        for lag in range(1, 6):
            rev[lag:] -= lam / 5 * eps[:-lag]

        earn = np.zeros((T, N))                    # earnings jumps + post-announcement drift
        offsets = rng.integers(0, 63, N)
        events = []
        for i in range(N):
            for e in range(int(offsets[i]) + 40, T - 1, 63):
                jump = rng.normal(0, 0.05)
                earn[e, i] += jump
                drift_days = min(20, T - e - 1)
                earn[e + 1 : e + 1 + drift_days, i] += 0.30 * k * jump / 20
                events.append((i, e))
        qual = 0.00015 * k * quality

        ret = beta * mkt[:, None] + mu + rev + earn + qual + eps
        ret = np.clip(ret, -0.6, 1.5)

        # Listing windows: some IPOs and some delistings to exercise missing data.
        first = np.zeros(N, dtype=int)
        last = np.full(N, T - 1)
        late = rng.random(N) < 0.12
        first[late] = rng.integers(300, T // 2, late.sum())
        gone = rng.random(N) < 0.05
        last[gone] = rng.integers(T // 2, T - 300, gone.sum())

        p0 = rng.uniform(15, 150, N)
        adj = p0 * np.exp(np.cumsum(np.log1p(ret), axis=0))
        overnight = 0.35 * ret + rng.normal(0, 0.003, (T, N))
        open_ = np.vstack([adj[:1], adj[:-1]]) * (1 + overnight)
        rngs = np.abs(rng.normal(0, 0.008, (T, N)))
        high = np.maximum(open_, adj) * (1 + rngs)
        low = np.minimum(open_, adj) * (1 - rngs)
        size = rng.lognormal(16.5, 1.3, N)
        volume = size[None, :] / adj * rng.lognormal(0, 0.35, (T, N))

        live = (np.arange(T)[:, None] >= first[None, :]) & (np.arange(T)[:, None] <= last[None, :])
        rows = []
        for i, t in enumerate(tickers):
            idx = np.where(live[:, i])[0]
            rows.append(pd.DataFrame({
                "date": dates[idx], "ticker": t, "open": open_[idx, i], "high": high[idx, i], "low": low[idx, i],
                "close": adj[idx, i], "adj_close": adj[idx, i], "volume": volume[idx, i], "split": 0.0,
            }))
        for sym, b, noise in [("SPY", 1.0, 0.0005), ("QQQ", 1.2, 0.003), ("IWM", 1.1, 0.004)]:
            r = b * mkt + rng.normal(0, noise, T)
            px = 100 * np.exp(np.cumsum(np.log1p(r)))
            o = np.concatenate([[px[0]], px[:-1]]) * (1 + 0.35 * r)
            rows.append(pd.DataFrame({"date": dates, "ticker": sym, "open": o, "high": np.maximum(o, px) * 1.002,
                                      "low": np.minimum(o, px) * 0.998, "close": px, "adj_close": px,
                                      "volume": 5e7, "split": 0.0}))
        vix = 100 * vol_state * np.sqrt(252) * 1.1
        rows.append(pd.DataFrame({"date": dates, "ticker": "^VIX", "open": vix, "high": vix, "low": vix, "close": vix,
                                  "adj_close": vix, "volume": 0.0, "split": 0.0}))
        prices = pd.concat(rows, ignore_index=True)[COLS]

        ciks = 100000 + np.arange(N)
        ind = rng.choice(INDUSTRIES, N)
        tick_table = pd.DataFrame({"cik": ciks, "name": [f"Synthetic Co {i}" for i in range(N)], "ticker": tickers,
                                   "exchange": rng.choice(["NYSE", "Nasdaq"], N)})
        meta = pd.DataFrame({"cik": ciks, "name": tick_table["name"], "sic": [int(x) * 100 + 1 for x in ind],
                             "sicDescription": "synthetic", "entityType": "operating"})

        # 8-K item 2.02 filed after the close the session *before* each jump.
        f_rows = []
        for i, e in events:
            if e - 1 < first[i] or e > last[i]:
                continue
            d = dates[e - 1]
            acc = f"0000{ciks[i]}-{d:%y}-{e:06d}"
            f_rows.append((int(ciks[i]), acc, "8-K", d.date().isoformat(), d.date().isoformat(),
                           f"{d:%Y-%m-%d}T16:05:00.000Z", "2.02,9.01", "earnings.htm"))
        filings = pd.DataFrame(f_rows, columns=["cik", "accession", "form", "filing_date", "report_date", "acceptance", "items", "primary_doc"])

        # Quarterly fundamentals whose gross profitability tracks the planted quality drift.
        q_rows = []
        q_ends = pd.date_range(dates[0], dates[-1], freq="QE")
        for i, t in enumerate(tickers):
            assets = size[i] * rng.uniform(3, 8)
            shares = assets / p0[i] * 0.5
            for j, qe in enumerate(q_ends):
                avail = qe + pd.Timedelta(days=int(rng.integers(25, 45)))
                if avail > dates[-1]:
                    continue
                assets *= np.exp(rng.normal(0.015, 0.03))
                q_rows.append({
                    "ticker": t, "end": qe, "available": avail, "assets": assets, "equity": assets * 0.45,
                    "shares_out": shares, "revenue_ttm": assets * 0.6, "net_income_ttm": assets * 0.05 * (1 + 0.3 * rng.standard_normal()),
                    "gp_assets": 0.30 + 0.08 * quality[i] + rng.normal(0, 0.03), "asset_growth": rng.normal(0.06, 0.05),
                    "sales_growth": rng.normal(0.06, 0.08), "accruals": rng.normal(0, 0.03), "roe": rng.normal(0.11, 0.05),
                    "issuance": rng.normal(0.01, 0.02), "sue": rng.normal(0, 1),
                })
        fundamentals = pd.DataFrame(q_rows)

        self.tables = {"prices": prices, "sec_tickers": tick_table, "sec_meta": meta, "sec_filings": filings,
                       "fundamentals": fundamentals}

    def write_store(self) -> None:
        from .. import store
        for name, df in self.tables.items():
            store.write(name, df)


@lru_cache(maxsize=4)
def market(n: int = 300, seed: int = 7, strength: float = 1.0, start: str = "2004-01-01", end: str = "2026-10-30") -> SyntheticMarket:
    return SyntheticMarket(n=n, seed=seed, strength=strength, start=start, end=end)


class SyntheticProvider:
    name = "synthetic"

    def __init__(self, m: SyntheticMarket | None = None) -> None:
        self.m = m or market()

    def fetch(self, tickers: list[str], start: str, end: str | None = None) -> tuple[pd.DataFrame, list[str]]:
        p = self.m.tables["prices"]
        out = p[p["ticker"].isin(tickers) & (p["date"] >= pd.Timestamp(start))]
        if end:
            out = out[out["date"] < pd.Timestamp(end)]
        failed = sorted(set(tickers) - set(out["ticker"].unique()))
        return out.copy(), failed
