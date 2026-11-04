"""Everything the feature functions need, assembled from the store as of one information date.

A Context holds wide (date x ticker) matrices. All feature functions must only look backwards
along the date axis; tests/test_leakage.py enforces this by truncating the future and checking
that past feature values do not change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import store
from ..calendar import sessions
from ..config import section
from ..data.industries import statistical_industries
from ..data.prices import price_matrices
from ..data.sec import reaction_session
from ..data.universe import membership
from ..util import log

L = log("context")


@dataclass
class Context:
    mats: dict[str, pd.DataFrame]
    member: dict[str, pd.DataFrame]
    market_ret: pd.Series
    industry: pd.DataFrame
    events: pd.DataFrame            # earnings events: ticker, session
    filings: pd.DataFrame           # all filings with reaction sessions (for intensity features)
    fundamentals: pd.DataFrame
    extras: dict = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.mats["adj_close"].index

    @property
    def tickers(self) -> pd.Index:
        return self.member["research"].columns

    @property
    def returns(self) -> pd.DataFrame:
        if "ret" not in self.extras:
            self.extras["ret"] = self.mats["adj_close"][self.tickers].pct_change(fill_method=None)
        return self.extras["ret"]


def load_context(start: str | None = None, end: str | None = None, prices: pd.DataFrame | None = None) -> Context:
    dcfg = section("data")
    prices = store.read("prices") if prices is None else prices
    if start:
        prices = prices[prices["date"] >= pd.Timestamp(start)]
    if end:
        prices = prices[prices["date"] <= pd.Timestamp(end)]
    mats = price_matrices(prices)
    bench = dcfg["benchmark"]
    if bench not in mats["adj_close"]:
        raise RuntimeError(f"benchmark {bench} missing from price store")
    member = membership(mats, exclude=dcfg["contextSymbols"])
    ever = member["research"].columns[member["research"].fillna(False).any(axis=0)]
    member = {k: v[ever] for k, v in member.items()}          # only names that are ever in the universe
    market_ret = mats["adj_close"][bench].pct_change(fill_method=None)
    tick = member["research"].columns
    rets = mats["adj_close"][tick].pct_change(fill_method=None)
    ind = statistical_industries(rets, market_ret, member["research"])
    events, filings = _events(mats["adj_close"].index)
    fundamentals = store.read_or_empty("fundamentals")
    L.info("context: %d sessions x %d tickers, %d earnings events, %d fundamental rows",
           len(mats["adj_close"]), len(tick), len(events), len(fundamentals))
    ctx = Context(mats=mats, member=member, market_ret=market_ret, industry=ind, events=events,
                  filings=filings, fundamentals=fundamentals)
    ctx.extras["ret"] = rets
    return ctx


def _events(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = pd.DataFrame(columns=["ticker", "session", "form", "items"])
    filings = store.read_or_empty("sec_filings")
    tick = store.read_or_empty("sec_tickers")
    if filings.empty or tick.empty or len(dates) == 0:
        return empty, empty
    cal = sessions(dates[0], dates[-1] + pd.Timedelta(days=10))
    cik_to_ticker = tick.drop_duplicates("cik").set_index("cik")["ticker"]
    f = filings.copy()
    f["ticker"] = f["cik"].map(cik_to_ticker)
    f = f.dropna(subset=["ticker"])
    f["session"] = reaction_session(f["filing_date"], f["acceptance"], cal)
    f = f.dropna(subset=["session"])
    f = f[(f["session"] >= dates[0]) & (f["session"] <= dates[-1])]
    items = section("sec")["earningsItems"]
    is_earn = (f["form"] == "8-K") & f["items"].fillna("").apply(lambda s: any(i in s for i in items))
    ev = f[is_earn][["ticker", "session"]].drop_duplicates()
    return ev.reset_index(drop=True), f[["ticker", "session", "form", "items"]].reset_index(drop=True)
