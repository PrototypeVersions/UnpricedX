"""Tests for the news/no-news return split and the return-seasonality feature (research 2026-11-08)."""
import numpy as np
import pandas as pd

from unpriced.features.context import Context
from unpriced.features.library import REGISTRY, news_ret_1m, ret_1w_nonews, season_amn


def _ctx(prices: pd.DataFrame, filings: pd.DataFrame | None = None, market: pd.Series | None = None) -> Context:
    member = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    mret = market if market is not None else pd.Series(0.0, index=prices.index)
    empty = pd.DataFrame(columns=["ticker", "session", "form", "items"])
    return Context(mats={"adj_close": prices, "adj_open": prices}, member={"research": member},
                   market_ret=mret, industry=pd.DataFrame(np.nan, index=prices.index, columns=prices.columns),
                   events=empty, filings=filings if filings is not None else empty, fundamentals=pd.DataFrame())


def test_families_are_new_so_existing_models_keep_their_features():
    assert REGISTRY["ret_1w_nonews"].family == "news"
    assert REGISTRY["news_ret_1m"].family == "news"
    assert REGISTRY["season_amn"].family == "seasonal"
    from unpriced.lab.specs import ALL
    assert "news" not in ALL and "seasonal" not in ALL


def test_news_split_separates_filing_days_from_quiet_days():
    dates = pd.bdate_range("2020-01-01", periods=300)
    r = pd.DataFrame(0.0, index=dates, columns=["AAA", "BBB"])
    r.iloc[290] = [0.10, 0.10]          # both jump on the same day
    prices = 100 * (1 + r).cumprod()
    filings = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"], "session": [dates[290], dates[100], dates[100]],
                            "form": ["8-K", "10-Q", "10-Q"], "items": ["2.02", "", ""]})
    ctx = _ctx(prices, filings)
    quiet = ret_1w_nonews(ctx)
    loud = news_ret_1m(ctx)
    t = dates[292]
    assert np.isclose(quiet.at[t, "AAA"], 0.0)                    # AAA's jump came with news: not a quiet move
    assert np.isclose(quiet.at[t, "BBB"], np.log(1.10))           # BBB's jump had no filing
    assert np.isclose(loud.at[t, "AAA"], np.log(1.10))
    assert np.isclose(loud.at[t, "BBB"], 0.0)
    # before the jump nothing is known about it
    assert np.isclose(quiet.at[dates[289], "BBB"], 0.0) and np.isclose(loud.at[dates[289], "AAA"], 0.0)


def test_news_features_need_filing_coverage():
    dates = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(100.0, index=dates, columns=["AAA", "BBB"])
    filings = pd.DataFrame({"ticker": ["AAA"], "session": [dates[5]], "form": ["8-K"], "items": [""]})
    ctx = _ctx(prices, filings)
    q = ret_1w_nonews(ctx)
    assert q["BBB"].isna().all()                 # never filed: coverage unknown, not "no news"
    assert q["AAA"].iloc[10:250].notna().all()
    assert q["AAA"].iloc[-1:].isna().all()       # coverage expires 252 sessions after the last filing
    assert ret_1w_nonews(_ctx(prices)).isna().all().all()     # no SEC data at all: family unavailable


def test_season_amn_finds_a_recurring_calendar_month():
    dates = pd.bdate_range("2010-01-01", "2016-12-31")
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.005, (len(dates), 4)), index=dates, columns=["AAA", "BBB", "CCC", "DDD"])
    r.loc[r.index.month == 3, "AAA"] += 0.005    # AAA always does well in March
    prices = 50 * (1 + r).cumprod()
    s = season_amn(_ctx(prices))
    feb_end = s.loc["2016-02-26"]                # holding period falls mostly in March
    mar = s.loc["2016-03-15"]
    jul = s.loc["2016-07-15"]
    assert feb_end["AAA"] == mar.max() and mar["AAA"] == mar.max()
    assert mar["AAA"] > 0.05 and jul["AAA"] < mar["AAA"] - 0.05
    assert s.loc["2011-03-15"].isna().all()      # fewer than three past years: no value


def test_season_amn_ignores_the_unfinished_month():
    dates = pd.bdate_range("2010-01-01", "2015-06-30")
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (len(dates), 3)), index=dates, columns=["AAA", "BBB", "CCC"])
    prices = 50 * (1 + r).cumprod()
    full = season_amn(_ctx(prices))
    cut = prices.loc[:"2015-06-12"]
    part = season_amn(_ctx(cut))
    pd.testing.assert_frame_equal(full.loc[:"2015-06-12"], part)


def test_gbm_fit_skips_a_feature_that_only_starts_in_the_validation_block():
    from unpriced.models import fit
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2015-01-01", periods=200)
    rows = pd.DataFrame({"date": np.repeat(dates, 50), "ticker": np.tile([f"T{i}" for i in range(50)], 200)})
    rows["ret_1w"] = rng.uniform(-0.5, 0.5, len(rows))
    late = rows["date"] >= dates[-20]                      # inside the last 15% of dates (validation)
    rows["season_amn"] = np.where(late, rng.uniform(-0.5, 0.5, len(rows)), np.nan)
    rows["target_5"] = rng.uniform(-0.5, 0.5, len(rows))
    m = fit({"kind": "gbm", "features": ["ret_1w", "season_amn"], "horizon": 5, "params": {"max_iter": 5}}, rows)
    assert m.features == ["ret_1w"]
