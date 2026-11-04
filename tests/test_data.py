import numpy as np
import pandas as pd

from unpriced.data import fundamentals as F
from unpriced.data.sec import reaction_session
from unpriced.data.symbols import parse
from unpriced.calendar import sessions


def test_symbol_directory_filters_non_common():
    nasdaq = pd.DataFrame({
        "Symbol": ["AAPL", "QQQ", "ABCDW", "XYZ", "TST"],
        "Security Name": ["Apple Inc. - Common Stock", "Invesco QQQ Trust, Series 1", "ABC Corp - Warrant",
                          "XYZ Holdings - Class A Common Stock", "Test Co - Common Stock"],
        "Market Category": "Q", "Test Issue": ["N", "N", "N", "N", "Y"], "Financial Status": "N",
        "Round Lot Size": "100", "ETF": ["N", "Y", "N", "N", "N"], "NextShares": "N",
    })
    other = pd.DataFrame({
        "ACT Symbol": ["BRK.B", "SPY", "JPM$C"], "Security Name": ["Berkshire Hathaway Inc. Class B Common Stock",
                                                                    "SPDR S&P 500 ETF Trust", "JPMorgan Preferred Series C"],
        "Exchange": ["N", "P", "N"], "CQS Symbol": "", "ETF": ["N", "Y", "N"], "Round Lot Size": "100",
        "Test Issue": "N", "NASDAQ Symbol": "",
    })
    out = parse(nasdaq, other)
    assert set(out["ticker"]) == {"AAPL", "XYZ", "BRK-B"}


def test_reaction_session_rules():
    cal = sessions("2026-01-01", "2026-01-31")
    fd = pd.Series(["2026-01-06", "2026-01-06", "2026-01-09"])
    acc = pd.Series(["2026-01-06T10:00:00.000Z", "2026-01-06T16:05:00.000Z", "2026-01-09T18:00:00.000Z"])
    out = reaction_session(fd, acc, cal)
    assert list(out.dt.strftime("%Y-%m-%d")) == ["2026-01-06", "2026-01-07", "2026-01-12"]


def _facts(rows):
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "val", "filed"])
    df["cik"], df["tag"], df["prio"] = 1, "x", 0
    for c in ["start", "end", "filed"]:
        df[c] = pd.to_datetime(df[c])
    return df


def test_quarters_from_year_to_date_and_first_reported_values():
    f = _facts([
        ("revenue", "2024-01-01", "2024-03-31", 100, "2024-05-01"),
        ("revenue", "2024-01-01", "2024-06-30", 210, "2024-08-01"),
        ("revenue", "2024-01-01", "2024-09-30", 330, "2024-11-01"),
        ("revenue", "2024-01-01", "2024-12-31", 460, "2025-02-15"),
        ("revenue", "2024-01-01", "2024-12-31", 999, "2025-06-01"),   # later restatement must not win
    ])
    q = F.quarterly_flow(f)
    assert list(q["val"]) == [100, 110, 120, 130]
    assert q["avail"].iloc[-1] == pd.Timestamp("2025-02-15")
    t = F.ttm(q)
    assert t["ttm"].iloc[-1] == 460


def test_fundamentals_become_effective_after_filing(market, ctx):
    from unpriced.features.library import _fund_matrix
    f = market.tables["fundamentals"]
    row = f.iloc[len(f) // 2]
    m = _fund_matrix(ctx, "gp_assets")
    avail = pd.Timestamp(row["available"])
    before = ctx.dates[ctx.dates <= avail]
    after = ctx.dates[ctx.dates > avail]
    assert row["ticker"] in m.columns and len(before) and len(after)
    assert np.isclose(m.at[after[0], row["ticker"]], row["gp_assets"])          # known the session after filing
    assert not np.isclose(m.at[before[-1], row["ticker"]], row["gp_assets"])    # not before


def test_sec_archive_access_denied_means_missing_not_refused():
    import threading
    from unpriced.data.sec import SecClient

    class Resp:
        def __init__(self, code, text):
            self.status_code, self.text = code, text

    class Session:
        def __init__(self, resp):
            self.resp, self.calls = resp, 0

        def get(self, url, timeout=30):
            self.calls += 1
            return self.resp

    c = SecClient.__new__(SecClient)
    c.min_interval, c._last, c._lock, c._explained = 0.0, 0.0, threading.Lock(), True
    c.session = Session(Resp(403, "<?xml version='1.0'?><Error><Code>AccessDenied</Code></Error>"))
    assert c.daily_index(pd.Timestamp("2026-11-03")) is None and c.session.calls == 1


def test_daily_sec_job_catches_up_missed_indexes(monkeypatch, tmp_path):
    from unpriced import store
    from unpriced.data import pipeline, sec as sec_mod

    published = {"2026-11-03": [(1, "A", "8-K")], "2026-11-05": [(2, "B", "10-Q")]}   # 11-04 never appears
    calls = {"index": [], "subs": []}

    class FakeSec:
        def tickers(self):
            return pd.DataFrame({"cik": [1, 2], "ticker": ["AAA", "BBB"], "exchange": ["N", "N"]})

        def daily_index(self, d):
            calls["index"].append(d.date().isoformat())
            rows = published.get(d.date().isoformat())
            if rows is None:
                return None
            return pd.DataFrame([(c, n, f, d.date().isoformat(), "x") for c, n, f in rows],
                                columns=["cik", "company", "form", "filing_date", "filename"])

        def submissions(self, cik, include_older=False):
            calls["subs"].append(cik)
            return {}, pd.DataFrame()

        def companyfacts(self, cik):
            return None

    monkeypatch.setenv("UNPRICED_STORE", str(tmp_path / "store"))   # private store; the shared one stays intact
    monkeypatch.setattr(sec_mod, "SecClient", FakeSec)
    monkeypatch.setattr(pipeline, "sec_enabled", lambda: True)
    store.write("sec_filings", pd.DataFrame({"accession": ["0"], "cik": [1], "form": ["8-K"], "filing_date": ["2026-10-01"]}))
    store.write("sec_index_days", pd.DataFrame({"day": ["2026-11-02"], "status": ["read"]}))

    # evening run for 11-04: 11-03 is read; 11-04 is not published yet, so it stays pending
    r = pipeline.update_sec_job("daily", session=pd.Timestamp("2026-11-04"), tickers=["AAA", "BBB"])
    assert r["indexDays"] == {"read": ["2026-11-03"], "absent": [], "pending": ["2026-11-04"]}
    assert calls["subs"] == [1]
    # next run for 11-05: 11-04 is retried and, still missing for an earlier day, recorded as absent
    r = pipeline.update_sec_job("daily", session=pd.Timestamp("2026-11-05"), tickers=["AAA", "BBB"])
    assert r["indexDays"] == {"read": ["2026-11-05"], "absent": ["2026-11-04"], "pending": []}
    assert calls["subs"] == [1, 2]
    assert list(store.read("sec_index_days")["day"]) == ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"]
