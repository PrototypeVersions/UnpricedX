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
