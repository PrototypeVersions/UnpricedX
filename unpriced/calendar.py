"""NYSE session calendar.

Uses pandas_market_calendars when installed (authoritative: holidays, special closures,
early closes). Falls back to a rule-based calendar that covers regular NYSE holidays,
which is only used in offline tests.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache

import pandas as pd

from .util import ET

try:  # pragma: no cover - exercised in CI
    import pandas_market_calendars as _mcal
except Exception:  # pragma: no cover
    _mcal = None


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _rule_holidays(year: int) -> set[date]:
    hol = {
        _observed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),   # MLK
        _nth_weekday(year, 2, 0, 3),   # Presidents
        _easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),     # Memorial
        _observed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),   # Labor
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(date(year, 12, 25)),
    }
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))
    return hol


@lru_cache(maxsize=64)
def _sessions(start: str, end: str) -> pd.DatetimeIndex:
    if _mcal is not None:
        sched = _mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
        return pd.DatetimeIndex(sched.index.normalize())
    days = pd.bdate_range(start, end)
    years = range(days[0].year if len(days) else 2000, (days[-1].year if len(days) else 2000) + 1)
    hol = set().union(*[_rule_holidays(y) for y in years]) if len(days) else set()
    return pd.DatetimeIndex([d for d in days if d.date() not in hol])


def sessions(start, end) -> pd.DatetimeIndex:
    return _sessions(pd.Timestamp(start).date().isoformat(), pd.Timestamp(end).date().isoformat())


@lru_cache(maxsize=512)
def session_close(day: str) -> datetime | None:
    """Close time (ET) for a session date, or None when the market is closed."""
    if _mcal is not None:
        sched = _mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day)
        if sched.empty:
            return None
        return sched.iloc[0]["market_close"].tz_convert(ET).to_pydatetime()
    if pd.Timestamp(day) not in sessions(day, day):
        return None
    return datetime.combine(pd.Timestamp(day).date(), time(16, 0), tzinfo=ET)


def session_open(day: str) -> datetime | None:
    if _mcal is not None:
        sched = _mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day)
        if sched.empty:
            return None
        return sched.iloc[0]["market_open"].tz_convert(ET).to_pydatetime()
    if session_close(day) is None:
        return None
    return datetime.combine(pd.Timestamp(day).date(), time(9, 30), tzinfo=ET)


def last_completed_session(now: datetime, settle_minutes: int = 60) -> pd.Timestamp:
    """Most recent session whose close was at least `settle_minutes` before `now`."""
    start = (pd.Timestamp(now.date()) - pd.Timedelta(days=14)).date().isoformat()
    for d in reversed(sessions(start, now.date().isoformat())):
        close = session_close(d.date().isoformat())
        if close is not None and close + timedelta(minutes=settle_minutes) <= now:
            return d
    raise RuntimeError("no completed session in the last 14 days")


def next_session(day) -> pd.Timestamp:
    d = pd.Timestamp(day)
    nxt = sessions(d + pd.Timedelta(days=1), d + pd.Timedelta(days=14))
    return nxt[0]
