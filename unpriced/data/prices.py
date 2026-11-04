"""Daily price data: provider adapters and incremental, adjustment-aware updates.

Table `prices` (long format):
    date, ticker, open, high, low, close, adj_close, volume
`close` is split-adjusted (Yahoo convention); `adj_close` is split+dividend adjusted.
Total-return-consistent opens are derived as open * adj_close / close.

Yahoo back-adjusts history after every split and dividend, so appending new rows to an old
history would silently mix adjustment bases. `update_prices` compares an overlap window
on every run and refetches a ticker's full history whenever the past has been re-based.
"""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd

from .. import store
from ..config import section
from ..util import log

L = log("prices")
COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "split"]
FIELD_MAP = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
             "Stock Splits": "split"}


class ProviderError(RuntimeError):
    pass


def _tidy(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = frame.rename(columns=FIELD_MAP)
    if "adj_close" not in df and "close" in df:
        df["adj_close"] = df["close"]
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume", "split"] if c in df]
    df = df[keep].dropna(subset=["close"]) if "close" in keep else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=COLS)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df = df.reset_index(names="date")
    df["ticker"] = ticker
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        if c not in df:
            df[c] = np.nan
    df["split"] = pd.to_numeric(df["split"], errors="coerce").fillna(0.0) if "split" in df else 0.0
    return df[COLS]


def _split_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Handle the column layouts yfinance has used across versions."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lv0 = set(map(str, raw.columns.get_level_values(0)))
        lv1 = set(map(str, raw.columns.get_level_values(1)))
        ticker_level = 0 if len(lv0 & set(tickers)) >= len(lv1 & set(tickers)) else 1
        for t in tickers:
            try:
                sub = raw.xs(t, axis=1, level=ticker_level)
            except KeyError:
                continue
            out[t] = sub
    elif len(tickers) == 1:
        out[tickers[0]] = raw
    return out


class YahooProvider:
    """Free, research-grade prices via yfinance. Not point-in-time; survivors only."""

    name = "yahoo"

    def __init__(self) -> None:
        import yfinance  # noqa: F401  (import here so offline tests never need it)
        self.cfg = section("data")

    def fetch(self, tickers: list[str], start: str, end: str | None = None) -> tuple[pd.DataFrame, list[str]]:
        import yfinance as yf

        batch = int(self.cfg["yahooBatchSize"])
        pause = float(self.cfg["yahooPauseSeconds"])
        frames, failed = [], []
        for i in range(0, len(tickers), batch):
            chunk = tickers[i : i + batch]
            raw = None
            for attempt in range(4):
                try:
                    raw = yf.download(
                        chunk, start=start, end=end, interval="1d", auto_adjust=False, actions=True,
                        group_by="ticker", threads=True, progress=False, timeout=30,
                    )
                    break
                except Exception as exc:  # rate limits, transient network errors
                    wait = pause * (3 ** attempt)
                    L.warning("yahoo batch %s failed (%s); retry in %.0fs", i // batch, exc, wait)
                    time.sleep(wait)
            parts = _split_download(raw, chunk) if raw is not None else {}
            for t in chunk:
                df = _tidy(parts[t], t) if t in parts else pd.DataFrame(columns=COLS)
                if df.empty:
                    failed.append(t)
                else:
                    frames.append(df)
            if i + batch < len(tickers):
                time.sleep(pause)
            if (i // batch) % 10 == 0:
                L.info("yahoo: %d/%d tickers requested, %d failed so far", min(i + batch, len(tickers)), len(tickers), len(failed))
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
        return data, failed


def get_provider(name: str | None = None):
    name = name or section("data")["priceProvider"]
    if name == "yahoo":
        return YahooProvider()
    if name == "synthetic":
        from .synthetic import SyntheticProvider
        return SyntheticProvider()
    raise ValueError(f"unknown price provider {name}")


def _compact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["ticker"] = df["ticker"].astype(str)
    # float32 keeps ~7 significant digits: ample for prices, and it halves memory and storage
    for c in ["open", "high", "low", "close", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("float32")
    if "split" not in df:
        df["split"] = 0.0
    df["split"] = pd.to_numeric(df["split"], errors="coerce").fillna(0.0).astype("float32")
    return df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)


def update_prices(tickers: Iterable[str], provider=None, full: bool = False, overlap_days: int = 12) -> dict:
    """Bring `prices` up to date for `tickers`. Returns a small report dict."""
    provider = provider or get_provider()
    cfg = section("data")
    tickers = sorted(set(tickers))
    existing = store.read_or_empty("prices")
    report = {"requested": len(tickers), "new": 0, "refetched": 0, "appended": 0, "failed": []}

    have = set(existing["ticker"].unique()) if not existing.empty else set()
    new = [t for t in tickers if t not in have or full]
    old = [t for t in tickers if t in have and not full]

    pieces = []
    if old:
        last = existing.groupby("ticker")["date"].max()
        start_ts = existing["date"].max() - pd.Timedelta(days=overlap_days * 2)
        stale = [t for t in old if last.loc[t] < start_ts]
        new += stale                      # gaps: refetch whole history
        old = [t for t in old if t not in set(stale)]
    if old:
        recent, failed = provider.fetch(old, start_ts.date().isoformat())
        report["failed"] += failed
        recent = _compact(recent) if not recent.empty else recent
        tol = float(cfg["adjustmentMismatchTolerance"])
        tail = existing[existing["date"] >= start_ts]
        tail_by = {t: g.set_index("date") for t, g in tail.groupby("ticker")}
        refetch = []
        for t, grp in (recent.groupby("ticker") if not recent.empty else []):
            hist = tail_by.get(t)
            g = grp.set_index("date")
            if hist is None:
                refetch.append(t)
                continue
            common = hist.index.intersection(g.index)
            if len(common):
                c_old, c_new = hist.loc[common, "close"], g.loc[common, "close"]
                a_old, a_new = hist.loc[common, "adj_close"], g.loc[common, "adj_close"]
                rel = max(float(((c_new / c_old) - 1).abs().max()), float(((a_new / a_old) - 1).abs().max()))
                if not np.isfinite(rel) or rel > tol:
                    refetch.append(t)
                    continue
            pieces.append(grp[grp["date"] > hist.index.max()])
        if refetch:
            L.info("adjustment basis changed for %d tickers; refetching full history", len(refetch))
            new += refetch
            report["refetched"] = len(refetch)
        report["appended"] = int(sum(len(p) for p in pieces))

    replace: set[str] = set()
    if new:
        full_hist, failed = provider.fetch(sorted(set(new)), cfg["historyStart"])
        report["failed"] += failed
        report["new"] = len(set(new)) - report["refetched"]
        if not full_hist.empty:
            full_hist = _compact(full_hist)
            replace = set(full_hist["ticker"].unique())   # only drop old rows we actually replaced
            pieces.append(full_hist)

    base = existing[~existing["ticker"].isin(replace)] if not existing.empty else existing
    combined = pd.concat([base] + [p for p in pieces if len(p)], ignore_index=True) if pieces else base
    if combined is not None and len(combined):
        combined = _compact(combined)
        store.write("prices", combined)
    report["failed"] = sorted(set(report["failed"]))
    report["rows"] = 0 if combined is None else int(len(combined))
    report["lastDate"] = None if combined is None or combined.empty else combined["date"].max().date().isoformat()
    L.info("price update: %s", {k: v for k, v in report.items() if k != "failed"} | {"failedCount": len(report["failed"])})
    return report


def wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Date x ticker matrix for one field (computed in float64 for numerical safety)."""
    return prices.pivot(index="date", columns="ticker", values=field).sort_index().astype("float64")


def price_matrices(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Total-return-consistent matrices used by features and labels."""
    close = wide(prices, "close")
    adj = wide(prices, "adj_close").reindex_like(close)
    factor = (adj / close).where(close > 0)
    out = {
        "close": close,                       # split-adjusted close (Yahoo basis)
        "adj_close": adj,                     # total-return close
        "adj_open": wide(prices, "open").reindex_like(close) * factor,
        "adj_high": wide(prices, "high").reindex_like(close) * factor,
        "adj_low": wide(prices, "low").reindex_like(close) * factor,
        "volume": wide(prices, "volume").reindex_like(close),
    }
    out["dollar_volume"] = close * out["volume"]
    # Yahoo's close is split-adjusted to today's share basis. Undo later splits to recover the
    # price that actually printed, which is what reported share counts must be multiplied by.
    split = wide(prices, "split").reindex_like(close).fillna(0.0)
    ratio = split.where(split > 0, 1.0)
    later = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1.0)   # product of splits strictly after t
    out["raw_close"] = close * later
    return out
