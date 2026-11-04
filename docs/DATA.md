# Data

| Source | Used for | Access | Notes |
|---|---|---|---|
| Nasdaq Trader symbol directory | Candidate list (common stocks, ETF/test-issue flags) | Free, no key | Refreshed weekly; names that delist stay in the store |
| Yahoo Finance via `yfinance` | Daily OHLCV, adjusted close, splits | Free, no key | Research-grade; rate limits possible; survivors only |
| SEC EDGAR (`data.sec.gov`) | Filings + acceptance times, 8-K items, XBRL fundamentals | Free; needs `SEC_CONTACT_EMAIL` | ≤ 8 requests/s (SEC limit is 10) |
| Claude API | AI filing reader | `ANTHROPIC_API_KEY` | Haiku 5.5 by default; ~150 filings/day cap |

## Point-in-time rules
- Prices: Yahoo back-adjusts after every split/dividend. Each update compares an overlap window and
  refetches a ticker's whole history when the past was re-based, so no series mixes bases.
- Market cap uses the price that actually printed (split-adjusted close × later splits) times the
  share count reported at the time.
- Filings count from their reaction session: same session if accepted before 16:00 ET, else next.
- Fundamentals: first-reported value per period; effective the session after the filing date.

## Daily SEC updates
Each daily cycle reads EDGAR's daily index (the list of filings accepted that day) for every day
since the last one it read, then refreshes those companies' filings and, for 10-Q/10-K filers, their
XBRL facts; a full refresh runs weekly. The index for a day is published overnight, so an evening
run finds tonight's file missing (EDGAR answers 403 `AccessDenied` for a file that does not exist
yet) and leaves that day pending for the next run. A file still missing once a later day is being
processed means EDGAR was closed that day (weekends, federal holidays) and is recorded as absent.

## Known bias
Free price history only covers companies still listed, so firms that failed or were acquired are
missing from the past. This flatters strategies that buy losers or small, risky stocks. The live
record is unaffected, and the store keeps the history of names that delist from now on.

## Upgrade path
Prices go through `unpriced/data/prices.py` (`get_provider`). A survivorship-free source (e.g.
Norgate Data, Sharadar via Nasdaq Data Link, CRSP through a university, or EODHD/Massive with
delisted coverage) can be added as another provider class returning the same columns. Record the
switch in `docs/DECISIONS.md`; old forecasts stay as they were.

The store (prices, filings, fundamentals, fitted models) lives in the `datastore` GitHub release,
not in git. It can always be rebuilt with `{"command": "bootstrap"}`.
