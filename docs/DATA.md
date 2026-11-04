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
