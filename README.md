# Unpriced

A stock-forecasting lab that freezes its predictions before the US open and keeps score in public.

Unpriced separates **discovering** ideas from **proving** them:

- The **lab** tests ideas on ~20 years of history with purged walk-forward evaluation. Every
  experiment is logged in an append-only ledger, and the bar an idea must clear rises with the
  number of ideas tried. Recent years are a sealed holdout.
- The **live record** is the only evidence that counts. After each trading session each live model ranks
  ~1,000 US stocks for the next five sessions; the ranking is frozen into a file chained by SHA-256
  to the day before, and scored once the horizon has passed. Challengers replace the champion only
  when an anytime-valid confidence sequence says they are better.
- A **$1,000 paper portfolio** trades the champion's ranking at the first open after each forecast
  is frozen, with liquidity-based trading costs, against SPY.

Site: https://prototypeversions.github.io/UnpricedX/ (after GitHub Pages is enabled; see below).

## What it predicts
Cross-sectional ranks of each stock's return from the next open over 1, 5 and 21 sessions
(5 is primary), within a universe chosen each day from that day's liquidity and price only.
Ranks remove market direction, so a rally does not count as skill.

## How it learns
| When (ET) | What | How |
|---|---|---|
| Mon–Fri 18:23, 20:23; Tue–Sat 02:23; daily 06:10, 07:10, 08:20 | Prices, filings, AI reads, forecasts frozen before the next open, portfolio orders, scoring | `daily.yml` |
| Saturday 09:40 | Full data refresh, research sweep, holdout tests, champion/challenger update, live refits | `weekly.yml` |
| Tue–Sat 08:50 | Claude health check: confirms the cycle ran, opens a fix PR if something broke | scheduled Claude session |
| Sunday 10:05 | Claude research session: reads the ledger and live record, proposes new features/experiments as a PR | scheduled Claude session |
| 1st of each month 11:36 | Claude methodology review (`docs/reviews/`) | scheduled Claude session |

Learning happens at four levels, each with its own guardrail:
1. **Parameters** refit weekly on newer data (fixed specs; artifacts hashed into every forecast).
2. **Search**: the sweep explores neighbors of the best model and anything queued in `lab/queue/`;
   each trial raises the significance bar (Bonferroni on the IC t-stat, never below 3).
3. **Selection**: ideas must pass development gates, then the sealed holdout, to be shadowed live.
4. **Promotion**: only live, paired, non-overlapping evidence can replace a champion.

See [`docs/LEARNING.md`](docs/LEARNING.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

GitHub schedules can be delayed or dropped. Evening and overnight attempts give the daily cycle
more time; they do not guarantee delivery. The entry session remains the session immediately after
the data session. A run that starts or finishes forecasting at or after that open fails visibly,
without moving the target to another day. Existing frozen forecasts make retries no-ops.

The **Publish site** workflow deploys committed site files separately from data/research jobs.
It runs after those workflows finish, on site changes, or on manual dispatch. A deployment or
public-content check failure makes that workflow red. `data/deployment.json` on the public site
identifies the deployed source commit and file hashes; its build time is not a forecast issue time.

## Data
Free sources by default: Yahoo daily prices (via `yfinance`), Nasdaq Trader's symbol directory,
and optionally SEC EDGAR filings and XBRL fundamentals. Known bias: free price history only covers
companies listed today. Details and the paid-data upgrade path are in [`docs/DATA.md`](docs/DATA.md).

## Setup (one time, in the repo settings)
1. **Pages**: Settings → Pages → Source: *GitHub Actions*.
2. **Optional secrets** (Settings → Secrets and variables → Actions):
   - `SEC_CONTACT_EMAIL` turns on SEC filings, earnings-event features and fundamentals.
   - `ANTHROPIC_API_KEY` (with SEC on) turns on the AI filing reader and its live challenger.
3. Start: commit `ops/requests/bootstrap.json` containing `{"command": "bootstrap", "minutes": 150}`
   or run the *Ops request* workflow manually.

## Running locally
```bash
pip install -r requirements.txt
python -m pytest -q tests          # offline, synthetic market with planted signals
python -m unpriced.cli demo        # whole system end to end on synthetic data, in a temp dir
python -m unpriced.cli power --ic 0.02   # how long a live edge of that size takes to confirm
```

Research and paper trading only. Nothing here is investment advice.
