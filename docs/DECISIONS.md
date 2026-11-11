# Decisions

Append-only log of choices that affect how results should be read.

## 2026-11-03: UnpricedX design (replaces the V1 prototype in PrototypeVersions/Unpriced)
- **Discovery on history, proof only live.** V1 tried to learn everything from a forward record of
  40 megacaps, which would take years to show anything. The lab now uses ~20 years and ~1,000
  stocks; the forward record is for confirmation.
- **Cross-sectional ranks, not 120 binary bets vs SPY.** Removes the shared market-timing bet that
  made every V1 forecast on a given day move together.
- **Next-open execution from frozen files.** No intraday job; robust to GitHub schedule delays.
- **Time-ordered early stopping.** V1's boosted model used scikit-learn's random validation split,
  which leaks with overlapping labels; on pure noise it produced 20-day probabilities from 0.27 to 0.67.
- **Statistical industries.** Avoids applying today's classification to the past and works without SEC access.
- **Pre-registered initial experiments and provisional champion (`ens_all`)**, fixed before any
  real-data result was seen.
- **Gates**: IC t ≥ max(3, Bonferroni over trials); PSR of net long–short ≥ 0.95; holdout IC t ≥ 2
  and positive net. Deflated Sharpe is reported but not gated (with heterogeneous trials, including
  single factors, its trial-variance term made every realistic signal fail).
- **Promotion** by asymptotic confidence sequences on weekly paired blocks, minimum 26 blocks. The
  finite-sample empirical-Bernstein version was tried first: with ICs bounded only by [−1, 1] it would
  need about ten years to confirm an IC of 0.02, so it is kept for reference only.
- **AI filing reader is forward-only**, because a language model has seen the outcomes of filings
  before its training cutoff.

## 2026-11-03 (evening): accounting change after the first real-data sweep
Made after seeing the first real-data results, so recorded here in full.
- **What we saw:** learned models rank stocks with real skill (development IC ≈ 0.016, t ≈ 4.5; the
  provisional champion's sealed-holdout IC 0.026, t 3.5, positive in every holdout year), but a weekly
  top-minus-bottom decile rebuilt from scratch each week replaced ~60% of each leg, and costs of
  ~14% a year erased a 6–14% gross spread.
- **Change:** the primary economic test now uses the same hysteresis the live portfolio always used
  (enter in the top/bottom 10%, keep until a name leaves the top/bottom 30%; long-only keeps names until
  rank 40). The from-scratch version is still reported as `longShortFull`. Gates, eras and thresholds
  are unchanged. `metricsVersion` 2 marks entries scored this way; earlier entries remain in the ledger.
- **Why it is not moving the goalposts:** the lab was scoring a portfolio the live system never trades.
  Re-scoring the same model is not counted as a new trial; the holdout re-looks it causes are logged and
  capped like any other.
- **Also:** trials are now counted by the model actually fit (`effSig`), so specs that collapse to the
  same features when data is missing are not double-counted (36 → 30), and the search explores around
  the three best models instead of stopping when the single best one's neighbors are exhausted.

## 2026-11-03 (night): stricter search, holdout and live statistics; inputs saved with every forecast
Prompted by the second real-data sweep and by GPT's cross-review of the design.
- **Search discipline.** The automatic search spent ~30 trials chaining small hyperparameter tweaks of one
  model (each raising the significance bar for everything after it). Now: at most 6 automatic trials per
  sweep, only around models with an IC t-stat ≥ 2, at most two modifications deep, structural changes
  (drop a feature family, change horizon) before hyperparameters, and readable names (`root~mod~mod`).
- **Live slots.** One slot per distinct fitted model (duplicates retired); exploratory challengers that failed
  the gates must be pre-registered models, not tweaked variants.
- **Holdout looks.** Every distinct model scored on the sealed 2022+ years raises the holdout bar
  (Bonferroni over looks, never below t = 2) and looks are capped at 3 a week and 30 a year.
- **Live bands.** The confidence sequence now uses a Newey-West long-run variance so persistent market
  regimes cannot make it too narrow; tested on autocorrelated null data (≤ 8% false alarms over ten years
  of weekly looks at ρ = 0.4). Median time to confirm IC 0.02 moved from ~119 to ~126 weeks.
- **Null controls.** Each sweep runs pre-registered models on data whose outcomes are shuffled among the
  stocks of each day and logs how often the gates pass (lab/null_controls.jsonl). Not counted as trials.
- **Recomputable forecasts.** Each snapshot now stores the exact ranked inputs every model saw, plus a
  fingerprint of that session's price rows and of the price store (idea borrowed from UnpricedZ).
- **SEC identification.** The first SEC attempt was refused (HTTP 403); the User-Agent now follows SEC's
  documented "<name> <email>" format, and refusals log SEC's explanation.

## 2026-11-03 (late): the paper portfolio holds at most 3 names from one statistical sector
- **What prompted it.** The first portfolio decision (from the 2026-11-03 forecast, filled at the
  2026-11-04 open) picked ten chip and hardware makers (MU, AMD, LRCX, AMAT, MRVL, WDC, TER, ASX, GLW,
  ALAB). That is one bet on one theme, not a test of stock selection. The decision was made after
  seeing the portfolio's composition and before any of its returns were known.
- **Change.** From the next rebalance decision on (expected from the 2026-11-10 forecast), at most 3 of
  the 10 holdings may come from one *statistical sector*: groups of about 50 stocks whose market-residual
  returns moved together over the past year, built the same way and on the same calendar-anchored
  January/July refresh as the statistical industries, so they use only past data. Held names count
  toward the cap and the lowest-ranked excess holding is sold. Names with too little history to be
  grouped are not capped. The 2026-11-04 orders stay exactly as decided.
- **Why sectors, not the existing industries.** The ~40 statistical industries (about 25 stocks each)
  split the theme across three groups, so a cap of 3 per industry would still have held six chip names.
  With groups of about 50 (~20 groups) the whole complex is one group and a cap of 3 gives MU, AMD,
  LRCX, BHP, JCI, FCX, CAT, STT, CCJ, NUE; groups of about 120 (~8) start merging unrelated industries
  (e.g. JCI and CAT with semiconductors). The size was chosen by looking at that night's groups, not at
  any backtest result, and will not be tuned on results.
- **Lab.** The lab's long-only test now calls the live portfolio's own selection function, so history and
  live follow the same rule. `metricsVersion` 3 marks entries scored this way; version-2 entries remain in
  the ledger. The primary tests (IC and the hysteresis long-short) do not change, so gates are unaffected.
  This Saturday's sweep re-scores everything anyway, because SEC data now adds fundamental families
  (new `dataSig`); re-scoring the same model is not a new trial.
- **Forecasts are untouched.** The cap changes which ranked names the paper portfolio buys, not the
  ranks that are frozen and scored. Each order now records the cap and the sector of every target.

## 2026-11-05: operational repair of timing and publication
- **Observed:** the November 4 morning schedules ran hours after the open. The latest daily job
  stayed green even though Pages waited ten minutes, timed out, and canceled its deployment.
  The forecast code would choose a later entry session if an unfrozen forecast ran late, changing
  the next-open task relative to the training labels.
- **Timing:** add 18:23 and 20:23 ET weekday attempts and 02:23 ET Tuesday–Saturday retries, keeping
  the existing morning attempts. The first complete, timely run freezes the same daily task;
  later attempts remain no-ops. GitHub cron remains best effort.
- **Deadline:** fix entry to the next exchange session after the data session. Check the open both
  before fetching and immediately before freezing. Missing it produces a visible operational error
  (exit 78), no new forecast, and no new paper order; no retroactive or shifted-entry forecast.
  Paper-account mutation follows the final check so a slow forecast cannot append orphan fills.
- **Publication:** one serialized Pages workflow, outside the state-job lock, checks out current main,
  deploys with an explicit environment and a twenty-minute timeout, and then checks hashes of the
  actual public HTML, assets, JSON and Hub export against the uploaded artifact. Errors are failures,
  not ignored steps. A public deployment manifest distinguishes a site rebuild from a forecast issue.
  Queued state jobs also check out current main; integrity must pass before their commits are saved.
- **Scope:** no model, feature, holdout, gate, or promotion-policy changes. No historical forecast,
  experiment, score, initial order, or registered result is rewritten. Regression tests cover late
  starts, crossing the deadline during computation, holiday/DST calendars, and stale public files.

## 2026-11-05: duplicate challenger; site wording
- **Bug.** Two live challengers, `gbm_all` and `gbm_all_h21-h5`, are the same fitted model: the
  neighbour tweak "h5" turned the 21-day variant back into `gbm_all` (same spec hash `d90c9d52`). The
  one-slot-per-model check only looked at models re-scored under the current accounting, and
  `gbm_all_h21-h5` never was, so it kept a slot. The check now identifies every live model from its
  spec, and the next registry update retires the duplicate. Its frozen forecasts stay in the record
  as they are; they are identical in substance to `gbm_all`'s, so no comparison was affected.
- **Lab page.** It listed every past evaluation of a model (91 rows for 66 ideas); it now shows each
  experiment's latest evaluation. Years with fewer than 60 out-of-sample sessions are no longer drawn
  in the by-year chart (the champion's "2007" bar was two December days).
- **Wording.** The site now says what it does in plain language ("Can it pick stocks better than
  chance?"), describes the evening timing introduced on 2026-11-05, names models by what they are,
  and links to UnpricedHub. No rule, gate or record changed.

## 2026-11-05: review fixes from the Risk Lab's author; benchmark; narrower AI reader
Prompted by an outside review (GPT, author of UnpricedZ). All points below were checked and accepted.
- **Missing-exit look-ahead (bug, research only).** The backtest portfolios dropped stocks whose
  5-day exit price was missing *before* choosing holdings, i.e. they knew in advance which stocks
  would stop trading. Now holdings are chosen from every ranked stock, a missing exit counts as a 0%
  return for that slot, and a day is skipped only if most exits are missing (a data gap, not a stock
  event). `METRICS_VERSION` 3 → 4. Old ledger lines stay as they were; the Saturday sweep re-scores
  every idea under version 4, and the site shows the version beside each result so both are visible.
  The ranking measure (IC) is computed separately and did not change; the gates that use long-short
  net returns can move either way under version 4, and the gate rules themselves are unchanged.
  Live forecasts and the live paper account were not affected (they never see future prices).
- **Benchmark.** The simple rule "favor last week's losers" (`factor_ret_1w`) scored nearly the same
  development IC as the champion blend (0.0154 vs 0.0156). It now runs live with role `benchmark`:
  forecast, frozen and scored like every model, exported to the Hub as role `baseline`, never
  promoted. If the champion cannot beat it live, that is the finding.
- **Hub export cutoff.** `informationCutoff` was a fixed 17:30 ET; it is now the forecast's own
  `decidedAt`, which is the true moment after which nothing else can enter it.
- **AI filing reader narrowed before its first read.** One question, on earnings releases only
  (8-K item 2.02): did management raise, maintain, lower, withdraw or first give guidance, with the
  supporting sentence quoted, and an explicit "unclear". A keyword rule reads the same text. Two
  shadow challengers (champion + AI guidance; champion + rule guidance) test whether the AI adds
  anything beyond the rule. Each reading records model, prompt version, read time, size and tokens.
  A reading counts only for forecasts frozen after it finished, is graded from the first open after
  it finished (previously: from the filing's reaction session), and reading has a 15-minute budget so
  it cannot delay the forecast. No reading has happened yet (no API key), so nothing is rewritten.
- **Not changed:** gates, eras, promotion rule, holdout. The three tree challengers that gave
  identical scores on Nov 3–4 are handled by the duplicate fix above plus Saturday's re-score.

## 2026-11-06: AI reader was missing most press releases (bug)
- **Bug.** The reader found an earnings 8-K's press release only when its file name contained
  "ex99"/"ex-99". Many companies name it otherwise (e.g. `q3pressrelease.htm`, `exhibit991.htm`),
  so on the first night 71 of 150 readings saw only the 8-K cover page and correctly answered
  "unclear" (median 4,000 characters of text versus 22,000 overall). The reader now takes the
  exhibits whose declared document type is EX-99.x from the filing's EDGAR index page, with the
  file-name guess only as a fallback, and gives the cover page a smaller share of the text budget.
- **Effect on the record.** The 2026-11-05 readings stay as they were (append-only). An "unclear"
  reading contributes 0 to the guidance challengers, so the Nov 5 forecast's AI and rule challengers
  simply had less coverage than intended; no champion, benchmark or portfolio forecast was affected.
  Those filings are not re-read. The prompt is unchanged (`guidance-1`).

## 2026-11-06: daily-cycle schedule written in UTC
GitHub ran the daily workflow's `timezone: America/New_York` crons as UTC times. The intended
18:23 and 20:23 ET evening attempts fired at 13:23 and 15:23 ET (before the close, so no-ops), and
the first real evening attempt was "02:23", which ran at 21:23 ET plus queue delay. On 2026-11-05 the
forecast was frozen at 19:16 ET only because the evening check dispatched the cycle by hand. The crons
are now plain UTC times, with ET equivalents in comments, giving attempts from 17:40 ET onward.
Timing only: no forecast, rule, gate or frozen file changed. The cycle's own deadline check is unchanged.
The Hub's `hub.yml` uses the same `timezone:` form and probably has the same problem; that is the Hub's
repository to fix.

## 2026-11-11: no repository links on the site; dates shown in New York time
- **Site.** At the owner's request the site no longer links to the GitHub repository. The forecast file's
  details (path, lock-in time, full SHA-256 and how the hash chain works) open in a popup instead.
- **Dates.** Timestamps are now shown in New York time for every viewer; previously a viewer outside US
  time zones could see the evening's lock-in date as the next day. Display only.
- **Not changed:** gates, eras, promotion, portfolio rules, holdout, any frozen file.
