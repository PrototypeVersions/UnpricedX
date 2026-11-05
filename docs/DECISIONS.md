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
