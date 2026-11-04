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
