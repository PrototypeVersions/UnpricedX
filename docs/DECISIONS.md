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
