# Methodology

## Target
For stock *i* and decision date *t* (after the close of session *t*, before the next open):

`fwd_h(i,t) = open(i, t+1+h) / open(i, t+1) − 1` (total-return adjusted)

Models are trained on the within-date percentile rank of `fwd_h`, so only relative ordering is
learned. h ∈ {1, 5, 21}; 5 is the primary horizon for the champion, the portfolio and promotion.

## Universe (point in time)
Each date: price that actually printed ≥ $5, ≥ ~1 year of history, ranked by 63-session median
dollar volume. Research universe = top 1,000; tradable = top 500. Nothing about a company's
future is used to decide whether it is in the universe.

## Features
All features are percentile-ranked within each date (−0.5…0.5). Families: momentum, reversal
(including industry-adjusted and overnight/intraday split), risk, liquidity, earnings events
(SEC 8-K item 2.02), fundamentals (SEC XBRL, first-reported values, effective the session after
filing) and valuation. Industries are statistical (k-means on a year of market-residual returns,
refreshed each January and July, used strictly afterwards). See `unpriced/features/library.py`;
each feature cites its source paper.

`tests/test_leakage.py` deletes the future and checks no past feature value changes.

## Models
- `factor`: one feature with its literature sign (benchmarks, never champions)
- `ridge`: ridge regression on ranked features
- `gbm`: histogram gradient boosting, early-stopped on a **time-ordered, purged** validation block
- `ensemble`: average of members' per-date z-scores

## Evaluation (lab)
Purged walk-forward: refit every 126 sessions on rows whose labels completed before the block
(purge = h+1 sessions), training dates sampled every 5th session.

Per experiment: daily IC (Spearman), Newey–West t (lag h), hit rate, IC by year; top-minus-bottom
decile long–short every h sessions after liquidity-tier costs (4–30 bps per side); long-only top-20
tradable vs SPY; probabilistic and deflated Sharpe ratios (Bailey & López de Prado 2014) using the
ledger's trial count and Sharpe dispersion.

**Development gate**: IC t ≥ max(3, Bonferroni z for 5% two-sided over N trials), PSR(net) ≥ 0.95,
positive net long–short. **Holdout gate** (2022→): IC t ≥ 2 and positive net long–short. At most
3 holdout looks per week, all logged.

## Selection
Pre-registered: if no learned model passes both gates, the champion is the pre-registered
`ens_all` (labelled provisional). The first model to pass both gates replaces a provisional
champion; after that only live evidence can.

## Live record
- Snapshots: `live/snapshots/YYYY/YYYY-MM-DD.json.gz`, deterministic gzip, SHA-256 chained in
  `live/chain.jsonl`. Entry session = first open after the snapshot was frozen.
- Scoring: IC, top–bottom decile spread and top-decile excess per model and horizon.
- Promotion: asymptotic confidence sequence (Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas, Annals of
  Statistics 2024), variance shrunk toward 0.1² for the first weeks, on paired weekly non-overlapping
  IC differences. Promote when the 95% band is above 0 after ≥ 26 blocks; retire a challenger when it
  is below 0. Simulated false-alarm rate when looking every week for ten years: ≤ 6%
  (`tests/test_live.py`). The finite-sample empirical-Bernstein CS is also implemented but needs ~a
  decade for realistic edges because ICs are only known to lie in [−1, 1].

## Paper portfolio
$1,000; long-only; 10 equal-weight names from the tradable universe by champion rank; keep until
rank > 40; at most 3 names from one statistical sector (~50 stocks whose residual returns co-moved
over the past year; refreshed each January and July from past data only); rebalance every 5 sessions
at the first open after the decision; 2 bps + liquidity-tier cost per side; 2% no-trade band;
total-return marking; SPY benchmark from the first fill. The lab's long-only test calls the same
selection function (`live/portfolio.py: select_holdings`).

## Power
With ~1,000 names a weekly IC has a standard deviation of roughly 0.05–0.10. At 0.08, the median time
to confirm a true edge (`python -m unpriced.cli power`) is about 119 weeks for IC 0.02, 59 weeks for
0.03 and 28 weeks for 0.05. A realistic edge takes one to two years of live evidence; that is a
property of markets, not something to tune away.
