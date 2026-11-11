# Plan: a more useful Hub site (to be built in PrototypeVersions/UnpricedHub)

Written 2026-11-11 at the owner's request. Today the Hub page mostly says whether each forecast arrived on
time. The owner wants it to show what the forecasts say and how they are doing, with no links to any
GitHub repository. A session with push access to UnpricedHub builds this; until then this file is the spec.

## Why keep the Hub
It is the only place that (1) keeps a copy of every forecast from both labs with the time it was received,
(2) grades both labs with the same code and its own prices, and (3) shows the two labs side by side.

## What the page shows (from `site/data/hub.json`, built by the Hub's own code)
1. **Latest forecasts, as received.** Stock Lab: decision session, entry open, top 10 and bottom 10 names.
   Risk Lab: each task's forecast for the next five sessions (expected variance vs. its own history;
   probability of a >3% drawdown). Each with its receipt time.
2. **Scorecard per lab, graded by the Hub.** Stock Lab: mean daily rank IC with a 95% band, top-vs-bottom
   decile spread, cumulative long-short return, and the same for its baseline. Risk Lab: loss vs. its
   baseline (e.g. QLIKE for variance), Brier score and a calibration table for the drawdown probability.
   Before anything is graded, say when the first grades arrive.
3. **Agreement check.** The Hub's score vs. each lab's own score for the same forecasts; flag gaps.
4. **On-time record.** A calendar strip per lab (on time / late / missed / invalid); the reason for anything
   not on time opens in a popup.
5. **How it works.** The rules (what "on time" means, how each forecast type is graded, why the labs are
   not ranked against each other), on the page or in popups.

## Links
Remove every github.com link (header, footer, per-lab repo link, contract link). The contract and any file
detail open in a popup or a section. The labs' public sites may still be linked.
Never rewrite `records/`; this is a site and site-data change only.
