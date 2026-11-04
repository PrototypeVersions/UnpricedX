# How Unpriced learns and improves

Learning is split into levels that move at different speeds, each with a guardrail that stops it
from fooling itself.

| Level | What changes | Cadence | Guardrail |
|---|---|---|---|
| 1. Parameters | Weights of each live model | Weekly refit | Spec frozen; artifact SHA-256 recorded in every forecast |
| 2. Search | Which specs get tried | Weekly sweep, ~3.5 h budget | Every trial logged; IC bar rises with trial count |
| 3. Selection | Which specs run live as challengers | Weekly | Development gates, then sealed holdout (capped looks) |
| 4. Promotion | Which model is champion | Daily check | Anytime-valid confidence sequence on live, paired, non-overlapping blocks |
| 5. Ideas | New features and data | Weekly AI research session + humans | PR review; new features go through levels 2–4 like anything else |
| 6. Operations | Bugs and outages | Daily AI health check | PRs only; frozen records are untouchable (CI-enforced) |

## The weekly sweep (Saturday)
1. Pre-registered experiments not yet run on the current data (re-run automatically when new data
   such as SEC fundamentals becomes available).
2. Queued experiments: `lab/queue/*.json` (moved to `lab/queue/done/` once evaluated).
3. Neighbors of the best learned model: drop one feature family, perturb hyperparameters, change horizon.
4. Holdout tests for new development passers (max 3/week).
5. Registry update and live refits.

Spec format (`lab/queue/my-idea.json`):
```json
{"name": "gbm_events_only", "kind": "gbm", "families": ["events", "reversal"], "horizon": 5,
 "params": {"learning_rate": 0.03}, "notes": "why this might work", "proposedBy": "ai-research-2026-11-08"}
```

## Scheduled Claude sessions
- **Health check** (Tue–Sat 08:50 ET): did the daily cycle run? Is `live/status.json` free of
  `lastError`? Is the chain intact? If broken, diagnose and open a fix PR. Quiet when healthy.
- **Research** (Sunday 10:05 ET): read `lab/ledger.jsonl`, `site/data/lab.json`, `site/data/live.json`;
  propose at most two new ideas grounded in published evidence, implement them as features/specs
  with tests, open a PR. Never touches gates, eras or records.
- **Monthly review** (1st of each month, 11:36 ET): write `docs/reviews/YYYY-MM.md`: what the live evidence says,
  what the lab has tried, whether any rule should change (as a proposal, not a change).

## What will not happen
- No retuning because of a bad week.
- No backtest result is ever shown as live performance.
- No model takes credit for a period before its first frozen forecast.
