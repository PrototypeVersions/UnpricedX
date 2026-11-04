# Working on Unpriced (read before changing anything)

Unpriced is a research lab plus a public forward record for cross-sectional US stock forecasts.
Its value depends entirely on the record being honest. These rules apply to people and to AI
sessions (including the scheduled health-check, research and review sessions).

## Never
- Edit, delete or regenerate anything under `live/snapshots/`, `live/llm/`, or rewrite lines in
  `live/chain.jsonl`, `live/scores.jsonl`, `live/trades.jsonl`, `lab/ledger.jsonl`. They are
  append-only. CI (`python -m unpriced.cli verify --base <ref>`) fails if they change.
- Backfill a missed day. A missed forecast stays missed.
- Change `lab.devEnd` / `lab.holdoutStart`, the gates, or the promotion rule to make a result pass.
  If a rule is wrong, propose the change in a PR with the reason, and record it in `docs/DECISIONS.md`.
- Evaluate on the holdout outside `lab/runner.py` (every look must be logged and capped).
- Put secrets or the owner's email in the repo (it is public). They live in GitHub secrets.

## Safe ways to improve the system
- **New idea** → add a feature in `unpriced/features/library.py` (backward-looking only; add a test),
  and/or drop an experiment spec JSON in `lab/queue/`. The Saturday sweep evaluates it, it counts as a
  trial, and it must pass the same gates as everything else.
- **Bug fix** → normal PR with a test. If the bug affected live forecasts, say so in `docs/DECISIONS.md`;
  do not touch the frozen files.
- **Data fix** → change the pipeline; the store is regenerable (`ops/requests/{"command":"bootstrap"}`).

## How to run things
- Tests: `python -m pytest -q tests` (offline; synthetic market).
- Full offline run: `python -m unpriced.cli demo` (writes to a temp dir, never to `live/`).
- Trigger a workflow without the Actions UI: commit `ops/requests/<name>.json` with
  `{"command": "weekly"|"sweep"|"daily"|"bootstrap"|"site", "minutes": N}`; the log tail appears in `ops/done/`.

## Map
- `unpriced/data/` prices (Yahoo), symbol list (Nasdaq Trader), SEC EDGAR, fundamentals, industries
- `unpriced/features/` feature library, point-in-time context, ranked panel + next-open labels
- `unpriced/lab/` walk-forward, metrics (IC, Newey-West, deflated Sharpe), ledger, specs, sweep
- `unpriced/live/` daily cycle, snapshots + hash chain, scoring, confidence sequences, promotion, portfolio
- `unpriced/llm/` Claude filing reader (forward-only)
- `site/` static site; `unpriced/site_build.py` writes `site/data/*.json`
- `docs/` methodology, learning loop, data, decisions
