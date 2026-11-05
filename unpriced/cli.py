"""Command line entry points used by the GitHub workflows and by people.

    python -m unpriced.cli bootstrap          # first full data download
    python -m unpriced.cli daily              # pre-open cycle
    python -m unpriced.cli weekly             # data refresh + research sweep + refits
    python -m unpriced.cli sweep --minutes 30
    python -m unpriced.cli site               # rebuild site/data
    python -m unpriced.cli verify --base origin/main
    python -m unpriced.cli demo               # full offline run on a synthetic market
    python -m unpriced.cli power --ic 0.02    # how long until a live edge of this size is visible?
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback

from .util import dumps, iso, log, now_et, read_json, write_json

L = log("cli")


def _record_error(stage: str, exc: Exception) -> None:
    from .config import paths
    p = paths().live / "status.json"
    st = read_json(p, {})
    st["lastError"] = {"stage": stage, "at": iso(now_et()), "error": f"{type(exc).__name__}: {exc}"}
    write_json(p, st)


def cmd_bootstrap(args) -> None:
    from .data import pipeline
    pipeline.refresh_candidates()
    print(dumps(pipeline.update_prices_job("full")))
    print(dumps(pipeline.update_sec_job("bootstrap")))


def cmd_daily(args) -> None:
    from .live.cycle import run_daily, StaleData
    from .live.snapshot import PublicationWindowMissed
    from . import site_build
    try:
        rep = run_daily(force=args.force)
        print(dumps({k: v for k, v in rep.items() if k not in ("prices",)}))
    except PublicationWindowMissed as exc:
        L.error("forecast window missed: %s", exc)
        _record_error("daily-publication-window-missed", exc)
        site_build.build()
        sys.exit(78)   # a missed deadline is a failure, not a temporary data retry
    except StaleData as exc:
        L.warning("data not ready: %s", exc)
        _record_error("daily-data-not-ready", exc)
        site_build.build()
        sys.exit(75)   # EX_TEMPFAIL: the workflow retries later
    except Exception as exc:
        _record_error("daily", exc)
        traceback.print_exc()
        site_build.build()
        raise
    site_build.build()


def cmd_weekly(args) -> None:
    from .data import pipeline
    from .lab.runner import run_sweep
    from . import site_build
    try:
        if not args.skip_data:
            try:
                pipeline.refresh_candidates()
            except Exception as exc:   # directory outage: keep last list
                L.warning("symbol directory unavailable: %s", exc)
            print(dumps(pipeline.update_prices_job("weekly")))
            try:
                print(dumps(pipeline.update_sec_job("weekly")))
            except Exception as exc:
                L.warning("SEC weekly update failed: %s", exc)
        rep = run_sweep(budget_minutes=args.minutes)
        print(dumps(rep))
    except Exception as exc:
        _record_error("weekly", exc)
        raise
    finally:
        site_build.build()


def cmd_sec(args) -> None:
    from .data import pipeline
    print(dumps(pipeline.update_sec_job("weekly")))


def cmd_sweep(args) -> None:
    from .lab.runner import run_sweep
    print(dumps(run_sweep(budget_minutes=args.minutes, max_new=args.max_new)))


def cmd_site(args) -> None:
    from . import site_build
    site_build.build()


def cmd_verify(args) -> None:
    """Integrity checks: hash chain intact, and append-only files only grew relative to `base`."""
    from .config import paths
    from .live.snapshot import verify_chain
    problems = []
    vc = verify_chain()
    if not vc["ok"]:
        problems += vc["problems"]
    if args.base:
        root = paths().root
        for rel in ["lab/ledger.jsonl", "live/chain.jsonl", "live/scores.jsonl", "live/trades.jsonl"]:
            try:
                old = subprocess.check_output(["git", "show", f"{args.base}:{rel}"], cwd=root, text=True, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue
            new = (root / rel).read_text() if (root / rel).exists() else ""
            if not new.startswith(old):
                problems.append(f"{rel} was edited (append-only file)")
        changed = subprocess.check_output(["git", "diff", "--name-status", args.base, "--", "live/snapshots", "live/llm"],
                                          cwd=root, text=True).splitlines()
        for line in changed:
            status, *rest = line.split("\t")
            if status != "A":
                problems.append(f"frozen record modified: {line}")
    print(json.dumps({"chain": vc, "problems": problems}, indent=2))
    if problems:
        sys.exit(1)


def cmd_demo(args) -> None:
    """Run the whole system offline on a synthetic market in a temporary directory."""
    tmp = args.dir or tempfile.mkdtemp(prefix="unpriced-demo-")
    os.environ["UNPRICED_ROOT"] = tmp
    os.environ["UNPRICED_STORE"] = os.path.join(tmp, "store")
    from .data.synthetic import SyntheticMarket
    from .lab.runner import run_sweep
    from .live.cycle import run_daily
    from . import site_build
    import pandas as pd
    from .calendar import sessions
    m = SyntheticMarket(n=args.stocks, start="2008-01-01", end=args.end, strength=args.strength)
    m.write_store()
    from . import store
    days = sessions(pd.Timestamp(args.end) - pd.Timedelta(days=60), args.end)[-args.days:]
    full = m.tables["prices"]
    store.write("prices", full[full["date"] < days[0]])       # the lab only sees data before the live days
    print(dumps(run_sweep(budget_minutes=args.minutes)))
    for d in days:
        store.write("prices", full[full["date"] <= d])
        os.environ["UNPRICED_NOW"] = (pd.Timestamp(d) + pd.Timedelta(days=1, hours=6, minutes=10)).isoformat()
        rep = run_daily(skip_data=True)
        print(d.date(), rep.get("status"), rep.get("trades"))
    site_build.build()
    print(f"demo written to {tmp}")


def cmd_power(args) -> None:
    """Expected weeks of live data before a confidence sequence can confirm an IC of a given size."""
    import numpy as np
    from .live.sequential import skill
    rng = np.random.default_rng(0)
    res = []
    for _ in range(args.sims):
        x = np.clip(rng.normal(args.ic, args.ic_sd, 520), -1, 1)
        cs = skill(x)
        hit = next((i + 1 for i, lo in enumerate(cs["lower"]) if lo > 0 and i + 1 >= 26), None)
        res.append(hit if hit else np.inf)
    res = np.array(res, dtype=float)
    print(json.dumps({"ic": args.ic, "icSdPerBlock": args.ic_sd, "medianBlocks": float(np.median(res)),
                      "shareConfirmedWithin10y": float(np.mean(np.isfinite(res)))}, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="unpriced")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap").set_defaults(fn=cmd_bootstrap)
    d = sub.add_parser("daily")
    d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_daily)
    w = sub.add_parser("weekly")
    w.add_argument("--minutes", type=float, default=None)
    w.add_argument("--skip-data", action="store_true")
    w.set_defaults(fn=cmd_weekly)
    s = sub.add_parser("sweep")
    s.add_argument("--minutes", type=float, default=None)
    s.add_argument("--max-new", type=int, default=None)
    s.set_defaults(fn=cmd_sweep)
    sub.add_parser("site").set_defaults(fn=cmd_site)
    sub.add_parser("sec").set_defaults(fn=cmd_sec)
    v = sub.add_parser("verify")
    v.add_argument("--base", default=None)
    v.set_defaults(fn=cmd_verify)
    dm = sub.add_parser("demo")
    dm.add_argument("--dir", default=None)
    dm.add_argument("--stocks", type=int, default=250)
    dm.add_argument("--strength", type=float, default=1.0)
    dm.add_argument("--minutes", type=float, default=5)
    dm.add_argument("--days", type=int, default=12)
    dm.add_argument("--end", default="2026-10-30")
    dm.set_defaults(fn=cmd_demo)
    pw = sub.add_parser("power")
    pw.add_argument("--ic", type=float, default=0.02)
    pw.add_argument("--ic-sd", type=float, default=0.16)
    pw.add_argument("--sims", type=int, default=200)
    pw.set_defaults(fn=cmd_power)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
