"""Pre-registered trading-rule trials (docs/DECISIONS.md 2026-11-23)."""
import json

import numpy as np
import pandas as pd


def _frame(seed=3, n=300, days=400):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=days)
    sig = rng.normal(size=n)
    rows = []
    for d in dates:
        sig = 0.9 * sig + 0.44 * rng.normal(size=n)
        fwd = 0.004 * sig + rng.normal(0, 0.04, n)
        fwd[rng.random(n) < 0.01] = np.nan
        bench = rng.normal(0.002, 0.02)
        for i in range(n):
            rows.append((d, f"T{i}", sig[i], fwd[i], bench, i % 600 + 1, (i % 7) != 0, i % 12))
    return pd.DataFrame(rows, columns=["date", "ticker", "pred", "fwd_5", "bench_5", "liq_rank", "tradable", "sector"])


def test_current_rule_results_do_not_change():
    from unpriced.lab.metrics import long_only
    o = long_only(_frame(), "pred", 5)
    # numbers computed with the code before trading-rule parameters were added
    assert len(o) == 80
    assert abs(o["net"].sum() - 0.4776933149) < 1e-9
    assert abs(o["turnover"].sum() - 35.9) < 1e-9
    assert abs(o["excess"].sum() - 0.2285866849) < 1e-9
    pd.testing.assert_frame_equal(o, long_only(_frame(), "pred", 5, trade_every=1))


def test_trade_every_two_keeps_holdings_between_trades():
    from unpriced.lab.metrics import long_only
    base = long_only(_frame(), "pred", 5)
    slow = long_only(_frame(), "pred", 5, trade_every=2)
    assert len(slow) == len(base)
    assert slow["turnover"].mean() < base["turnover"].mean()
    # in-between rebalances only sell names that left the tradable universe: turnover there is small
    assert slow["turnover"].iloc[1::2].mean() < 0.1 * slow["turnover"].iloc[2::2].mean() + 1e-9


def test_rule_parameters_are_whitelisted():
    import pytest
    from unpriced.lab.runner import rule_kwargs
    assert rule_kwargs({"params": {"holdings": 20, "exitRank": 80, "maxPerSector": 6, "tradeEvery": 2}}) == \
        {"n": 20, "exit_rank": 80, "max_per_sector": 6, "trade_every": 2}
    with pytest.raises(ValueError):
        rule_kwargs({"params": {"noTradeBand": 0.5}})


def test_rule_gates_use_existing_thresholds():
    from unpriced.lab.runner import rule_gates
    ok = {"longOnly": {"excessAnnual": 0.02, "deflated": {"psr": 0.96}}}
    weak = {"longOnly": {"excessAnnual": 0.02, "deflated": {"psr": 0.90}}}
    assert rule_gates(ok, "dev")["passed"] and not rule_gates(weak, "dev")["passed"]
    assert rule_gates({"longOnly": {"excessAnnual": 0.001}}, "holdout")["passed"]
    assert not rule_gates({"longOnly": {"excessAnnual": -0.001}}, "holdout")["passed"]


def test_registered_rules_are_the_four_declared():
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "lab" / "rules"
    names = sorted(json.loads(f.read_text())["name"] for f in d.glob("rule_*.json"))
    assert names == ["rule_exit80", "rule_hold10", "rule_hold10_exit80", "rule_n20"]


def test_run_rules_logs_trials_once_and_leaves_models_alone(ctx, tmp_path, monkeypatch):
    from pathlib import Path
    from unpriced.lab import ledger
    from unpriced.lab.runner import run_rules
    monkeypatch.setenv("UNPRICED_ROOT", str(tmp_path))
    (tmp_path / "lab" / "rules").mkdir(parents=True)
    src = Path(__file__).resolve().parents[1] / "lab" / "rules"
    for f in src.glob("rule_*.json"):
        (tmp_path / "lab" / "rules" / f.name).write_text(f.read_text())
    spec = {"name": "ridge_rev", "kind": "ridge", "families": ["reversal", "momentum"], "horizon": 5}
    (tmp_path / "lab" / "registry.json").write_text(json.dumps(
        {"champion": "ridge_rev-x", "championStatus": "provisional", "history": [],
         "models": {"ridge_rev-x": {"spec": spec, "role": "champion"}}}))
    before = ledger.dev_trials()[0]
    rep = run_rules(ctx)
    lines = ledger.entries()
    assert len(lines) == 4 and all(e["status"] == ledger.RULE_STATUS and e["era"] == "dev" for e in lines)
    assert ledger.dev_trials()[0] == before + 4                  # every rule is a trial
    assert ledger.current("dev") == {}                            # but never a model (registry, site, search)
    assert len(rep["dev"]) == 4 and "dev" in rep["baseline"]
    assert (tmp_path / "lab" / "rules" / "last_run.json").exists()
    run_rules(ctx)
    assert len(ledger.entries()) == 4                             # evaluated once per data signature
