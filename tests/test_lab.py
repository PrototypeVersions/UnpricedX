import numpy as np
import pandas as pd

from unpriced.features.library import REGISTRY
from unpriced.features.panel import build_panel
from unpriced.lab.metrics import daily_ic, deflated_sharpe, newey_west_t
from unpriced.lab.walkforward import PanelIndex, training_dates, walk_forward


def test_training_rows_are_purged(ctx):
    panel = build_panel(ctx, ["ret_1w", "mom_12_1"])
    idx = PanelIndex(panel)
    block = pd.Timestamp(idx.dates[1500])
    for h in (1, 5, 21):
        td = training_dates(idx, block, h, 5)
        last = np.searchsorted(idx.dates, td[-1])
        start = np.searchsorted(idx.dates, np.datetime64(block))
        assert last + 1 + h < start + 1, "a training label overlaps the test block"


def test_walk_forward_recovers_planted_reversal(ctx):
    panel = build_panel(ctx, list(REGISTRY))
    idx = PanelIndex(panel)
    spec = {"name": "t", "kind": "ridge", "families": ["reversal", "momentum"], "horizon": 5}
    oos, blocks, _ = walk_forward(idx, spec, "all")
    ic = daily_ic(oos, "pred", "fwd_5")
    assert ic.mean() > 0.005
    assert newey_west_t(ic.values, 5) > 2


def test_gbm_early_stopping_uses_time_split(ctx):
    from unpriced.models import fit
    panel = build_panel(ctx, list(REGISTRY))
    m = fit({"name": "g", "kind": "gbm", "families": ["reversal"], "horizon": 5}, panel.dropna(subset=["target_5"]))
    assert m.info["iterations"] < 500     # stops on honest validation instead of running to max_iter


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.002, 0.02, 600))
    one = deflated_sharpe(r, 1)["dsr"]
    many = deflated_sharpe(r, 500, 0.003)["dsr"]
    assert many < one


def test_ic_bar_rises_with_trials():
    from unpriced.lab.runner import ic_threshold
    assert ic_threshold(5) == 3.0
    assert ic_threshold(1000) > ic_threshold(100) > 3.0
