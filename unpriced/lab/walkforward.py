"""Purged walk-forward evaluation.

For each test block of `refit` sessions the model is fit only on rows whose labels were fully
realized before the block starts (a purge of horizon+1 sessions), sampled every `stride`
sessions to limit overlap. Nothing from the test block or after it can reach the fit.

Eras: the *development* era is used for all exploration. The *holdout* era (from
lab.holdoutStart) is sealed; each holdout evaluation is recorded in the ledger and capped per
week, so it cannot quietly become a second development set.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..config import section
from ..models import fit as fit_model, Fitted
from ..util import log

L = log("walkforward")


class PanelIndex:
    """Fast date-slicing for a date-sorted panel."""

    def __init__(self, panel: pd.DataFrame) -> None:
        self.panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        d = self.panel["date"].to_numpy()
        self.dates = np.unique(d)
        self.starts = np.searchsorted(d, self.dates, side="left")
        self.ends = np.searchsorted(d, self.dates, side="right")

    def rows(self, dates: np.ndarray) -> pd.DataFrame:
        pos = np.searchsorted(self.dates, dates)
        pos = pos[(pos < len(self.dates)) & (self.dates[np.minimum(pos, len(self.dates) - 1)] == dates)]
        if len(pos) == 0:
            return self.panel.iloc[0:0]
        idx = np.concatenate([np.arange(self.starts[p], self.ends[p]) for p in pos])
        return self.panel.iloc[idx]


def era_bounds(era: str, idx: PanelIndex, h: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = section("lab")
    dates = idx.dates
    first_ok = pd.Timestamp(cfg["trainStart"])
    if era == "dev":
        end = pd.Timestamp(cfg["devEnd"])
        # drop the last h+1 dev dates: their labels would peek into the sealed holdout
        dev_dates = dates[dates <= np.datetime64(end)]
        end = pd.Timestamp(dev_dates[-(h + 2)]) if len(dev_dates) > h + 2 else end
        start_pos = np.searchsorted(dates, np.datetime64(first_ok)) + int(cfg["minTrainDates"])
        start = pd.Timestamp(dates[min(start_pos, len(dates) - 1)])
        return start, end
    if era == "holdout":
        return pd.Timestamp(cfg["holdoutStart"]), pd.Timestamp(dates[-1])
    if era == "all":
        start_pos = np.searchsorted(dates, np.datetime64(first_ok)) + int(cfg["minTrainDates"])
        return pd.Timestamp(dates[min(start_pos, len(dates) - 1)]), pd.Timestamp(dates[-1])
    raise ValueError(era)


def training_dates(idx: PanelIndex, before: pd.Timestamp, h: int, stride: int) -> np.ndarray:
    cfg = section("lab")
    dates = idx.dates
    pos = int(np.searchsorted(dates, np.datetime64(before)))
    last = pos - (h + 1)                       # label of date d needs opens through d+1+h < block start
    first = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(cfg["trainStart"]))))
    if last <= first:
        return dates[:0]
    cand = np.arange(first, last)
    cand = cand[cand % stride == 0]            # fixed phase -> same rows in every run
    return dates[cand]


FIT_CACHE: dict = {}   # member fits reused within one process (e.g. a sweep), keyed by spec and training window


def walk_forward(idx: PanelIndex, spec: dict, era: str, refit: int | None = None, stride: int | None = None,
                 keep_fits: bool = False) -> tuple[pd.DataFrame, list[dict], list[Fitted]]:
    cfg = section("lab")
    h = int(spec.get("horizon", 5))
    refit = int(refit or cfg["refitEverySessions"])
    stride = int(stride or cfg["trainEveryNthDate"])
    start, end = era_bounds(era, idx, h)
    test_dates = idx.dates[(idx.dates >= np.datetime64(start)) & (idx.dates <= np.datetime64(end))]
    outs, log_rows, fits = [], [], []
    if len(FIT_CACHE) > 400:
        FIT_CACHE.clear()
    for b in range(0, len(test_dates), refit):
        block = test_dates[b : b + refit]
        tdates = training_dates(idx, pd.Timestamp(block[0]), h, stride)
        if len(tdates) * stride < int(cfg["minTrainDates"]) * 0.5:
            continue
        t0 = time.time()
        train = idx.rows(tdates)
        model = fit_model(spec, train, cache=FIT_CACHE)
        test = idx.rows(block)
        pred = model.predict(test)
        keep = ["date", "ticker", "liq_rank", "tradable"] + [c for c in test.columns if c.startswith(("fwd_", "bench_"))]
        o = test[keep].copy()
        o["pred"] = pred.astype("float32")
        outs.append(o)
        log_rows.append({"blockStart": pd.Timestamp(block[0]).date().isoformat(), "trainRows": int(len(train)),
                         "seconds": round(time.time() - t0, 1), **{k: v for k, v in model.info.items() if k != "coef"}})
        if keep_fits:
            fits.append(model)
    oos = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    return oos, log_rows, fits
