"""Model specifications, fitting and prediction.

A spec is a small JSON-able dict, and its hash is the model's identity: two runs of the same
spec are the same experiment (the ledger refuses to count it twice), and a live model's forecasts
can always be traced to the exact spec and fitted artifact that produced them.

Kinds:
  factor   - one literature feature with its documented sign (pre-registered benchmarks; no fitting)
  ridge    - ridge regression on ranked features -> ranked forward return
  gbm      - histogram gradient boosting with early stopping on a *time-ordered*, purged
             validation block (never a random split, which leaks with overlapping labels)
  ensemble - average of member models' per-date standardized scores
"""
from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

from .features.library import REGISTRY, families
from .util import stable_hash

GBM_DEFAULTS = {"learning_rate": 0.04, "max_leaf_nodes": 31, "min_samples_leaf": 0, "l2_regularization": 1.0,
                "max_iter": 500, "max_features": 0.7, "n_iter_no_change": 25, "validation_dates": 0.15}
RIDGE_DEFAULTS = {"alpha": 50.0}


def spec_id(spec: dict) -> str:
    body = {k: v for k, v in spec.items() if k not in ("name", "notes", "proposedBy")}
    return f"{spec.get('name', spec['kind'])}-{stable_hash(body, 8)}"


def resolve_features(spec: dict) -> list[str]:
    if spec["kind"] == "factor":
        return [spec["factor"]]
    if spec["kind"] == "ensemble":
        out: list[str] = []
        for m in spec["members"]:
            out += [f for f in resolve_features(m) if f not in out]
        return out
    names = list(spec.get("features", []))
    fam = families()
    for f in spec.get("families", []):
        names += [n for n in fam.get(f, []) if n not in names]
    drop = set(spec.get("exclude", []))
    names = [n for n in names if n not in drop]
    missing = [n for n in names if n not in REGISTRY]
    if missing:
        raise ValueError(f"unknown features in spec: {missing}")
    return names


def _zscore_by_date(dates: np.ndarray, x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    g = s.groupby(dates)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()


@dataclass
class Fitted:
    spec: dict
    features: list[str]
    model: object = None
    members: list["Fitted"] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        kind = self.spec["kind"]
        if kind == "factor":
            return float(self.spec.get("sign", REGISTRY[self.spec["factor"]].sign or 1)) * df[self.features[0]].fillna(0.0).to_numpy()
        if kind == "ensemble":
            dates = df["date"].to_numpy()
            parts = [_zscore_by_date(dates, m.predict(df)) for m in self.members]
            w = self.spec.get("weights") or [1.0] * len(parts)
            return np.sum([wi * p for wi, p in zip(w, parts)], axis=0) / sum(w)
        X = df[self.features].to_numpy(dtype="float32")
        if kind == "ridge":
            X = np.nan_to_num(X, nan=0.0)
        return self.model.predict(X)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "Fitted":
        with path.open("rb") as fh:
            return pickle.load(fh)


def fit(spec: dict, train: pd.DataFrame, seed: int = 0, cache: dict | None = None) -> Fitted:
    """Fit a spec. `cache` (optional) reuses identical member fits on identical training data,
    so an ensemble evaluated after its members does not refit them."""
    kind = spec["kind"]
    feats = resolve_features(spec)
    if kind == "factor":
        return Fitted(spec, feats)
    if cache is not None:
        key = (spec_id(spec), len(train), str(train["date"].min()), str(train["date"].max()))
        if key in cache:
            return cache[key]
    if kind == "ensemble":
        members = [fit(m, train, seed, cache) for m in spec["members"]]
        return Fitted(spec, feats, members=members, info={"rows": max(int(x.info.get("rows", 0)) for x in members)})

    h = int(spec.get("horizon", 5))
    target = f"target_{h}"
    data = train.dropna(subset=[target])
    # drop features that are essentially unavailable in this window (e.g. no SEC data yet)
    avail = [f for f in feats if data[f].notna().mean() > 0.02]
    if not avail:
        raise ValueError("no usable features in training window")
    y = data[target].to_numpy(dtype="float64")
    if kind == "ridge":
        p = {**RIDGE_DEFAULTS, **spec.get("params", {})}
        X = np.nan_to_num(data[avail].to_numpy(dtype="float32"), nan=0.0)
        model = Ridge(alpha=float(p["alpha"])).fit(X, y)
        out = Fitted(spec, avail, model, info={"rows": len(data), "coef": dict(zip(avail, np.round(model.coef_, 5)))})
        if cache is not None:
            cache[key] = out
        return out
    if kind == "gbm":
        p = {**GBM_DEFAULTS, **spec.get("params", {})}
        dates = np.sort(data["date"].unique())
        n_val = max(20, int(len(dates) * float(p["validation_dates"])))
        cut = pd.Timestamp(dates[-n_val])
        purge_before = cut - pd.tseries.offsets.BDay(h + 2)   # labels of these rows would overlap validation
        tr = data[data["date"] < purge_before]
        va = data[data["date"] >= cut]
        msl = int(p["min_samples_leaf"]) or max(200, len(tr) // 400)
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=float(p["learning_rate"]), max_leaf_nodes=int(p["max_leaf_nodes"]),
            min_samples_leaf=msl, l2_regularization=float(p["l2_regularization"]), max_iter=int(p["max_iter"]),
            max_features=float(p["max_features"]), early_stopping=True, n_iter_no_change=int(p["n_iter_no_change"]),
            scoring="loss", random_state=seed,
        )
        model.fit(tr[avail].to_numpy(dtype="float32"), tr[target].to_numpy(dtype="float64"),
                  X_val=va[avail].to_numpy(dtype="float32"), y_val=va[target].to_numpy(dtype="float64"))
        out = Fitted(spec, avail, model, info={"rows": len(tr), "valRows": len(va), "iterations": int(model.n_iter_),
                                              "minSamplesLeaf": msl})
        if cache is not None:
            cache[key] = out
        return out
    raise ValueError(f"unknown model kind {kind}")


@dataclass
class Calibration:
    """Maps a model's within-day score percentile to P(beat the day's median) and expected excess return."""
    x: list
    p: list
    excess_bins: list

    def prob(self, pct: np.ndarray) -> np.ndarray:
        return np.interp(pct, self.x, self.p)

    def excess(self, pct: np.ndarray) -> np.ndarray:
        bins = np.clip((np.asarray(pct) * len(self.excess_bins)).astype(int), 0, len(self.excess_bins) - 1)
        return np.asarray(self.excess_bins)[bins]

    def to_dict(self) -> dict:
        return {"x": self.x, "p": self.p, "excessBins": self.excess_bins}

    @staticmethod
    def from_dict(d: dict) -> "Calibration":
        return Calibration(d["x"], d["p"], d["excessBins"])


def calibrate(oos: pd.DataFrame, h: int, n_bins: int = 10) -> Calibration:
    """Fit on out-of-sample predictions only (walk-forward output)."""
    d = oos.dropna(subset=[f"fwd_{h}"]).copy()
    d["pct"] = d.groupby("date")["pred"].rank(pct=True)
    med = d.groupby("date")[f"fwd_{h}"].transform("median")
    mean = d.groupby("date")[f"fwd_{h}"].transform("mean")
    d["beat"] = (d[f"fwd_{h}"] > med).astype(float)
    d["excess"] = d[f"fwd_{h}"] - mean
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    sample = d.sample(min(len(d), 400_000), random_state=0)
    iso.fit(sample["pct"], sample["beat"])
    grid = np.linspace(0, 1, 21)
    probs = iso.predict(grid)
    bins = pd.cut(d["pct"], np.linspace(0, 1, n_bins + 1), include_lowest=True, labels=False)
    excess = d.groupby(bins)["excess"].mean().reindex(range(n_bins)).fillna(0.0)
    return Calibration([float(x) for x in grid], [float(round(p, 5)) for p in probs], [float(round(e, 6)) for e in excess])


def with_horizon(spec: dict, h: int) -> dict:
    s = copy.deepcopy(spec)
    s["horizon"] = h
    for m in s.get("members", []):
        m["horizon"] = h
    return s
