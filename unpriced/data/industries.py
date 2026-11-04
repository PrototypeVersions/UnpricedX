"""Point-in-time statistical industries.

Every `refresh` sessions, stocks in the research universe are clustered on the shape of their
past year of market-residual returns (PCA + k-means). The grouping is used only for dates
strictly after the refresh date, so it never uses future information. This replaces static SIC
codes, which are today's classification applied to the past (and need SEC access).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    # k-means++ initialisation
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        centers.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(centers)
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab, c


def refresh_positions(dates: pd.DatetimeIndex, window: int) -> list[int]:
    """Calendar-anchored refreshes (first session of January and July) so that a research run over
    full history and a live run over a recent slice produce identical groupings."""
    s = pd.Series(np.arange(len(dates)), index=dates)
    firsts = s.groupby([dates.year, (dates.month - 1) // 6]).min()
    return [int(p) for p in firsts.values if p >= window]


def statistical_industries(returns: pd.DataFrame, market: pd.Series, member: pd.DataFrame,
                           window: int = 252, comps: int = 12, per_cluster: int = 25) -> pd.DataFrame:
    """Date x ticker matrix of integer industry ids (NaN when unknown)."""
    dates = returns.index
    out = pd.DataFrame(np.nan, index=dates, columns=returns.columns, dtype="float32")
    resid_all = returns.sub(market.reindex(dates), axis=0)
    starts = refresh_positions(dates, window)
    for n, pos in enumerate(starts):
        end_pos = starts[n + 1] if n + 1 < len(starts) else len(dates)
        rng = np.random.default_rng(int(dates[pos].strftime("%Y%m%d")))
        win = resid_all.iloc[pos - window : pos]
        cols = [c for c in win.columns if member.iloc[pos - 1].get(c, False) and win[c].notna().mean() > 0.8]
        if len(cols) < 4 * per_cluster:
            continue
        z = win[cols].fillna(0.0)
        z = (z - z.mean()) / z.std().replace(0, np.nan)
        z = z.fillna(0.0).to_numpy().T                           # stocks x days
        u, s, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
        emb = u[:, :comps] * s[:comps]
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        k = max(4, len(cols) // per_cluster)
        lab, _ = _kmeans(emb, k, rng)
        block = dates[pos:end_pos]                               # applies strictly after the window
        out.loc[block, cols] = np.broadcast_to(lab.astype("float32"), (len(block), len(cols)))
    return out
