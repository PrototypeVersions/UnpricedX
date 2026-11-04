"""Anytime-valid confidence sequences for comparing forecasters.

A fixed-sample test ("check after 60 days") is invalid if you look every day and stop when it
looks good. Confidence sequences stay valid under continuous monitoring: the probability that
the true mean ever leaves the band is at most alpha, at every point in time simultaneously.

Two constructions are provided:
  * `asymptotic_cs` (used for decisions): the Gaussian-mixture asymptotic confidence sequence of
    Waudby-Smith, Arbour, Sinha, Kennedy & Ramdas ("Time-uniform central limit theory and
    asymptotic confidence sequences", Annals of Statistics, 2024). It adapts to the observed
    variance, which matters because weekly ICs sit in a narrow range inside [-1, 1].
  * `eb_confidence_sequence`: the predictable plug-in empirical-Bernstein CS for bounded data
    (Waudby-Smith & Ramdas, JRSS-B 2023). Valid in finite samples but, with the [-1, 1] range,
    so conservative that a realistic edge would take a decade to show; kept for reference.
Choe & Ramdas ("Comparing Sequential Forecasters", Operations Research) use both styles for paired
score differences. Observations are non-overlapping blocks, so they are not double-counted, and
decisions also require a minimum number of blocks so the asymptotics are reasonable.
"""
from __future__ import annotations

import math

import numpy as np


def eb_confidence_sequence(x: np.ndarray, lo: float, hi: float, alpha: float = 0.05, c: float = 0.5) -> dict:
    """Running two-sided CS for the mean of observations bounded in [lo, hi].

    Returns arrays (one entry per observation) of the lower and upper bounds and the running mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "lower": [], "upper": [], "mean": []}
    width = hi - lo
    z = np.clip((x - lo) / width, 0.0, 1.0)
    log_term = math.log(2 / alpha)
    mu_hat_prev, sig2_prev = 0.5, 0.25
    sum_lam, sum_lam_x, sum_v_psi = 0.0, 0.0, 0.0
    run_sum, run_sq = 0.0, 0.0
    lower, upper, mean = [], [], []
    for t in range(1, n + 1):
        xi = z[t - 1]
        lam = min(math.sqrt(2 * log_term / (sig2_prev * t * math.log(1 + t))), c)
        v = 4 * (xi - mu_hat_prev) ** 2
        psi = (-math.log(1 - lam) - lam) / 4
        sum_lam += lam
        sum_lam_x += lam * xi
        sum_v_psi += v * psi
        center = sum_lam_x / sum_lam
        radius = (log_term + sum_v_psi) / sum_lam
        lower.append(lo + width * max(0.0, center - radius))
        upper.append(lo + width * min(1.0, center + radius))
        run_sum += xi
        mu_hat = (0.5 + run_sum) / (t + 1)
        run_sq += (xi - mu_hat) ** 2
        sig2_prev = (0.25 + run_sq) / (t + 1)
        mu_hat_prev = mu_hat
        mean.append(lo + width * (run_sum / t))
    return {"n": n, "lower": lower, "upper": upper, "mean": mean}


def asymptotic_cs(x: np.ndarray, alpha: float = 0.05, t_opt: int = 52, prior_sd: float = 0.1, prior_n: int = 5) -> dict:
    """Running two-sided asymptotic CS for the mean.

    The variance is estimated online and shrunk toward `prior_sd` with `prior_n` pseudo-observations
    so the band is not spuriously narrow in the first few weeks.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "lower": [], "upper": [], "mean": []}
    la = math.log(1 / alpha)
    rho2 = (2 * la + math.log(2 * la + 1)) / t_opt
    t = np.arange(1, n + 1)
    mean = np.cumsum(x) / t
    ss = np.array([np.sum((x[:i] - x[:i].mean()) ** 2) for i in t])
    var = (prior_n * prior_sd**2 + ss) / (prior_n + t - 1)
    radius = np.sqrt(var) * np.sqrt(2 * (t * rho2 + 1) / (t**2 * rho2) * np.log(np.sqrt(t * rho2 + 1) / alpha))
    return {"n": n, "lower": list(mean - radius), "upper": list(mean + radius), "mean": list(mean)}


def compare(ic_challenger: np.ndarray, ic_champion: np.ndarray, alpha: float = 0.05) -> dict:
    """CS for E[IC_challenger - IC_champion] on paired blocks (each IC in [-1, 1])."""
    a, b = np.asarray(ic_challenger, float), np.asarray(ic_champion, float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    cs = asymptotic_cs(d, alpha)
    verdict = "undecided"
    if cs["n"]:
        if cs["lower"][-1] > 0:
            verdict = "challenger better"
        elif cs["upper"][-1] < 0:
            verdict = "challenger worse"
    return {**cs, "verdict": verdict}


def skill(ic: np.ndarray, alpha: float = 0.05) -> dict:
    """CS for one model's mean IC; 'skill shown' once the whole band is above zero."""
    cs = asymptotic_cs(np.asarray(ic, float), alpha)
    verdict = "undecided"
    if cs["n"]:
        verdict = "positive skill" if cs["lower"][-1] > 0 else "negative skill" if cs["upper"][-1] < 0 else "undecided"
    return {**cs, "verdict": verdict}
