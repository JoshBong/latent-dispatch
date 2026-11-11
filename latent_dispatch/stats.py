"""Paired statistics for arm comparisons on the same instances. Inputs are 0/1 (EM) or [0, 1] (F1) arrays aligned
by instance."""
from math import comb

import numpy as np


def mcnemar(a, b) -> dict:
    """Exact two-sided McNemar on paired 0/1 outcomes."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n01": n01, "n10": n10, "p": p}


def bootstrap_ci(diffs, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-instance differences with a percentile bootstrap CI."""
    d = np.asarray(diffs, float)
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def interaction(diffs_base, diffs_test, paired: bool, n_boot: int = 10_000, alpha: float = 0.05,
                seed: int = 0) -> tuple[float, float, float]:
    """Difference of gaps: mean(diffs_test) - mean(diffs_base), with a bootstrap CI.

    paired=True: the two arrays are aligned by question (same question at two placements); resample questions
    jointly. paired=False: different questions; resample each group on its own."""
    b, t = np.asarray(diffs_base, float), np.asarray(diffs_test, float)
    rng = np.random.default_rng(seed)
    if paired:
        if b.shape != t.shape:
            raise ValueError("paired interaction needs aligned arrays")
        return bootstrap_ci(t - b, n_boot, alpha, seed)
    est = t.mean() - b.mean()
    boots = (t[rng.integers(0, t.size, (n_boot, t.size))].mean(axis=1)
             - b[rng.integers(0, b.size, (n_boot, b.size))].mean(axis=1))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


def tost(diffs, margin: float = 0.02, alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Equivalence within ±margin (0.02 = 2 EM points): the (1 - 2·alpha) bootstrap CI must sit inside the margin."""
    est, lo, hi = bootstrap_ci(diffs, n_boot, 2 * alpha, seed)
    return {"diff": est, "lo": lo, "hi": hi, "equivalent": bool(-margin < lo and hi < margin)}


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj
