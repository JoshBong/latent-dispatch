"""Paired statistics for arm comparisons on the same instances. Inputs are 0/1 (EM) or [0, 1] (F1) arrays aligned
by instance.

Rows that share a source question (its 3 seeds, its placements at one crossing level) are not independent, so
analysis.py first averages each source question's differences and the tests below then treat source questions
as the unit. With one row per source question this changes nothing: sign_flip on 0/±1 differences is the
exact McNemar test."""
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


def sign_flip(diffs, n_perm: int = 20_000, seed: int = 0) -> float:
    """Two-sided paired permutation test of mean(diffs) = 0: each unit's difference keeps or flips its sign.

    Exact when the differences sit on a grid of 1/k for some k <= 12: EM averaged per source question, as long as the
    row counts' least common multiple is at most 12 (1, 2, 3, 4 or 6 rows each). With one row each, this is the exact
    McNemar test. Monte Carlo otherwise (e.g. F1)."""
    d = np.asarray(diffs, float)
    if not np.isfinite(d).all():
        raise ValueError("sign_flip got NaN or infinite differences")
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    for k in range(1, 13):
        if np.allclose(np.rint(d * k), d * k, rtol=0, atol=1e-9):
            return _exact_sign_flip(np.rint(d * k).astype(int))
    rng = np.random.default_rng(seed)
    observed, hits = abs(d.sum()), 0
    for start in range(0, n_perm, 1000):           # in chunks, so memory stays small with thousands of units
        signs = rng.choice([-1.0, 1.0], size=(min(1000, n_perm - start), d.size))
        hits += int(np.sum(np.abs(signs @ d) >= observed - 1e-12))
    return (hits + 1) / (n_perm + 1)


def _exact_sign_flip(steps: np.ndarray) -> float:
    """P(|sum of ±steps| >= |observed sum|) with fair independent signs, by convolving over the integer sums."""
    total = int(np.abs(steps).sum())
    prob = np.zeros(2 * total + 1)                 # prob[i] = P(sum = i - total)
    prob[total] = 1.0
    for step in np.abs(steps):
        prob = 0.5 * (np.roll(prob, step) + np.roll(prob, -step))
    sums = np.arange(-total, total + 1)
    return float(min(1.0, prob[np.abs(sums) >= abs(int(steps.sum()))].sum()))


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
