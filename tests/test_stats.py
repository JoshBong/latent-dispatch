import numpy as np
import pytest

from latent_dispatch.stats import bootstrap_ci, holm, interaction, mcnemar, sign_flip, tost


def test_mcnemar_exact():
    assert mcnemar([1, 1, 0], [1, 1, 0])["p"] == 1.0
    r = mcnemar([1] * 10 + [0] * 10, [0] * 10 + [0] * 10)   # 10 discordant, all one way
    assert r["n10"] == 10 and r["n01"] == 0 and abs(r["p"] - 2 / 2 ** 10) < 1e-12


def test_bootstrap_ci_covers_the_mean():
    d = np.r_[np.ones(30), np.zeros(70)]
    est, lo, hi = bootstrap_ci(d)
    assert abs(est - 0.3) < 1e-12 and lo < 0.3 < hi


def test_interaction_paired_and_unpaired():
    base, test = np.zeros(200), np.r_[np.ones(100), np.zeros(100)]
    for paired in (True, False):
        est, lo, hi = interaction(base, test, paired=paired)
        assert abs(est - 0.5) < 1e-12 and lo > 0.3


def test_tost():
    assert tost(np.zeros(500))["equivalent"]
    assert not tost(np.r_[np.ones(100), np.zeros(400)])["equivalent"]


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_sign_flip_is_mcnemar_on_one_row_per_question():
    rng = np.random.default_rng(3)
    for n in (40, 400):
        a, b = rng.integers(0, 2, n), rng.integers(0, 2, n)
        assert abs(sign_flip(a - b) - mcnemar(a, b)["p"]) < 1e-12
    a = np.r_[np.ones(17), np.zeros(5)]
    assert abs(sign_flip(a - np.zeros(22)) - 2 / 2 ** 17) < 1e-15           # 17 discordant pairs, all one way
    assert sign_flip(np.zeros(10)) == 1.0


def test_sign_flip_exact_on_averaged_rows_and_monte_carlo_off_grid():
    """Per-source means of EM over 3 seeds sit on a grid of thirds: exact, checked against brute force. Off the
    grid (F1) it falls back to Monte Carlo."""
    from itertools import product
    d = np.array([2, 2, -1, 1, 3, -2]) / 3
    brute = np.mean([abs(np.dot(s, d)) >= abs(d.sum()) - 1e-12 for s in product([-1, 1], repeat=d.size)])
    assert abs(sign_flip(d) - brute) < 1e-12 and abs(brute - 0.4375) < 1e-12
    f1 = np.array([0.31, -0.12, 0.47, 0.05, 0.22, -0.08, 0.36, 0.19, 0.11, -0.27])
    brute = np.mean([abs(np.dot(s, f1)) >= abs(f1.sum()) - 1e-12 for s in product([-1, 1], repeat=f1.size)])
    assert abs(sign_flip(f1) - brute) < 0.01


def test_sign_flip_refuses_nan():
    with pytest.raises(ValueError):
        sign_flip([0.5, float("nan")])
