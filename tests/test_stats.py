import numpy as np

from latent_dispatch.stats import bootstrap_ci, holm, interaction, mcnemar, tost


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
