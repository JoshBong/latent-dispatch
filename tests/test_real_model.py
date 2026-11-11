"""Real Qwen3-0.6B weights (same architecture and tokenizer as Qwen3-4B). Run with: pytest -m slow"""
import pytest
import torch

from latent_dispatch import arms, toy
from latent_dispatch.model import Model
from latent_dispatch.score import score

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real():
    return Model.load("Qwen/Qwen3-0.6B", device="cpu", dtype=torch.float32)


def test_k0_one_worker_matches_b1_text_on_real_weights(real):
    for q in toy.questions(1, 2, 0, ["AB"]):
        one = {**q, "workers": [[p for w in q["workers"] for p in w]]}
        assert arms.b1(real, one)["text"] == arms.k0(real, one)["text"]


@pytest.mark.parametrize("arm,options", [
    (arms.b0, {}), (arms.b1, {}), (arms.b1_mask, {}), (arms.k0, {}), (arms.k0, {"merge": "shared"}),
    (arms.k0, {"visible": [0]}), (arms.k1, {"m": 4}), (arms.t1, {"report_tokens": 64}), (arms.scan, {"rounds": 2}),
    (arms.b1, {"thinking": "forced", "think_budget": 32}),
])
def test_every_arm_runs_and_scores(real, arm, options):
    q = toy.questions(0, 1, 0, ["AB"])[0]
    out = arm(real, q, **options)
    pred, em, f1 = score(out["text"], q["answers"])
    assert isinstance(pred, str) and out["gen_tokens"] > 0
    if options.get("thinking") == "forced":
        assert out["think_tokens"] <= 32
