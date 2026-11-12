"""Real weights, end to end. Default Qwen3-0.6B; run any model in the matrix before its first experiment:

    pytest -m slow tests/test_real_model.py                                         # Qwen3-0.6B, CPU fp32
    MODEL=meta-llama/Llama-3.1-8B-Instruct DTYPE=bfloat16 pytest -m slow tests/test_real_model.py    # on a GPU

Two kinds of check. Exactness: K0 with one worker must reproduce B1, and a shifted cache must match a cache encoded
at the shifted position. Not soup: on six invented facts (so closed book can't know them), B1 and K0 must read the
fact back, answers must stop on their own (stop tokens right), and K1's latent steps must not wreck the output (they
did on Llama-3.2-3B in Cheng, Das & Ramnath 2026). The thresholds are loose: a broken prompt, cache or stop token
scores near 0, not 4/6.
"""
import os
import re
from collections import Counter

import pytest
import torch

from latent_dispatch import arms, kv, toy
from latent_dispatch.model import Model
from latent_dispatch.score import score

pytestmark = pytest.mark.slow

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DTYPE = os.environ.get("DTYPE", "bfloat16" if DEVICE == "cuda" else "float32")

FACTS = [
    ("The capital of Velmora is Kastrel.", "What is the capital of Velmora?", ["Kastrel"]),
    ("Ines Farrow's favorite fruit is the quince.", "What is Ines Farrow's favorite fruit?", ["quince"]),
    ("The Orinth Bridge was finished in 1873.", "In what year was the Orinth Bridge finished?", ["1873"]),
    ("Tomas Reddy plays the bassoon in the Sundmark orchestra.", "Which instrument does Tomas Reddy play?",
     ["bassoon", "the bassoon"]),
    ("The river Ossory flows into Lake Brenn.", "Which lake does the river Ossory flow into?", ["Lake Brenn", "Brenn"]),
    ("Priya Castellan was born in Quillford.", "Where was Priya Castellan born?", ["Quillford"]),
]
DISTRACTOR = "Marco Duval works as a baker in Tarnova. The mayor of Ashgrove is Celia Vance."


def _fact_questions(placement):
    """BB: the fact sits in the second worker's block, so K0 must read keys it shifted. AA: in the first block."""
    out = []
    for i, (fact, question, answers) in enumerate(FACTS):
        workers = [[fact], [DISTRACTOR]] if placement == "AA" else [[DISTRACTOR], [fact]]
        out.append(dict(id=f"fact{i}:{placement}", dataset="facts", source_id=f"fact{i}", question=question,
                        answers=answers, qtype="hop0", hops=1, placement=placement, crossings=0, seed=0,
                        workers=workers, gold=[{"hop": 0, "worker": 0 if placement == "AA" else 1, "index": 0}],
                        bridges=[], decomposition=[]))
    return out


@pytest.fixture(scope="module")
def real():
    return Model.load(MODEL, device=DEVICE, dtype=DTYPE)


def _em(model, arm, qs, **options):
    outs = [arm(model, q, **options) for q in qs]
    return sum(score(o["text"], q["answers"])[1] for o, q in zip(outs, qs)), outs


def _looping(text, n=3, times=4):
    words = re.findall(r"\w+", text.lower())
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return bool(grams) and max(grams.values()) >= times


# ---- exactness

def test_k0_one_worker_matches_b1_text_on_real_weights(real):
    for q in toy.questions(1, 2, 0, ["AB"]):
        one = {**q, "workers": [[p for w in q["workers"] for p in w]]}
        assert arms.b1(real, one)["text"] == arms.k0(real, one)["text"]


def test_shift_matches_native_encoding_on_real_weights(real):
    ids = real.ids("The mayor of Velmora is Bram Holt. The river the Calder flows through Irvale.")
    at_p, _ = real.prefill(ids, start_pos=10)
    at_pd, _ = real.prefill(ids, start_pos=1210)
    shifted = kv.layers(kv.shift(at_p, 1200, real.inv_freq))[0][0].float()      # layer-0 keys
    native = kv.layers(at_pd)[0][0].float()
    rel = ((shifted - native).norm() / native.norm()).item()
    assert rel < (6e-3 if real.dtype_name == "bfloat16" else 1e-4), rel


# ---- not soup

def test_closed_book_cannot_know_the_invented_facts(real):
    em, _ = _em(real, arms.b0, _fact_questions("BB"))
    assert em <= 1


@pytest.mark.parametrize("arm,placement", [(arms.b1, "BB"), (arms.k0, "AA"), (arms.k0, "BB"), (arms.b1_mask, "BB")])
def test_reads_the_fact_back(real, arm, placement):
    em, outs = _em(real, arm, _fact_questions(placement), max_answer_tokens=256)
    stopped = sum(o["gen_tokens"] - o["think_tokens"] < 256 for o in outs)
    assert em >= 4, [o["text"] for o in outs]
    assert stopped >= 5, "answers ran to the cap: stop tokens wrong?"


def test_k1_latent_steps_do_not_wreck_the_output(real):
    em, outs = _em(real, arms.k1, _fact_questions("BB"), m=8, max_answer_tokens=256)
    assert not any(_looping(o["text"]) for o in outs), [o["text"] for o in outs]
    assert em >= 3, [o["text"] for o in outs]


@pytest.mark.parametrize("arm,options", [
    (arms.b0, {}), (arms.b1, {}), (arms.b1_mask, {}), (arms.k0, {}), (arms.k0, {"merge": "shared"}),
    (arms.k0, {"visible": [0]}), (arms.k1, {"m": 4}), (arms.t1, {"report_tokens": 64}), (arms.scan, {"rounds": 2}),
    (arms.b1, {"thinking": "forced", "think_budget": 32}),
])
def test_every_arm_runs_and_scores(real, arm, options):
    if options.get("thinking") == "forced" and not real.fmt.thinking:
        pytest.skip(f"{real.fmt.name} has no thinking mode")
    q = toy.questions(0, 1, 0, ["AB"])[0]
    out = arm(real, q, **options)
    pred, em, f1 = score(out["text"], q["answers"])
    assert isinstance(pred, str) and out["gen_tokens"] > 0
    if options.get("thinking") == "forced":
        assert out["think_tokens"] <= 32
