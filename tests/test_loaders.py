"""MuSiQue and HotpotQA loaders on the real datasets (downloads them if not cached). Run with: pytest -m slow"""
import pytest

from latent_dispatch import data

pytestmark = pytest.mark.slow


def _check(qs):
    for q in qs:
        assert len({len(w) for w in q["workers"]}) == 1                     # balanced paragraph counts
        assert [g["hop"] for g in q["gold"]] == list(range(q["hops"]))
        assert [q["placement"][g["hop"]] for g in q["gold"]] == [chr(65 + g["worker"]) for g in q["gold"]]


def test_musique_placements():
    qs = data.musique_questions(0, {2: ["AB", "BB"], 3: ["ABA"]}, limit_per_type=5)
    _check(qs)
    raw = {it["id"]: it for it in data.musique_items()}
    for q in qs:
        assert [q["workers"][g["worker"]][g["index"]] for g in q["gold"]] == raw[q["source_id"]]["gold"]
        assert q["crossings"] == {"AB": 1, "BB": 0, "ABA": 2}[q["placement"]]


def test_hotpotqa_placements():
    qs = data.hotpotqa_questions(0, ["AB", "BB"], limit_per_type=5)
    _check(qs)
    assert {q["qtype"] for q in qs} == {"bridge", "comparison"}
