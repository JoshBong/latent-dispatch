import hashlib
import random
from pathlib import Path

import pytest

from latent_dispatch import data, toy


def test_crossings():
    assert [data.crossings(p) for p in ("AA", "BB", "AB", "AAB", "ABA", "ABAB", "AABB")] == [0, 0, 1, 1, 2, 3, 1]


def test_place_balances_and_records_gold():
    gold, rest = ["g0", "g1", "g2"], [f"d{i}" for i in range(17)]
    workers, refs = data.place(gold, rest, "ABA", 2, random.Random(0))
    assert [len(w) for w in workers] == [10, 10]
    assert sorted(p for w in workers for p in w) == sorted(gold + rest)
    assert [(r["hop"], r["worker"]) for r in refs] == [(0, 0), (1, 1), (2, 0)]
    assert all(workers[r["worker"]][r["index"]] == gold[r["hop"]] for r in refs)
    with pytest.raises(ValueError):
        data.place(["g0", "g1"], [], "AA", 2, random.Random(0))       # 2 gold on a worker whose quota is 1


def test_toy_distractor_pools_are_disjoint():
    assert not set(toy.PEOPLE) & set(toy.X_PEOPLE) and not set(toy.CITIES) & set(toy.X_CITIES)
    assert not set(toy.MAYORS) & set(toy.X_MAYORS) and not set(toy.RIVERS) & set(toy.X_RIVERS)


def test_toy_questions():
    for q in toy.questions(2, 20, 0, ["AB", "BA", "AA", "BB"], n_distract=2):
        text = " ".join(p for w in q["workers"] for p in w)
        assert q["answers"][0] in text and all(b in text for b in q["bridges"])
        assert q["crossings"] == (0 if q["placement"] in ("AA", "BB") else 2)
        assert [len(w) for w in q["workers"]] == [2, 2]
        assert {g["worker"] for g in q["gold"]} == {"AA": {0}, "BB": {1}}.get(q["placement"], {0, 1})
    assert all(q["hops"] == 1 and q["crossings"] == 0 for q in toy.questions(0, 4, 0, ["AB"]))


def test_toy_matches_old_v2_generator_without_distractors():
    """n_distract = 0 reproduces the 2026-09-30 v2 items (golds recorded from Paper A toy_gen2.gen_items, seed 0)."""
    old = {1: [("Dev Anand", "Elin Ruud"), ("Ines Farrow", "Mira Solano"), ("Nora Halvorsen", "Jora Lind")],
           2: [("Tomas Reddy", "Bellhaven"), ("Sana Ilori", "Kastrel"), ("Felix Brandt", "Peldane")]}
    for hop, expected in old.items():
        assert [(x["chain"][0], x["gold"]) for x in toy.items(hop, 3, seed=0)] == expected


def test_per_cell_and_filters(tmp_path, monkeypatch):
    qs = toy.questions(1, 5, 0, ["AB", "BB"], n_distract=2)
    monkeypatch.setattr(data, "BENCHMARK", tmp_path)
    data.write(qs, tmp_path / "t.jsonl")
    assert len(data.load("t", placement="BB")) == 5 and len(data.per_cell(data.load("t"), 2)) == 4
    with pytest.raises(SystemExit):
        data.write(qs, tmp_path / "t.jsonl")                           # built once, never overwritten


def test_benchmark_files_match_checksums():
    """Everyone's benchmark files must be byte-identical (they're gitignored and rebuilt locally)."""
    sums = data.BENCHMARK / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, name = line.split()
        path = Path(data.BENCHMARK / name)
        if not path.exists():
            pytest.skip(f"{name} not built yet")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"{name} differs from SHA256SUMS: rebuild it"
