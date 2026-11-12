import json

import pytest

from latent_dispatch import analysis, arms, results

from .conftest import make_question


def test_run_resumes_and_refuses_other_options(tiny, two_workers, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    qs = [two_workers, {**two_workers, "id": "t:1"}]
    results.run(tiny, qs[:1], "k0", arms.k0, max_answer_tokens=4)
    results.run(tiny, qs, "k0", arms.k0, max_answer_tokens=4)          # only t:1 is new
    path = results.results_path(tiny.name, tiny.dtype_name, "k0")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["t:0", "t:1"] and rows[0]["options"] == {"max_answer_tokens": 4}
    with pytest.raises(ValueError):
        results.run(tiny, qs, "k0", arms.k0, max_answer_tokens=8)


def test_load_rescores_and_joins_metadata(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    q = make_question([["People: Ines Farrow's favorite city is Velmora."]])
    path = results.results_path(tiny.name, tiny.dtype_name, "b1")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"id": q["id"], "arm": "b1", "options": {}, "commit": "x",
                                "text": "\\boxed{bram holt}", "em": False}) + "\n")
    df = results.load_results([q], {"b1": (arms.b1, {})}, tiny.name, tiny.dtype_name)
    assert df.em.tolist() == [1.0] and df.placement.tolist() == ["AB"]     # saved em ignored, re-scored


def test_paired_and_trend_tables():
    import pandas as pd
    rows = []
    for i in range(40):
        for pat, c in (("BB", 0), ("AB", 1)):
            meta = dict(id=f"{i}:{pat}", dataset="d", qtype="hop1", placement=pat, crossings=c, hops=2, seed=0,
                        source_id=str(i))
            rows += [{**meta, "arm": "b1", "em": 1.0}, {**meta, "arm": "k0", "em": float(c == 0 or i % 2 == 0)}]
    df = pd.DataFrame(rows)
    p = analysis.paired(df, "b1", "k0", by=("crossings",))
    assert p.loc[0, "diff"] == 0 and p.loc[1, "diff"] == 50
    assert analysis.crossing_trend(df, "b1", "k0", "d").loc[1, "gap_minus_base"] == 50
    assert analysis.equivalent(df, "b1", "k0", placement="BB").equivalent.all()


def test_seeds_of_one_question_count_as_one_unit():
    """The same 40 source questions under 3 seeds with identical outcomes carry no more evidence than 1 seed: the
    CI and p-value must not shrink."""
    import pandas as pd

    def table(seeds):
        rows = []
        for i in range(40):
            for seed in seeds:
                meta = dict(id=f"{i}:AB:s{seed}", dataset="d", qtype="hop1", placement="AB", crossings=1, hops=2,
                            seed=seed, source_id=str(i))
                rows += [{**meta, "arm": "b1", "em": 1.0}, {**meta, "arm": "k0", "em": float(i % 4 == 0)}]
        return analysis.paired(pd.DataFrame(rows), "b1", "k0", by=("crossings",)).loc[1]

    one, three = table([0]), table([0, 1, 2])
    assert three.n == 120 and three.sources == 40
    assert (one["diff"], one.lo, one.hi, one.p) == (three["diff"], three.lo, three.hi, three.p)


def test_load_refuses_rows_made_with_other_options(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    q = make_question([["x"]])
    path = results.results_path(tiny.name, tiny.dtype_name, "t1_1024")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"id": q["id"], "arm": "t1", "options": {"report_tokens": 24}, "commit": "x",
                                "text": "\\boxed{a}"}) + "\n" + '{"id": "cut off mid-wri')
    with pytest.raises(ValueError):
        results.load_results([q], {"t1_1024": (arms.t1, {"report_tokens": 1024})}, tiny.name, tiny.dtype_name)
    assert len(results._read(path)) == 1                                   # the broken last line is skipped


def test_bad_option_values_raise(tiny, two_workers):
    for bad in ({"thinking": "on"}, {"positions": "global"}, {"merge": "sequential"}):
        with pytest.raises(ValueError):
            arms.k0(tiny, two_workers, **bad)


def test_audit_and_load_all(tiny, two_workers, tmp_path, monkeypatch):
    """Every row records cut; the audit turns that into per-arm rates, across every arm file of a model."""
    from latent_dispatch import data
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(data, "BENCHMARK", tmp_path)
    q = {**two_workers, "dataset": "toy_d0"}
    data.write([q], tmp_path / "toy_d0.jsonl")
    results.run(tiny, [q], "k0", arms.k0, max_answer_tokens=2)            # random weights: always cut at 2
    results.run(tiny, [q], "scan_1", arms.scan, rounds=1, note_tokens=2, max_answer_tokens=2)
    df = results.load_all(tiny.name, tiny.dtype_name)
    assert set(df.arm) == {"k0", "scan_1"}
    t = analysis.audit(df)
    assert t.loc["k0", "n"] == 1 and t.loc["k0", "boxed %"] in (0.0, 100.0)
    assert {"cut %", "notes cut %", "think cut %", "EM"} <= set(t.columns)
