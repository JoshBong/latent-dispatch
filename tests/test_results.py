import io
import json
import tarfile

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


@pytest.mark.parametrize("fragment", [b'{"id": "partial', b'{"id": "partial", "text": "\xc3'])
def test_resume_discards_only_the_interrupted_final_fragment(tmp_path, fragment):
    path = tmp_path / "arm.jsonl"
    complete = b'{"id": "complete", "text": "kept"}\n'
    path.write_bytes(complete + fragment)
    assert list(results._read(path)) == ["complete"]
    results._repair_tail(path)
    assert path.read_bytes() == complete
    with path.open("ab") as f:
        f.write(b'{"id": "next"}\n')
    assert list(results._read(path)) == ["complete", "next"]


def test_resume_preserves_a_complete_row_without_a_final_newline(tmp_path):
    path = tmp_path / "arm.jsonl"
    row = b'{"id": "complete", "text": "kept"}'
    path.write_bytes(row)
    results._repair_tail(path)
    assert path.read_bytes() == row + b"\n"


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


def test_load_all_marks_the_smoke_questions(tiny, two_workers, tmp_path, monkeypatch):
    """The first 2 questions of a cell are the ones a smoke run answers; the gate judges only those."""
    from latent_dispatch import data
    monkeypatch.setattr(results, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(data, "BENCHMARK", tmp_path)
    qs = [{**two_workers, "dataset": "toy_d0", "id": f"t:{i}"} for i in range(3)]
    data.write(qs, tmp_path / "toy_d0.jsonl")
    results.run(tiny, qs, "k0", arms.k0, max_answer_tokens=2)
    df = results.load_all(tiny.name, tiny.dtype_name)
    assert df.set_index("id").smoke.to_dict() == {"t:0": True, "t:1": True, "t:2": False}


def test_gate_flags_broken_output_only():
    import pandas as pd

    def rows(arm, n, text="\\boxed{a}", cut=False):
        return [{"arm": arm, "text": text, "cut": cut}] * n

    df = pd.DataFrame(rows("b1", 9) + rows("b1", 1, cut=True)                          # one capped answer is fine
                      + rows("k0", 10, text="no box")                                   # nothing boxed
                      + rows("k1_m8", 7) + rows("k1_m8", 3, cut=True)                   # 30% ran to the cap
                      + rows("b1_only_hop1", 10, text="ramble", cut=True)              # unanswerable by design
                      + rows("b1_forced", 10, text="<think>\\boxed{a}</think> none"))  # boxed only inside thinking
    t = analysis.gate(df)
    assert t.broken.to_dict() == {"b1": False, "k0": True, "k1_m8": True, "b1_only_hop1": False, "b1_forced": True}
    assert t.loc["b1", ["n", "unboxed", "capped"]].tolist() == [10, 0, 1] and t.loc["b1_only_hop1", "exempt"]
    assert not analysis.gate(df, cut_max=101).loc["k1_m8", "broken"]
    assert analysis.gate(df.iloc[:0]).empty


# ------------------------------------------------------------------------------------- start from an earlier run

def _tar(path, files):
    """A tarball holding {name: text}."""
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return path


def _row(i=0, **changes):
    return json.dumps({"id": f"q{i}", "arm": "k0", "options": {}, "model": "org/M", "dtype": "bfloat16",
                       "commit": "abc1234", "text": "\\boxed{a}", "cut": False, **changes}) + "\n"


def test_load_archive_unpacks_and_keeps_files_already_there(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "_same_inputs", lambda commit, where: None)       # the git check is tested below
    tar = _tar(tmp_path / "run.tar.gz", {"results/M-bfloat16/k0.jsonl": _row(0) + _row(1),
                                         "results/M-bfloat16/b1.jsonl": _row(0, arm="b1"),
                                         "notebooks/runs/M-h1.ipynb": "{}"})
    root = tmp_path / "machine"
    (root / "results/M-bfloat16").mkdir(parents=True)
    (root / "results/M-bfloat16/b1.jsonl").write_text("rows added on this machine\n")
    results.load_archive(tar, root)
    assert len((root / "results/M-bfloat16/k0.jsonl").read_text().splitlines()) == 2
    assert (root / "notebooks/runs/M-h1.ipynb").exists()
    assert (root / "results/M-bfloat16/b1.jsonl").read_text() == "rows added on this machine\n"     # not overwritten


@pytest.mark.parametrize("name, text", [
    ("results/M-bfloat16/k0.jsonl", _row(0) + _row(1, arm="b1")),                 # two arms in one file
    ("results/M-bfloat16/k0.jsonl", _row(0) + _row(1, options={"m": 8})),         # two sets of options
    ("results/M-bfloat16/k0.jsonl", _row(0) + '{"id": "cut off mid-wri'),         # a cut-off line
    ("results/M-bfloat16/k0.jsonl", json.dumps({"id": "q0", "arm": "k0"}) + "\n"),  # fields missing
    ("results/Other-bfloat16/k0.jsonl", _row(0)),                                 # filed under another model
    ("latent_dispatch/arms.py", "print('not results')"),                          # outside results/
    ("results/../../escape.jsonl", _row(0)),
], ids=["two arms", "two options", "cut-off line", "missing fields", "wrong model", "outside results", "dot-dot"])
def test_load_archive_refuses_what_run_would_not_write(tmp_path, monkeypatch, name, text):
    monkeypatch.setattr(results, "_same_inputs", lambda commit, where: None)
    tar = _tar(tmp_path / "run.tar.gz", {"results/M-bfloat16/b1.jsonl": _row(0, arm="b1"), name: text})
    with pytest.raises(ValueError):
        results.load_archive(tar, tmp_path / "machine")
    assert not (tmp_path / "machine").exists()                                    # nothing was unpacked


def test_loaded_rows_must_come_from_this_checkouts_benchmark_and_prompts():
    head = results._commit().removesuffix("-dirty")
    if head == "unknown":
        pytest.skip("not a git checkout")
    results._same_inputs(head, "x")                               # the same commit: the same inputs
    for commit in ("a1a3e04", head + "-dirty", "unknown"):        # a1a3e04 had other benchmark checksums
        with pytest.raises(ValueError):
            results._same_inputs(commit, "x")


@pytest.mark.parametrize("path", sorted((results.ROOT / "archive").glob("*.tar.gz")), ids=lambda p: p.name)
def test_archived_runs_are_in_the_format_run_writes(path, tmp_path, monkeypatch):
    monkeypatch.setattr(results, "_same_inputs", lambda commit, where: None)       # format only: inputs may move on
    results.load_archive(path, tmp_path)
    assert list((tmp_path / "results").glob("*/*.jsonl"))
