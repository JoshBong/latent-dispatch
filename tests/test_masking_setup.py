"""Notebook orchestration checks that do not load or run a language model."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from latent_dispatch import data, results

ROOT = Path(__file__).resolve().parents[1]


def source(path, contains):
    cells = json.loads((ROOT / path).read_text())["cells"]
    return next("".join(c["source"]) for c in cells
                if c["cell_type"] == "code" and contains in "".join(c["source"]))


def test_part0_smoke_does_not_overwrite_loaded_full_labels(tmp_path):
    path = tmp_path / "part0_requires_composition.json"
    original = json.dumps({"seeds": [0], "smoke": False, "labels": {"d:q": True}})
    path.write_text(original)
    scope = dict(json=json, results_path=lambda *args: tmp_path / "x.jsonl", MODEL="M", DTYPE="float32",
                 SMOKE=True, seeds=[0], labels={"d:smoke": False})
    exec(source("notebooks/part0_ruler.ipynb", 'filename = "part0_requires_composition-smoke.json"'), scope)
    assert path.read_text() == original
    assert json.loads((tmp_path / "part0_requires_composition-smoke.json").read_text())["smoke"] is True


def scope_for_plan(tmp_path, monkeypatch, smoke):
    monkeypatch.setattr(results, "RESULTS", tmp_path)
    monkeypatch.setattr(results, "_commit", lambda: "test-commit")
    def load(name, **where):
        return [dict(id=name, dataset=name, source_id="q", qtype=where["qtype"], placement="AB", seed=0,
                     workers=[["gold", "distractor"], ["gold"]],
                     gold=[{"hop": 0, "worker": 0, "index": 0}, {"hop": 1, "worker": 1, "index": 0}])]
    def arm(*args, **kwargs):
        raise AssertionError("planning must not run a model")
    return dict(json=json, pd=pd, data=SimpleNamespace(load=load, per_cell=data.per_cell),
                results=results, results_path=results.results_path, display=lambda value: None,
                arms=SimpleNamespace(masked_recompute=arm), MODEL="M", DTYPE="float32", SEEDS=[0],
                SMOKE=smoke, N_PER_CELL=100, SKIP=[], DISTRACTOR_SEED=0, MAX_ANSWER_TOKENS=256)


def test_masking_smoke_uses_fixed_questions_and_full_uses_labels(tmp_path, monkeypatch):
    folder = tmp_path / "M-float32"
    folder.mkdir()
    (folder / "part0_requires_composition.json").write_text(json.dumps(
        {"seeds": [0], "smoke": False, "labels": {"toy:q": True, "musique:q": False}}))
    code = source("notebooks/h1_masking.ipynb", "labels_path =")
    smoke = scope_for_plan(tmp_path, monkeypatch, True)
    exec(code, smoke)
    assert [q["id"] for q in smoke["qs"]] == ["toy", "musique"]
    full = scope_for_plan(tmp_path, monkeypatch, False)
    exec(code, full)
    assert [q["id"] for q in full["qs"]] == ["toy"]
    assert all(item["remaining"] == 1 for item in full["plan"])
    assert (folder / "h1_masking-plan.json").exists()


def test_masking_full_refuses_missing_part0_labels(tmp_path, monkeypatch):
    scope = scope_for_plan(tmp_path, monkeypatch, False)
    with pytest.raises(FileNotFoundError, match="Part 0"):
        exec(source("notebooks/h1_masking.ipynb", "labels_path ="), scope)
