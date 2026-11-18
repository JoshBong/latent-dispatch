"""scripts/paper_tables.py rebuilds the fresh-source tables from the archived rows. These tests pin it two ways:
against numbers the runner itself saved at run time (an independent implementation), and against the values quoted
in JOURNAL.md, so a change to the scorer, the statistics or an archive shows up here."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_tables", ROOT / "scripts" / "paper_tables.py")
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


@pytest.fixture(scope="module")
def qs():
    return tables.questions()


@pytest.fixture(scope="module")
def main(qs):
    return tables.main_table(qs)


@pytest.fixture(scope="module")
def position(qs):
    return tables.position_table(qs)


@pytest.mark.parametrize("model", list(tables.MODELS))
def test_main_run_is_complete_and_agrees_with_the_runners_saved_summary(model, qs):
    tarball, job, _ = tables.MODELS[model]
    scores, sources, cohort = tables.main_run(model, qs)
    assert len(sources) == 200 and len(scores) == 2400            # 200 sources x AB/BA x six conditions
    saved = json.loads(tables.read_tar(tarball, [f"{job}/main/summary.json"])[f"{job}/main/summary.json"])
    assert saved["probes"]["strict_paired_sources"] == len(cohort)
    for contrast, (a, b) in {"gold-blocked": ("gold", "blocked"), "gold-full": ("gold", "full"),
                             "gold-matched_irrelevant": ("gold", "matched_irrelevant"), "b1-k0": ("b1", "k0")}.items():
        theirs = next(c["mean"] for c in saved["contrasts"] if c["contrast"] == contrast and c["metric"] == "f1"
                      and c["cohort"] == "all" and c.get("placement") == "AB+BA")
        ours = sum(tables.difference(scores, a, b, sources)) / len(sources)
        assert ours == pytest.approx(theirs, abs=1e-9), (model, contrast)


def test_main_table_values_quoted_in_the_journal(main):
    c = main["contrasts"]
    assert c[("gold", "blocked")]["Qwen3-8B"]["all"] == (7.4, 3.5, 11.5)
    assert c[("gold", "blocked")]["Llama 3.1 8B"]["all"] == (5.6, 1.8, 9.5)
    assert c[("gold", "blocked")]["Qwen3-4B"]["all"] == (9.8, 6.5, 13.2)
    assert c[("gold", "blocked")]["pooled"] == (7.6, 5.1, 10.2)
    assert c[("matched_irrelevant", "blocked")]["pooled"] == (-0.8, -2.6, 1.0)
    assert c[("gold", "full")]["Qwen3-4B"]["all"] == (3.6, 0.6, 6.7)
    assert c[("gold", "full")]["pooled"] == (2.0, 0.1, 3.8)
    assert main["levels"]["Qwen3-8B"]["gold"] == (40.0, 49.8)
    assert main["pooled_exact_match"][("gold", "full")] == (1.2, -0.7, 3.2)


def test_position_table_values_quoted_in_the_journal(position):
    c = position["contrasts"]
    assert c[("gold", "blocked")]["Qwen3-8B"]["all"] == (10.9, 6.7, 15.3)
    assert c[("gold", "matched_late")]["Llama 3.1 8B"]["all"] == (7.9, 3.9, 12.0)
    assert c[("gold", "full")]["Qwen3-8B"]["all"] == (-3.1, -5.9, -0.3)
    assert c[("gold", "full")]["pooled"] == (-2.1, -4.4, 0.2)
    assert position["change"][("gold", "full")]["pooled"] == (3.3, 0.5, 6.2)
    assert position["change"][("full", "blocked")]["pooled"] == (-6.4, -9.8, -3.1)


def test_mask_checks_and_validation_counts():
    checks = tables.checks_table()
    for model in tables.POSITION:
        assert checks[model]["layouts"] == 48
        assert checks[model]["swap_blocked_identical"] == 48 and checks[model]["swap_full_changed"] == 48
        assert checks[model]["full_equals_joint_one_kernel"] == 48 and checks[model]["full_equals_joint_default"] == 0
    validation = tables.validation_table()
    assert validation["Qwen3-8B"]["blocked"] == {"pairs": 24, "identical": 10, "em_differs": 9, "steps": 2170, "steps_agree": 2160}
    assert validation["Qwen3-4B"]["full"]["identical"] == 23


def test_toy_table(qs):
    toy = tables.toy_table(qs)
    assert toy["chains"] == 25
    assert toy["levels"]["blocked"] == {("off", "split"): 89.0, ("off", "together"): 97.0,
                                        ("forced", "split"): 41.0, ("forced", "together"): 97.0}
    assert toy["contrasts"]["blocked, split: forced − off"] == (-48.0, -62.0, -33.0)
    assert toy["cap"] == {"answers": 100, "thinking_cut": 60, "cut_and_right": 1.0, "finished_and_right": 40.0}
