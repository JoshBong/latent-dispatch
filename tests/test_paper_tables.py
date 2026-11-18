"""scripts/paper_tables.py rebuilds the fresh-source tables from the archived rows. These tests pin it two ways:
against numbers the runner itself saved at run time (an independent implementation), and against the values quoted
in JOURNAL.md, so a change to the scorer, the statistics or an archive shows up here. The control layout (both
supporting paragraphs in one worker) has no runner summary; its rows are checked against what they record instead."""
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


@pytest.fixture(scope="module")
def together(qs):
    return tables.together_table(qs)


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


@pytest.mark.parametrize("model", list(tables.TOGETHER_FOLDER))
def test_control_layout_rows_are_complete_and_record_the_layout_and_mask(model, qs):
    """Both supporting paragraphs in one worker: every source in AA and BB, blocked and full, and what each row says
    about its own layout and mask."""
    assert len(tables.together_run(model, qs)) == 800              # 200 sources x AA/BB x two conditions, no duplicate
    folder = tables.TOGETHER_FOLDER[model]
    names = [f"{folder}/together/rows.jsonl", f"{folder}/together/plan.json"]
    blobs = tables.read_tar(tables.TOGETHER, names)
    rows, plan = tables.jsonl(blobs[names[0]]), json.loads(blobs[names[1]])
    main_rows = tables.jsonl(tables.read_tar(tables.A100, [f"{tables.MODELS[model][1]}/main/rows.jsonl"])
                             [f"{tables.MODELS[model][1]}/main/rows.jsonl"])
    assert {r["commit"] for r in rows} == {r["commit"] for r in main_rows}          # the main run's code
    assert {r["source_id"] for r in rows} == {r["id"].split(":")[0] for r in main_rows} and plan["dropped"] == {}
    input_tokens = {}
    for r in rows:
        mask, condition = r["mask"], r["condition"].removesuffix("_off")
        assert r["id"] == f"{r['source_id']}:{r['placement']}:s0" and r["base_id"] == f"{r['source_id']}:AB:s0"
        assert qs[r["base_id"]]["source_id"] == r["source_id"]
        assert r["layout_protocol"] == "support-together-v1" and r["question_sha256"] == plan["question_sha256"][r["id"]]
        assert r["options"]["thinking"] == "off" and r["think_tokens"] == 0 and mask["condition"] == condition
        assert len(mask["a_gold_paragraphs"]) == {"AA": 2, "BB": 0}[r["placement"]]   # A holds both or neither
        assert mask["a_visible_tokens"] == {"blocked": 0, "full": mask["a_tokens"]}[condition]
        assert input_tokens.setdefault(r["id"], r["context_tokens_sha256"]) == r["context_tokens_sha256"]
    assert len(input_tokens) == 400                                # the same input under both conditions, 400 layouts


def test_control_layout_table_values_quoted_in_the_journal(together):
    assert together["levels"]["Qwen3-8B"] == {("AA", "blocked"): (42.0, 52.0), ("AA", "full"): (46.5, 55.6),
                                              ("BB", "blocked"): (45.0, 54.5), ("BB", "full"): (41.5, 51.9)}
    assert together["levels"]["Llama 3.1 8B"] == {("AA", "blocked"): (36.5, 44.9), ("AA", "full"): (36.5, 44.9),
                                                  ("BB", "blocked"): (40.5, 48.5), ("BB", "full"): (37.0, 44.3)}
    f1, em = together["contrasts"]["F1"], together["contrasts"]["exact match"]
    assert f1["split"] == {"Qwen3-8B": (6.4, 2.3, 10.7), "Llama 3.1 8B": (4.3, -0.1, 8.6), "pooled": (5.4, 2.0, 8.8)}
    assert f1["together"] == {"Qwen3-8B": (0.5, -2.7, 3.7), "Llama 3.1 8B": (-2.1, -5.9, 1.7), "pooled": (-0.8, -3.3, 1.7)}
    assert f1["split − together"] == {"Qwen3-8B": (6.0, 1.1, 10.9), "Llama 3.1 8B": (6.4, 1.2, 11.6),
                                      "pooled": (6.2, 2.5, 9.9)}
    assert f1["AA"] == {"Qwen3-8B": (3.6, -1.4, 8.7), "Llama 3.1 8B": (0.0, -4.5, 4.5), "pooled": (1.8, -1.6, 5.2)}
    assert f1["BB"] == {"Qwen3-8B": (-2.6, -6.8, 1.4), "Llama 3.1 8B": (-4.1, -9.7, 1.3), "pooled": (-3.4, -6.8, 0.0)}
    assert em["together"]["pooled"] == (-0.6, -3.1, 1.9) and em["split − together"]["pooled"] == (5.8, 1.9, 9.6)
    assert em["AA"] == {"Qwen3-8B": (4.5, -0.5, 9.5), "Llama 3.1 8B": (0.0, -4.5, 4.5), "pooled": (2.2, -1.0, 5.5)}
    assert em["BB"] == {"Qwen3-8B": (-3.5, -7.5, 0.5), "Llama 3.1 8B": (-3.5, -9.0, 2.0), "pooled": (-3.5, -7.0, 0.0)}


def test_the_split_row_of_the_control_table_is_the_main_tables_full_minus_blocked(main, together):
    for model in tables.TOGETHER_FOLDER:
        assert together["contrasts"]["F1"]["split"][model] == main["contrasts"][("full", "blocked")][model]["all"]


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
