"""HotpotQA under the fresh-source mask protocol: hop order and bridge, layouts, selection, the frozen question set,
and the runner on tiny random models."""
import copy
import json
import random

import pytest

from latent_dispatch import wmac_experiments
from latent_dispatch import wmac_hotpot as wh
from latent_dispatch.paper_plan import digest
from latent_dispatch.wmac_plan import grounding_flags, support_last, supports
from scripts import run_wmac_hotpot as runner
from .conftest import tiny_model

FIRST = "Velmora Guild: The Velmora Guild is a painters' society founded by Ines Farrow in the town of Dunmere."
SECOND = "Ines Farrow: Ines Farrow is a painter from Dunmere. Her teacher was Bram Holt."


def distractors(tag):
    return [f"Town {tag}{i}: Town {tag}{i} lies on the river Calder and keeps a covered market, a ferry and three bridges."
            for i in range(8)]


def hotpot(source="s1", answer="Bram Holt", first=FIRST, second=SECOND, swap=False, qtype="bridge", question=None):
    """One benchmark-style HotpotQA question: AB layout, 5 paragraphs per worker, chain order not meaningful."""
    d = distractors(source)
    a, b = ([second] + d[:4], [first] + d[4:]) if swap else ([first] + d[:4], [second] + d[4:])
    return dict(id=f"{source}:AB:s0", dataset="hotpotqa", source_id=source, qtype=qtype, hops=2, placement="AB",
                question=question or f"Who taught the founder of the Velmora Guild ({source})?", answers=[answer],
                crossings=1, seed=0, workers=[a, b], bridges=[], decomposition=[],
                gold=[{"hop": 0, "worker": 0, "index": 0}, {"hop": 1, "worker": 1, "index": 0}])


def variant(source):
    """A source with its own support text and question, so sources are independent of one another."""
    return hotpot(source, answer=f"Bram Holt {source}",
                  first=f"Guild {source}: Guild {source} is a painters' society founded by Ines Farrow {source} in Dunmere.",
                  second=f"Ines Farrow {source}: Ines Farrow {source} is a painter. Her teacher was Bram Holt {source}.",
                  question=f"Who taught the founder of Guild {source}?")


# ---- hop order and bridge ------------------------------------------------------------------------------------------

def test_title_drops_a_trailing_parenthesis():
    assert wh.title("A Night at the Opera (Queen album): A Night at the Opera is the fourth album.") == "A Night at the Opera"
    assert wh.title("Bully Boy: Bully Boy is a play.") == "Bully Boy"


@pytest.mark.parametrize("swap", [False, True])
def test_the_answer_paragraph_is_hop_two_and_a_supporting_title_is_the_bridge(swap):
    original = hotpot(swap=swap)
    kept = copy.deepcopy(original)
    q, flags = wh.chain(original)
    assert flags == [] and original == kept
    assert supports(q) == [FIRST, SECOND]                       # hop 1 has no answer, hop 2 holds it
    assert q["bridges"] == ["Ines Farrow"] and grounding_flags(q) == []


def test_questions_without_a_usable_chain_are_flagged():
    both = hotpot(first=FIRST + " Bram Holt visited once.")
    neither = hotpot(answer="Aria Venn")
    yes_no = hotpot(answer="yes")
    assert wh.chain(both)[1] == ["answer_in_both_supports"]
    assert wh.chain(neither)[1] == ["answer_absent_from_supports"]
    assert wh.chain(yes_no)[1] == ["yes_no_answer"]
    no_bridge, flags = wh.chain(hotpot(first="Velmora Guild: The Velmora Guild is a painters' society in Dunmere."))
    assert flags == [] and no_bridge["bridges"] == []
    assert grounding_flags(no_bridge) == ["bridge_not_explicit_in_both_supports"]


# ---- layouts ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("swap", [False, True])
def test_ab_and_ba_put_each_support_last_and_keep_each_workers_distractors(swap):
    source = hotpot(swap=swap)
    q, _ = wh.chain(source)
    kept = copy.deepcopy(q)
    ab, ba = wh.layouts(q)
    d = distractors("s1")
    assert q == kept
    assert ab["workers"] == [d[:4] + [FIRST], d[4:] + [SECOND]]
    assert ba["workers"] == [d[:4] + [SECOND], d[4:] + [FIRST]]
    assert [(g["hop"], g["worker"], g["index"]) for g in ab["gold"]] == [(0, 0, 4), (1, 1, 4)]
    assert [(g["hop"], g["worker"], g["index"]) for g in ba["gold"]] == [(0, 1, 4), (1, 0, 4)]
    assert (ab["id"], ab["placement"], ba["id"], ba["placement"]) == ("s1:AB:s0", "AB", "s1:BA:s0", "BA")
    for made in (ab, ba):
        assert made["layout_protocol"] == "support-last-v1" and support_last(made) == made
        assert sorted(p for w in made["workers"] for p in w) == sorted(p for w in source["workers"] for p in w)
        assert supports(made) == [FIRST, SECOND] and grounding_flags(made) == []


def test_layouts_need_hop_labels_on_two_workers():
    q, _ = wh.chain(hotpot())
    q["gold"][1]["worker"] = 0
    with pytest.raises(ValueError, match="one supporting paragraph in each"):
        wh.layouts(q)


# ---- selection -------------------------------------------------------------------------------------------------------

def pool():
    sources = [variant(f"v{i}") for i in range(6)]
    sources.append(hotpot("bad", first=FIRST + " Bram Holt visited once."))                 # answer in both supports
    sources.append(hotpot("cmp", qtype="comparison"))                                       # not a bridge question
    return sources


def test_selection_walks_a_fixed_order_and_explains_every_drop():
    questions = pool()
    selected, audit = wh.select_sources(questions, set(), 2, 7)
    again, _ = wh.select_sources(random.Random(3).sample(questions, len(questions)), set(), 2, 7)
    assert selected == again                                               # the order of the input does not matter
    assert [q["placement"] for q in selected] == ["AB", "BA", "AB", "BA"]
    assert selected[0]["source_id"] == selected[1]["source_id"] != selected[2]["source_id"]
    assert "cmp" not in {row["source_id"] for row in audit}
    one, _ = wh.select_sources(questions, set(), 1, 7)
    assert one == selected[:2]                                             # asking for more keeps the earlier sources
    everything, full_audit = wh.select_sources(questions, set(), 6, 7)
    assert {row["source_id"]: row["flags"] for row in full_audit}["bad"] == ["answer_in_both_supports"]
    assert len(everything) == 12
    with pytest.raises(ValueError, match="only 6 eligible"):
        wh.select_sources(questions, set(), 7, 7)


def test_selection_leaves_out_archived_sources_shared_supports_and_repeated_questions():
    questions = pool()
    first = wh.select_sources(questions, set(), 1, 7)[0][0]["source_id"]
    twin = variant("twin")                                # same supporting text and question as the first source
    for field in ("workers", "question", "answers"):
        twin[field] = copy.deepcopy(next(q for q in questions if q["source_id"] == first)[field])
    selected, audit = wh.select_sources(questions + [twin], {first}, 5, 7)
    flags = {row["source_id"]: row["flags"] for row in audit}
    # as in the MuSiQue planner, an archived source also collides with its own supports and question
    assert flags[first] == ["duplicate_question", "historical_source", "shared_support_text"]
    assert flags["twin"] == ["duplicate_question", "shared_support_text"]
    assert first not in {q["source_id"] for q in selected} and "twin" not in {q["source_id"] for q in selected}


def test_selection_drops_a_source_when_either_layout_cannot_be_run():
    questions = pool()
    refuse = wh.select_sources(questions, set(), 1, 7)[0][1]["id"]         # the BA layout of the first source
    selected, audit = wh.select_sources(questions, set(), 2, 7, lambda q: "context_too_long" if q["id"] == refuse else None)
    assert audit[0] == dict(source_id=refuse.split(":")[0], flags=["context_too_long"])
    assert refuse.split(":")[0] not in {q["source_id"] for q in selected}


# ---- the equal-token window ------------------------------------------------------------------------------------------

def test_the_window_precedes_the_support_and_skips_paragraphs_that_mention_the_bridge(tokenizer):
    source = hotpot()
    source["workers"][0][2] = "Town s11: Ines Farrow once painted the covered market of this town on the river."
    q, _ = wh.chain(source)
    stand_in = wh.adapter(tokenizer, "qwen3")
    for layout in wh.layouts(q):
        chosen, meta = wmac_experiments.matched_indices(stand_in, layout)
        support = meta["a_gold_indices"]
        assert len(chosen) == len(support) and max(chosen) < min(support)
        assert 1 not in meta["a_paragraphs"]                       # the paragraph that mentions the bridge: slot 1 of A
        assert wh.feasibility({"m": stand_in}, 4096)(layout) is None
        assert wh.feasibility({"m": stand_in}, 10)(layout) == "context_too_long"
    short = copy.deepcopy(wh.layouts(q)[0])
    short["workers"][0][:4] = ["Town: A.", "Town: B.", "Town: C.", "Town: D."]
    assert wh.feasibility({"m": stand_in}, 4096)(short) == "input_token_control_ineligible"


# ---- the frozen question set -----------------------------------------------------------------------------------------

def config(n=2):
    cfg = json.loads(wh.CONFIG.read_text())
    cfg["hotpot_sources"] = n
    return cfg


@pytest.fixture(scope="module")
def bundle(tokenizer):
    adapters = {key: wh.adapter(tokenizer, "qwen3") for key in ("8b", "llama", "4b")}
    return wh.build_bundle(config(), pool(), set(), adapters)


def test_the_bundle_freezes_questions_hashes_and_token_sequences(bundle, tmp_path):
    assert len(bundle["hotpot_ids"]) == 4 and set(bundle["preflight"]) == {"8b", "llama", "4b"}
    row = bundle["preflight"]["8b"][bundle["hotpot_ids"][0]]
    assert row["selected_keys_precede_gold"] and not set(wh.HEAVY) & set(row)
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(bundle))
    loaded, questions = wh.load(path)
    assert loaded == json.loads(json.dumps(bundle)) and list(questions) == bundle["hotpot_ids"]
    assert wh.prepare(destination=path)[0] == loaded                       # an existing file is loaded, not rebuilt


@pytest.mark.parametrize("damage,message", [
    (lambda b: b["questions"][b["hotpot_ids"][0]]["workers"][0].reverse(), "hash mismatch"),
    (lambda b: b["hotpot_ids"].reverse(), "AB layout then its BA layout"),
    (lambda b: b["hotpot_ids"].pop(), "inconsistent"),
    (lambda b: b["config"].update(protocol="wmac_gpu_v3"), "not a wmac_hotpot_v1"),
])
def test_a_changed_question_set_is_refused(bundle, tmp_path, damage, message):
    changed = json.loads(json.dumps(bundle))
    damage(changed)
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match=message):
        wh.load(path)


def test_a_layout_that_breaks_the_protocol_is_refused_even_with_matching_hashes(bundle, tmp_path):
    changed = json.loads(json.dumps(bundle))
    q = changed["questions"][changed["hotpot_ids"][0]]
    q["workers"][0].reverse()                                              # the support is no longer last in A
    q["gold"][0]["index"] = 0
    changed["question_sha256"][q["id"]] = digest(q)
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="breaks the protocol|same paragraphs"):
        wh.load(path)


def test_the_committed_question_set_follows_the_protocol():
    if not wh.PREPARED.exists():
        pytest.skip("experiments/wmac_hotpot_v1/prepared.json is not built")
    frozen, questions = wh.load()
    cfg = json.loads(wh.CONFIG.read_text())
    assert frozen["config"] == cfg and len(questions) == 2 * cfg["hotpot_sources"]
    assert frozen["tokenizer_sha256"] == {key: model["tokenizer_sha256"] for key, model in cfg["models"].items()}
    sources = {q["source_id"] for q in questions.values()}
    assert len(sources) == cfg["hotpot_sources"] and not sources & set(frozen["excluded_historical_sources"])
    kept = [row["source_id"] for row in frozen["selection_audit"] if not row["flags"]]
    assert kept == list(dict.fromkeys(q["source_id"] for q in questions.values()))
    for q in questions.values():
        assert [len(worker) for worker in q["workers"]] == [5, 5] and q["dataset"] == "hotpotqa" and q["qtype"] == "bridge"
    for key in cfg["models"]:
        assert set(frozen["preflight"][key]) == set(questions)
        assert all(row["selected_keys_precede_gold"] for row in frozen["preflight"][key].values())


# ---- the runner ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["qwen3", "llama3"])
def test_main_and_probes_run_on_a_tiny_model_and_resume(bundle, tokenizer, tmp_path, family):
    model = tiny_model(family, tokenizer)
    folder = tmp_path / "tiny"
    assert runner.run_part(model, bundle, "8b", folder / "main", "main", sources=1, answer_tokens=3)
    assert runner.run_part(model, bundle, "8b", folder / "probes", "probes", sources=1, answer_tokens=3)
    rows = runner.read_rows(folder / "main" / "rows.jsonl")
    ab, ba = bundle["hotpot_ids"][:2]
    assert [(r["id"], r["condition"]) for r in rows] == [(id, f"{c}_off") for id in (ab, ba) for c in wh.CONDITIONS]
    for id in (ab, ba):                                         # every condition answers at the same position
        assert len({r["final_answer_position"] for r in rows if r["id"] == id}) == 1
        assert len({r["context_tokens_sha256"] for r in rows if r["id"] == id}) == 1
    window = next(r for r in rows if r["condition"] == "matched_irrelevant_off")
    assert window["mask"]["a_visible_tokens"] == window["mask"]["a_gold_tokens"] and window["mask"]["selected_keys_precede_gold"]
    probes = runner.read_rows(folder / "probes" / "rows.jsonl")
    assert [(r["id"], r["condition"]) for r in probes] == [(ab, name) for name in wh.PROBES]
    status = json.loads((folder / "main" / "status.json").read_text())
    assert status["complete"] and status["saved_rows"] == 12
    # a rerun adds nothing; a row cut off mid-write is redone; more sources continue the same file
    before = (folder / "main" / "rows.jsonl").read_bytes()
    assert runner.run_part(model, bundle, "8b", folder / "main", "main", sources=1, answer_tokens=3)
    assert (folder / "main" / "rows.jsonl").read_bytes() == before
    (folder / "main" / "rows.jsonl").write_bytes(before[:len(before) - 40])
    assert runner.run_part(model, bundle, "8b", folder / "main", "main", sources=1, answer_tokens=3)
    assert len(runner.read_rows(folder / "main" / "rows.jsonl")) == 12
    assert runner.run_part(model, bundle, "8b", folder / "main", "main", sources=2, answer_tokens=3)
    assert len(runner.read_rows(folder / "main" / "rows.jsonl")) == 24
    with pytest.raises(ValueError, match="plan differs"):                  # another answer cap is another run
        runner.run_part(model, bundle, "8b", folder / "main", "main", sources=1, answer_tokens=4)


def test_rows_from_another_question_set_are_refused(bundle, tokenizer, tmp_path):
    model = tiny_model("qwen3", tokenizer)
    assert runner.run_part(model, bundle, "8b", tmp_path / "main", "main", sources=1, answer_tokens=3, conditions=("blocked",))
    changed = json.loads(json.dumps(bundle))
    changed["questions"][changed["hotpot_ids"][0]]["question"] += " (edited)"
    with pytest.raises(ValueError, match="different question"):
        runner.run_part(model, changed, "8b", tmp_path / "main", "main", sources=1, answer_tokens=3, conditions=("blocked",))


def test_a_tokenizer_that_cuts_the_questions_differently_is_refused(bundle, tokenizer, tmp_path):
    changed = json.loads(json.dumps(bundle))
    changed["preflight"]["8b"][changed["hotpot_ids"][0]]["input_ids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tokenize differently"):
        runner.run_part(tiny_model("qwen3", tokenizer), changed, "8b", tmp_path / "main", "main", sources=1, answer_tokens=3)
    assert not (tmp_path / "main" / "rows.jsonl").exists() or not runner.read_rows(tmp_path / "main" / "rows.jsonl")


def test_the_first_look_uses_only_sources_with_every_answer(bundle, tmp_path):
    ids = bundle["hotpot_ids"]
    answer = lambda id: bundle["questions"][id]["answers"][0]                 # noqa: E731
    rows = runner.Rows(tmp_path / "m" / "main" / "rows.jsonl")
    for id in ids:
        for condition in ("blocked", "gold"):
            if id == ids[3] and condition == "gold":
                continue                                                   # the second source lacks one answer
            right = condition == "gold" or id == ids[1]
            rows.add(f"{id}|{condition}", dict(id=id, condition=f"{condition}_off", cut=False,
                                               text="\\boxed{" + (answer(id) if right else "nobody") + "}"))
    lines = []
    result = runner.summarize(tmp_path / "m", bundle, lines.append)
    assert result["sources"] == 1 and result["contrasts"] == {"gold-blocked": 50.0}
    assert lines[0].startswith("1 sources with every answer")
    assert runner.summarize(tmp_path / "empty", bundle, lines.append) is None
