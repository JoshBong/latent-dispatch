#!/usr/bin/env python
"""Both supporting paragraphs in one worker: the control layout for the fresh-source mask result.
Uses the fresh-source library as it is (commit 64044d8); changes nothing in it.

The main run puts one supporting paragraph in each worker and finds that B must be able to read A's. If that loss
comes from a fact having to cross between workers, it should vanish when nothing has to cross. For every source of
the main run this builds two layouts from its AB layout:

  AA   both supporting paragraphs in A, in its last two slots (hop 1, then hop 2); B holds only distractors
  BB   both supporting paragraphs in B, in its last two slots;                      A holds only distractors

The worker that gives up its supporting paragraph gets, in its last slot, the distractor that made room for it.
Every paragraph is used exactly once and both workers keep their number of paragraphs.

Conditions (thinking off, same decoding as the main run):
  blocked   B cannot attend to A
  full      B attends to all of A
  b1        joint reading, no mask (optional; the same computation as full)

Rows are appended to <out>/<model>/together/rows.jsonl as they are produced; a rerun skips rows already saved.

  python wmac_together.py selftest                                   # seconds, CPU, tiny random models
  python wmac_together.py preflight --prepared prepared.json         # tokenizers only: is every new layout usable?
  python wmac_together.py run --prepared prepared.json --model 8b --out results [--conditions blocked,full]
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from latent_dispatch import masking, paper_experiments, prompts, wmac_experiments
from latent_dispatch.model import Model
from latent_dispatch.paper_plan import digest

PREPARED_SHA256 = "8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058"   # the main run's question set
CONDITIONS = ("blocked", "full", "b1")
PLACEMENTS = ("AA", "BB")
LAYOUT = "support-together-v1"


# ------------------------------------------------------------------------------------------------ small helpers

class Rows:
    """Append-only JSONL, one row per key. A row cut off when the machine stopped is ignored and redone."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            for line in raw.split(b"\n"):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    self.rows[row["key"]] = row
            if raw and not raw.endswith(b"\n"):
                with self.path.open("ab") as f:
                    f.write(b"\n")

    def add(self, key, row):
        row = {"key": key, **row}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.rows[key] = row


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def frozen(path, value):
    """Saved rows may only be continued under the same plan."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(value)):
            raise ValueError(f"{path}: the plan differs from the one these rows were made under; use a new --out")
    else:
        write_json(path, value)


def load_bundle(path, check_sha=True):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if check_sha and sha != PREPARED_SHA256:
        raise ValueError(f"{path} is not the main run's question set (sha256 {sha[:12]}..., expected {PREPARED_SHA256[:12]}...)")
    return json.loads(raw)


def load_model(bundle, key, device=None, dtype="bfloat16", repo=None, fmt=None):
    """The model exactly as the main run loaded it. repo/fmt replace it with a small model for a local smoke test."""
    cfg, entry = bundle["config"], bundle["config"]["models"][key]
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if repo:
        return Model.load(repo, device=device, dtype=dtype, fmt=fmt, attn_implementation=cfg["attention"])
    return Model.load(entry["name"], device=device, dtype=dtype, fmt=entry["format"], revision=entry["revision"],
                      attn_implementation=cfg["attention"])


def runtime(model):
    import transformers
    cuda = model.device.type == "cuda"
    return dict(model=model.name, revision=getattr(model.hf.config, "_commit_hash", None), dtype=model.dtype_name,
                device=str(model.device), device_name=torch.cuda.get_device_name(0) if cuda else model.device.type,
                torch=torch.__version__, transformers=transformers.__version__,
                attention=getattr(model.hf.config, "_attn_implementation", None))


def code_commit():
    try:
        root = Path(masking.__file__).resolve().parents[1]
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def free(model):
    if model.device.type == "cuda":
        torch.cuda.empty_cache()


def tokenizer_only(name, fmt, revision=None):
    """Enough of a Model for the layout checks: no weights."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, **({"revision": revision} if revision else {}))
    return SimpleNamespace(tok=tok, fmt=prompts.FORMATS[fmt], ids=lambda text: tok(text, add_special_tokens=False).input_ids)


# ---------------------------------------------------------------------------------------------------- layouts

def together(q, placement):
    """A copy of an AB layout with both supporting paragraphs in one worker. The original is not touched."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, not {placement!r}")
    q = copy.deepcopy(q)
    hops = sorted(q["gold"], key=lambda g: g["hop"])
    if len(q["workers"]) != 2 or len(hops) != 2 or [g["worker"] for g in hops] != [0, 1]:
        raise ValueError(f"{q['id']}: need the AB layout: hop 1's supporting paragraph in A and hop 2's in B")
    a, b = q["workers"]
    first, second = a[hops[0]["index"]], b[hops[1]["index"]]
    a_rest = [p for i, p in enumerate(a) if i != hops[0]["index"]]
    b_rest = [p for i, p in enumerate(b) if i != hops[1]["index"]]
    if placement == "BB":
        if not b_rest:
            raise ValueError(f"{q['id']}: B has no distractor to make room")
        q["workers"] = [a_rest + [b_rest.pop()], b_rest + [first, second]]
    else:
        if not a_rest:
            raise ValueError(f"{q['id']}: A has no distractor to make room")
        moved = a_rest.pop()
        q["workers"] = [a_rest + [first, second], b_rest + [moved]]
    home = PLACEMENTS.index(placement)                     # 0 = A, 1 = B
    n = len(q["workers"][home])
    q["gold"] = [{**hops[0], "worker": home, "index": n - 2}, {**hops[1], "worker": home, "index": n - 1}]
    q["base_id"], q["id"] = q["id"], f"{q['source_id']}:{placement}:s0"
    q["placement"], q["crossings"], q["layout_protocol"] = placement, 0, LAYOUT
    return q


def base_ids(bundle, sources=None):
    """The AB layout of every source of the main run, in the main run's order."""
    questions = bundle["questions"]
    ids = [i for i in bundle["musique_ids"] if questions[i]["placement"] == "AB"]
    return ids[:sources] if sources else ids


def together_layouts(model, questions, ids, max_tokens):
    """Build and check both layouts of every source before any answer is generated. A source is kept only if both
    pass. -> (kept {new id: question}, dropped {new id: reason})"""
    kept, dropped = {}, {}
    for base in ids:
        made = {}
        for placement in PLACEMENTS:
            name = f"{questions[base]['source_id']}:{placement}:s0"
            try:
                q = together(questions[base], placement)
                before = sorted(p for w in questions[base]["workers"] for p in w)
                if sorted(p for w in q["workers"] for p in w) != before:
                    raise ValueError("the layout does not hold the same paragraphs as the original")
                tokens, _, _ = paper_experiments.context_tokens(model, q)
                pieces = [prompts.prefix(model.fmt, q["question"]), *[prompts.block(w) for w in q["workers"]],
                          prompts.tail(model.fmt, "off")]
                if [t for piece in pieces for t in model.ids(piece)] != model.ids("".join(pieces)):
                    raise ValueError("tokenizing the pieces differs from tokenizing the whole prompt")
                if len(tokens) > max_tokens:
                    raise ValueError(f"context of {len(tokens)} tokens is over {max_tokens}")
                masking.paragraph_tokens(model, q["workers"][0])
                made[name] = q
            except ValueError as error:
                dropped[name] = str(error)
        if len(made) == len(PLACEMENTS):
            kept.update(made)
        else:
            for name in made:
                dropped[name] = "the other layout of this source was dropped"
    return kept, dropped


# -------------------------------------------------------------------------------------------------------- run

def run_together(model, bundle, folder, conditions=CONDITIONS, sources=None):
    cfg, questions = bundle["config"], bundle["questions"]
    kept, dropped = together_layouts(model, questions, base_ids(bundle, sources), cfg["max_context_tokens"])
    info, commit = runtime(model), code_commit()
    frozen(folder / "plan.json", dict(layout_protocol=LAYOUT, conditions=list(conditions), model=info["model"], revision=info["revision"],
                                      dtype=info["dtype"], commit=commit, answer_tokens=cfg["answer_tokens"],
                                      question_sha256={id: digest(q) for id, q in kept.items()}, dropped=dropped))
    total = len(kept) * len(conditions)
    print(f"together: {len(kept)} layouts ({len(kept) // 2} sources) x {len(conditions)} conditions = {total} answers; "
          f"dropped before any answer: {len(dropped)}", flush=True)
    rows = Rows(folder / "rows.jsonl")
    todo = [(id, c) for id in kept for c in conditions if f"{id}|{c}" not in rows.rows]
    start, done = time.monotonic(), 0
    for id, condition in todo:
        q, t = kept[id], time.monotonic()
        fields = wmac_experiments.generate(model, q, condition, "off", cfg["think_budget"], cfg["answer_tokens"])
        rows.add(f"{id}|{condition}", dict(
            id=id, base_id=q["base_id"], source_id=q["source_id"], placement=q["placement"], condition=f"{condition}_off",
            layout_protocol=LAYOUT, commit=commit, question_sha256=digest(q),
            options=dict(condition=condition, thinking="off", answer_tokens=cfg["answer_tokens"], think_budget=cfg["think_budget"]),
            device_name=info["device_name"], seconds=time.monotonic() - t, **fields))
        done += 1
        if done % 25 == 0 or done == len(todo):
            pace = (time.monotonic() - start) / done
            print(f"  together {len(rows.rows)}/{total} answers, {pace:.2f} s each, about {(len(todo) - done) * pace / 60:.0f} min left",
                  flush=True)
        free(model)
    complete = len(rows.rows) == total
    write_json(folder / "status.json", dict(complete=complete, saved_rows=len(rows.rows), expected_rows=total,
                                            layouts=len(kept), dropped=len(dropped), runtime=info))
    return complete


def run_preflight(bundle, key, repo=None, fmt=None):
    entry = bundle["config"]["models"][key]
    stand_in = tokenizer_only(repo or entry["name"], fmt or entry["format"], None if repo else entry["revision"])
    same = digest(stand_in.tok.backend_tokenizer.to_str()) == bundle.get("tokenizer_sha256", {}).get(key)
    ids = base_ids(bundle)
    kept, dropped = together_layouts(stand_in, bundle["questions"], ids, bundle["config"]["max_context_tokens"])
    sizes = sorted(len(paper_experiments.context_tokens(stand_in, q)[0]) for q in kept.values())
    print(f"{key}: tokenizer {'matches' if same else 'DOES NOT MATCH'} the main run's; {len(kept)} of {2 * len(ids)} layouts usable, "
          f"{len(dropped)} dropped {sorted(set(dropped.values()))[:3]}")
    if sizes:
        print(f"   context {sizes[0]}..{sizes[-1]} tokens (median {sizes[len(sizes) // 2]})")
    return same, kept, dropped


# ---------------------------------------------------------------------------------------------------- selftest

def selftest():
    """The script's own logic on tiny random models (CPU, float32). Run from the repository folder."""
    sys.path.insert(0, os.getcwd())
    import tempfile
    from tests.conftest import make_question, tiny_model
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    a = ["Aria Venn was born in Dunmere, a quiet town by the sea.", "The harbor of Dunmere closes every winter for repairs.",
         "Velmora is known for its long covered market.", "People: Ines Farrow's favorite city is Velmora.",
         "Old maps show three bridges across the Calder river."]
    b = ["The river the Calder flows through Irvale.", "A ferry leaves Irvale at dawn.", "Cities: The mayor of Velmora is Bram Holt."]
    q = make_question([a, b], gold=((0, 0, 3), (1, 1, 2)), id="t:AB:s0")
    q["source_id"], q["placement"] = "t", "AB"
    original = copy.deepcopy(q)
    # 1. the two layouts: both supports in one worker, last two slots, hop order; the displaced distractor goes last
    bb, aa = together(q, "BB"), together(q, "AA")
    assert q == original
    assert bb["workers"] == [[a[0], a[1], a[2], a[4], b[1]], [b[0], a[3], b[2]]], bb["workers"]
    assert aa["workers"] == [[a[0], a[1], a[2], a[3], b[2]], [b[0], b[1], a[4]]], aa["workers"]
    assert [(g["hop"], g["worker"], g["index"]) for g in bb["gold"]] == [(0, 1, 1), (1, 1, 2)]
    assert [(g["hop"], g["worker"], g["index"]) for g in aa["gold"]] == [(0, 0, 3), (1, 0, 4)]
    assert (bb["id"], bb["base_id"], bb["placement"], aa["id"]) == ("t:BB:s0", "t:AB:s0", "BB", "t:AA:s0")
    for made in (bb, aa):
        assert sorted(p for w in made["workers"] for p in w) == sorted(a + b) and made["layout_protocol"] == LAYOUT
        assert [len(w) for w in made["workers"]] == [len(a), len(b)]
    for wrong in ("AB", "BA"):
        try:
            together(q, wrong)
            raise AssertionError("a placement outside AA / BB was accepted")
        except ValueError:
            pass
    try:
        together(bb, "AA")                                    # not an AB layout any more
        raise AssertionError("a layout that is not AB was accepted")
    except ValueError:
        pass
    for family in ("qwen3", "llama3"):
        m = tiny_model(family, tok)
        for made, gold_in_a in ((bb, []), (aa, [3, 4])):
            ids, mask, sizes, meta = masking.layout(m, made, "blocked")
            n_prefix = len(ids) - sum(sizes)
            b_start = n_prefix + sizes[0]
            lowest = torch.finfo(mask.dtype).min
            # 2. blocked: no B row may attend to any A column; full: every B row may attend to every A column
            assert meta["a_gold_paragraphs"] == gold_in_a and meta["a_visible_tokens"] == 0
            assert bool((mask[0, 0, b_start:, n_prefix:b_start] == lowest).all())
            _, full, _, full_meta = masking.layout(m, made, "full")
            assert full_meta["a_visible_tokens"] == sizes[0] and bool((full[0, 0, b_start:, n_prefix:b_start] == 0).all())
            # 3. every condition generates on the new layout, at the same positions
            for condition in CONDITIONS:
                row = wmac_experiments.generate(m, made, condition, "off", 8, 4)
                assert row["final_answer_position"] == len(ids) and row["worker_tokens"] == sizes
        # 4. a source is kept only if both of its layouts pass
        kept, dropped = together_layouts(m, {"t:AB:s0": q}, ["t:AB:s0"], 4096)
        assert sorted(kept) == ["t:AA:s0", "t:BB:s0"] and not dropped
        kept, dropped = together_layouts(m, {"t:AB:s0": q}, ["t:AB:s0"], 5)
        assert kept == {} and sorted(dropped) == ["t:AA:s0", "t:BB:s0"]
    # 5. rows survive a cut-off line
    with tempfile.TemporaryDirectory() as folder:
        rows = Rows(Path(folder) / "rows.jsonl")
        rows.add("a", {"x": 1})
        with rows.path.open("a") as f:
            f.write('{"key": "b", "x"')                                           # the machine stopped here
        again = Rows(rows.path)
        assert list(again.rows) == ["a"]
        again.add("b", {"x": 2})
        assert list(Rows(rows.path).rows) == ["a", "b"]
    print("selftest passed")


# -------------------------------------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["selftest", "preflight", "run"])
    p.add_argument("--prepared")
    p.add_argument("--model", default="8b", help="8b or llama: the main run's entry for that model")
    p.add_argument("--out", default="results")
    p.add_argument("--conditions", default="blocked,full")
    p.add_argument("--sources", type=int, help="only the first N sources (default all)")
    p.add_argument("--device")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--repo", help="small local smoke test: another model in place of the main run's")
    p.add_argument("--fmt", help="chat format for --repo")
    args = p.parse_args()
    if args.command == "selftest":
        return selftest()
    bundle = load_bundle(args.prepared)
    if args.command == "preflight":
        for key in args.model.split(","):
            run_preflight(bundle, key, args.repo, args.fmt)
        return None
    conditions = tuple(args.conditions.split(","))
    if set(conditions) - set(CONDITIONS):
        raise ValueError(f"conditions must be among {CONDITIONS}")
    model = load_model(bundle, args.model, args.device, args.dtype, args.repo, args.fmt)
    print("loaded", runtime(model), flush=True)
    folder = Path(args.out) / (args.model if not args.repo else args.repo.split("/")[-1])
    return 0 if run_together(model, bundle, folder / "together", conditions, args.sources) else 2


if __name__ == "__main__":
    raise SystemExit(main())
