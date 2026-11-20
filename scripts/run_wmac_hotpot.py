#!/usr/bin/env python
"""HotpotQA bridge questions under the fresh-source mask protocol: prepare, check, run, first look.

  python scripts/run_wmac_hotpot.py prepare                        # tokenizers only; freezes the question set, once
  python scripts/run_wmac_hotpot.py preflight --model 8b,llama,4b  # tokenizers only; does this machine tokenize the
                                                                   # frozen questions as preparation did?
  python scripts/run_wmac_hotpot.py run --model 8b --part probes --out results/wmac_hotpot_v1
  python scripts/run_wmac_hotpot.py run --model 8b --part main --out results/wmac_hotpot_v1
  python scripts/run_wmac_hotpot.py summary --out results/wmac_hotpot_v1

main    the six conditions of the MuSiQue main run (blocked, gold, full, matched_irrelevant, b1, k0) on the AB and BA
        layout of every source, thinking off: 12 answers per source.
probes  closed book, and each supporting paragraph alone with all distractors, on AB layouts: 3 answers per source.

Rows are appended to <out>/<model>/<part>/rows.jsonl as they are produced; a rerun skips rows already saved.
The questions, the hop-order and bridge rules and the selection are in latent_dispatch/wmac_hotpot.py.
"""
import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from latent_dispatch import score, stats, wmac_hotpot  # noqa: E402
from latent_dispatch.paper_plan import digest, file_sha  # noqa: E402

CONTRASTS = [("gold", "blocked"), ("gold", "matched_irrelevant"), ("matched_irrelevant", "blocked"),
             ("full", "blocked"), ("gold", "full"), ("b1", "k0")]


# ---- small helpers ---------------------------------------------------------------------------------------------------

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
                with self.path.open("ab") as handle:
                    handle.write(b"\n")

    def add(self, key, row):
        row = {"key": key, **row}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
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


def code_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def load_model(cfg, key, device=None, dtype=None, repo=None, fmt=None):
    """The model as the MuSiQue run loaded it. repo and fmt put a small model in its place for a smoke test."""
    import torch
    from latent_dispatch.model import Model
    entry = cfg["models"][key]
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if repo:
        return Model.load(repo, device=device, dtype=dtype or "float32", fmt=fmt or entry["format"],
                          attn_implementation=cfg["attention"])
    return Model.load(entry["name"], device=device, dtype=dtype or entry["dtype"], fmt=entry["format"],
                      revision=entry["revision"], attn_implementation=cfg["attention"])


def runtime(model):
    import torch
    import transformers
    cuda = model.device.type == "cuda"
    return dict(model=model.name, revision=getattr(model.hf.config, "_commit_hash", None), dtype=model.dtype_name,
                device=str(model.device), device_name=torch.cuda.get_device_name(0) if cuda else model.device.type,
                torch=torch.__version__, transformers=transformers.__version__,
                attention=getattr(model.hf.config, "_attn_implementation", None))


# ---- checks before any answer ----------------------------------------------------------------------------------------

def same_tokenization(found, frozen_lengths, ids):
    """Does this tokenizer cut the frozen questions exactly as the tokenizer at preparation did?"""
    wrong = [id for id in ids if found[id]["input_ids_sha256"] != frozen_lengths[id]["input_ids_sha256"]
             or found[id]["worker_tokens"] != frozen_lengths[id]["worker_tokens"]]
    if wrong:
        raise ValueError(f"{len(wrong)} of {len(ids)} layouts tokenize differently from preparation (first: {wrong[0]})")


def model_checks(model, bundle, key, ids, stand_in):
    """The MuSiQue runner's checks on the loaded model: native chat template, the key rotation K0 relies on, and the
    token sequence of every layout. Raises before any answer is generated."""
    from latent_dispatch import wmac_experiments
    questions, cfg = bundle["questions"], bundle["config"]
    template = wmac_experiments.check_prompt_template(model, questions[ids[0]])
    rotation = wmac_experiments.check_rope_shift(model)
    lengths = wmac_experiments.preflight(model, questions, ids, cfg["max_context_tokens"])
    if not stand_in:
        same_tokenization(lengths, bundle["preflight"][key], ids)
    return dict(prompt_template=template, rope_shift=rotation, layouts_checked=len(ids),
                tokenization_matches_preparation=not stand_in)


def run_preflight(bundle, keys):
    """Tokenizers only: every model's tokenizer must be the frozen one and cut every layout as preparation did."""
    from latent_dispatch import wmac_experiments
    cfg, questions, ids = bundle["config"], bundle["questions"], bundle["hotpot_ids"]
    adapters = wmac_hotpot.load_adapters({**cfg, "models": {k: cfg["models"][k] for k in keys}})
    for key, stand_in in adapters.items():
        if digest(stand_in.tok.backend_tokenizer.to_str()) != bundle["tokenizer_sha256"][key]:
            raise ValueError(f"{key}: tokenizer differs from the one the questions were prepared with")
        lengths = wmac_experiments.preflight(stand_in, questions, ids, cfg["max_context_tokens"])
        same_tokenization(lengths, bundle["preflight"][key], ids)
        sizes = sorted(row["context_tokens"] for row in lengths.values())
        print(f"{key}: tokenizer and all {len(ids)} layouts match preparation; context {sizes[0]}..{sizes[-1]} tokens "
              f"(median {sizes[len(sizes) // 2]})", flush=True)


# ---- run -------------------------------------------------------------------------------------------------------------

def jobs(bundle, part, conditions, sources=None):
    """(layout id, name) in run order: sources in the frozen order, so a run that stops early holds whole sources."""
    ids = bundle["hotpot_ids"]
    ids = ids[:2 * sources] if sources else ids
    if part == "main":
        return [(id, c) for id in ids for c in conditions]
    return [(id, probe) for id in ids if bundle["questions"][id]["placement"] == "AB" for probe in wmac_hotpot.PROBES]


def run_part(model, bundle, key, folder, part, conditions=wmac_hotpot.CONDITIONS, sources=None, answer_tokens=None,
             stand_in=False, prepared_sha256=None):
    from latent_dispatch import wmac_experiments
    cfg, questions = bundle["config"], bundle["questions"]
    folder = Path(folder)
    todo_all = jobs(bundle, part, conditions, sources)
    ids = list(dict.fromkeys(id for id, _ in todo_all))
    tokens = answer_tokens or (cfg["answer_tokens"] if part == "main" else cfg["probe_answer_tokens"])
    info, commit = runtime(model), code_commit()
    frozen(folder / "plan.json", dict(protocol=cfg["protocol"], part=part, model=info["model"], revision=info["revision"],
                                      dtype=info["dtype"], answer_tokens=tokens, think_budget=cfg["think_budget"],
                                      stand_in=stand_in,
                                      names=list(conditions) if part == "main" else list(wmac_hotpot.PROBES)))
    rows = Rows(folder / "rows.jsonl")
    for row in rows.rows.values():
        if row["question_sha256"] != digest(questions[row["id"]]):
            raise ValueError(f"{row['id']}: a saved row was made from a different question; use a new --out")
    checks = model_checks(model, bundle, key, ids, stand_in)
    write_json(folder / "runtime.json", dict(runtime=info, commit=commit, prepared_sha256=prepared_sha256, checks=checks))
    todo = [(id, name) for id, name in todo_all if f"{id}|{name}" not in rows.rows]
    print(f"{part}: {len(ids)} layouts, {len(todo_all)} answers, {len(todo_all) - len(todo)} already saved", flush=True)
    start, done = time.monotonic(), 0
    for id, name in todo:
        q, began = questions[id], time.monotonic()
        if part == "main":
            fields = wmac_experiments.generate(model, q, name, "off", cfg["think_budget"], tokens)
            condition, options = f"{name}_off", dict(condition=name, thinking="off", answer_tokens=tokens,
                                                     think_budget=cfg["think_budget"])
        else:
            fields = wmac_experiments.probe(model, q, name, tokens)
            condition, options = name, dict(thinking="off", answer_tokens=tokens)
        rows.add(f"{id}|{name}", dict(id=id, source_id=q["source_id"], placement=q["placement"], condition=condition,
                                      part=part, protocol=cfg["protocol"], commit=commit, question_sha256=digest(q),
                                      options=options, device_name=info["device_name"],
                                      seconds=time.monotonic() - began, **fields))
        done += 1
        if done % 25 == 0 or done == len(todo):
            pace = (time.monotonic() - start) / done
            print(f"  {part} {len(rows.rows)}/{len(todo_all)} answers, {pace:.2f} s each, about "
                  f"{(len(todo) - done) * pace / 60:.0f} min left", flush=True)
        if model.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    saved = sum(f"{id}|{name}" in rows.rows for id, name in todo_all)
    complete = saved == len(todo_all)
    write_json(folder / "status.json", dict(complete=complete, saved_rows=saved, expected_rows=len(todo_all),
                                            layouts=len(ids), runtime=info))
    return complete


# ---- first look ------------------------------------------------------------------------------------------------------

def read_rows(path):
    return list(Rows(path).rows.values()) if Path(path).exists() else []


def interval(differences, resamples):
    mean, low, high = stats.bootstrap_ci(differences, n_boot=resamples, seed=0)
    return f"{100 * mean:+.1f} ({100 * low:.1f}, {100 * high:.1f})"


def summarize(folder, bundle, out=print):
    """Exact match and F1 per condition, and the paired contrasts, on the sources that have every answer.
    F1 points with 95% source-bootstrap intervals; AB and BA are averaged within a source."""
    questions, cfg = bundle["questions"], bundle["config"]
    cells = {}
    for r in read_rows(Path(folder) / "main" / "rows.jsonl"):
        _, em, f1 = score.score(r["text"], questions[r["id"]]["answers"])
        cells[(r["condition"].removesuffix("_off"), r["id"])] = (float(em), float(f1), bool(r["cut"]))
    conditions = sorted({c for c, _ in cells}, key=lambda c: (list(wmac_hotpot.CONDITIONS) + [c]).index(c))
    sources = [s for s in dict.fromkeys(id.split(":")[0] for _, id in cells)
               if all((c, f"{s}:{layout}:s0") in cells for c in conditions for layout in ("AB", "BA"))]
    if not sources:
        out("no source has every answer yet")
        return None
    def per_source(condition, metric, keep=sources):
        return [(cells[(condition, f"{s}:AB:s0")][metric] + cells[(condition, f"{s}:BA:s0")][metric]) / 2 for s in keep]
    out(f"{len(sources)} sources with every answer ({len(conditions)} conditions x AB, BA)")
    out("condition            exact match / F1    answers cut at the cap")
    for c in conditions:
        cut = sum(cells[(c, f"{s}:{layout}:s0")][2] for s in sources for layout in ("AB", "BA"))
        out(f"  {c:18s} {100 * statistics.mean(per_source(c, 0)):5.1f} / {100 * statistics.mean(per_source(c, 1)):5.1f}   {cut}")
    probes = collections.defaultdict(dict)
    for r in read_rows(Path(folder) / "probes" / "rows.jsonl"):
        _, _, f1 = score.score(r["text"], questions[r["id"]]["answers"])
        probes[r["source_id"]][r["condition"]] = (f1, r["cut"])
    strict = [s for s in sources if len(probes[s]) == len(wmac_hotpot.PROBES)
              and all(f1 < 0.5 and not cut for f1, cut in probes[s].values())]
    result = {}
    for a, b in CONTRASTS:
        if a in conditions and b in conditions:
            d = [x - y for x, y in zip(per_source(a, 1), per_source(b, 1))]
            line = f"  {a} - {b}: {interval(d, cfg['bootstrap_resamples'])}"
            result[f"{a}-{b}"] = 100 * statistics.mean(d)
            if len(strict) >= 20:
                e = [x - y for x, y in zip(per_source(a, 1, strict), per_source(b, 1, strict))]
                line += f"   | sources failing all three probes (n={len(strict)}): {interval(e, cfg['bootstrap_resamples'])}"
            out(line)
    return dict(sources=len(sources), strict=len(strict), contrasts=result)


# ---- command line ----------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["prepare", "preflight", "run", "summary"])
    p.add_argument("--prepared", type=Path, default=wmac_hotpot.PREPARED)
    p.add_argument("--model", default="8b", help="8b, llama or 4b (preflight: comma-separated)")
    p.add_argument("--part", choices=["main", "probes"], default="main")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "wmac_hotpot_v1")
    p.add_argument("--conditions", default=",".join(wmac_hotpot.CONDITIONS))
    p.add_argument("--sources", type=int, help="only the first N sources in the frozen order (default: all)")
    p.add_argument("--device")
    p.add_argument("--dtype", help="default: the precision the config gives this model")
    p.add_argument("--repo", help="smoke test only: a small model in place of the real one")
    p.add_argument("--fmt", help="chat format for --repo")
    p.add_argument("--answer-tokens", type=int, help="smoke test only: a smaller answer cap")
    args = p.parse_args(argv)
    if args.command == "prepare":
        existed = args.prepared.exists()
        bundle, _ = wmac_hotpot.prepare(destination=args.prepared)
        print(json.dumps(dict(prepared=str(args.prepared), already_existed=existed, sha256=file_sha(args.prepared),
                              sources=len(bundle["hotpot_ids"]) // 2, sources_walked=len(bundle["selection_audit"]),
                              dropped=bundle["exclusion_counts"]), indent=2))
        return 0
    bundle, _ = wmac_hotpot.load(args.prepared)
    if args.command == "preflight":
        run_preflight(bundle, args.model.split(","))
        return 0
    if args.command == "summary":
        for folder in sorted(path for path in args.out.iterdir() if (path / "main" / "rows.jsonl").exists()):
            print(f"\n===== {folder.name} =====")
            summarize(folder, bundle)
        return 0
    conditions = tuple(args.conditions.split(","))
    if set(conditions) - set(wmac_hotpot.CONDITIONS):
        raise ValueError(f"conditions must be among {wmac_hotpot.CONDITIONS}")
    if (args.repo or args.answer_tokens) and args.out == ROOT / "results" / "wmac_hotpot_v1":
        raise ValueError("a smoke test (--repo or --answer-tokens) needs its own --out")
    model = load_model(bundle["config"], args.model, args.device, args.dtype, args.repo, args.fmt)
    print("loaded", runtime(model), flush=True)
    folder = args.out / (args.model if not args.repo else args.repo.split("/")[-1]) / args.part
    done = run_part(model, bundle, args.model, folder, args.part, conditions, args.sources, args.answer_tokens,
                    stand_in=bool(args.repo), prepared_sha256=file_sha(args.prepared))
    return 0 if done else 2


if __name__ == "__main__":
    raise SystemExit(main())
