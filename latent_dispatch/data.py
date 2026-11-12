"""Benchmark questions: build them once, then load and filter them.

A question is a plain dict:
    id, source_id, dataset, qtype, question, answers (gold first, then aliases)
    workers      paragraphs per worker, in stitching order (what K0 stitches and B1 reads, in the same order)
    gold         where each gold paragraph sits: [{"hop", "worker", "index"}], in chain order
    placement    one worker letter per gold paragraph, in chain order ("ABA"); crossings = letter changes (ABA -> 2)
    hops, crossings, seed, bridges (intermediate answers), decomposition (MuSiQue sub-questions)

Build once (deterministic, seeded; refuses to overwrite):
    python -m latent_dispatch.data toy
    python -m latent_dispatch.data toy_d0
    python -m latent_dispatch.data musique
    python -m latent_dispatch.data hotpotqa
then check the files match everyone else's:  cd benchmark && shasum -a 256 -c SHA256SUMS

The 0-crossing control is BB-type. AA/AAA/AAAA put all gold in the FIRST stitched block, which K0 encodes exactly
like B1 (a causal first block never sees later blocks anyway; only the distractor block differs), so they are a
reference, not the control. BB/BBB/BBBB put all gold in the second block: encoded separately and RoPE-shifted, with
nothing to cross.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark"

MUSIQUE_PATTERNS = {2: ["AA", "BB", "AB", "BA"], 3: ["AAA", "BBB", "AAB", "ABA"],
                    4: ["AAAA", "BBBB", "AABB", "ABBA", "ABAB"]}
HOTPOTQA_PATTERNS = ["AA", "BB", "AB"]
TOY_PATTERNS = ["AB", "BA", "AA", "BB"]


# ------------------------------------------------------------------------------------------------- load

def load(name: str, **where) -> list[dict]:
    """Questions from benchmark/<name>.jsonl, filtered: load("musique", seed=0, placement=["AB", "BA"])."""
    with open(BENCHMARK / f"{name}.jsonl", encoding="utf-8") as f:
        qs = [json.loads(line) for line in f]
    for field, allowed in where.items():
        allowed = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
        qs = [q for q in qs if q[field] in allowed]
    return qs


def per_cell(qs: list[dict], n: int | None) -> list[dict]:
    """The first n questions of every (dataset, qtype, placement, seed) cell. n=None keeps everything.
    MuSiQue and HotpotQA list their source questions in the same order under every seed, so the same n source
    questions come back under each seed (the same question in 3 layouts). Toy seeds are different questions."""
    if n is None:
        return qs
    seen, out = defaultdict(int), []
    for q in qs:
        cell = (q["dataset"], q["qtype"], q["placement"], q["seed"])
        if seen[cell] < n:
            seen[cell] += 1
            out.append(q)
    return out


# ------------------------------------------------------------------------------------------------- placement

def crossings(pattern: str) -> int:
    return sum(a != b for a, b in zip(pattern, pattern[1:]))


def place(gold: list[str], rest: list[str], pattern: str, n_workers: int, rng: random.Random):
    """Put gold paragraphs (in chain order) on the workers the pattern names, pad every worker with distractors to
    the same paragraph count (not token count; ±1 when the total is odd), in random order within each worker.
    -> (workers, gold refs).

    Seed `rng` per question and seed, NOT per pattern: the random draws here don't depend on the pattern, so every
    placement of one question deals the distractors in the same order and lists paragraphs in the same relative
    order. Two placements then differ only in where the gold sits (and the distractor it displaces)."""
    if len(pattern) != len(gold) or not all("A" <= c <= "Z" for c in pattern):
        raise ValueError(f"pattern {pattern!r} needs one worker letter (A-Z) per gold paragraph ({len(gold)})")
    n = max(n_workers, 1 + max(ord(c) - ord("A") for c in pattern))
    total = len(gold) + len(rest)
    sizes = [total // n + (1 if i < total % n else 0) for i in range(n)]
    paragraphs = list(gold) + list(rest)                  # a worker holds indices into this list
    deal = list(range(len(gold), total))
    rng.shuffle(deal)                                     # the order distractors are dealt to workers
    sort_key = [rng.random() for _ in range(total)]       # the order paragraphs are listed within a worker
    held = [[hop for hop, letter in enumerate(pattern) if ord(letter) - ord("A") == w] for w in range(n)]
    for w, items in enumerate(held):
        if len(items) > sizes[w]:
            raise ValueError(f"pattern {pattern!r}: worker {w} holds {len(items)} gold but its quota is {sizes[w]}")
    for w, items in enumerate(held):
        while len(items) < sizes[w]:
            items.append(deal.pop())
        items.sort(key=lambda i: sort_key[i])
    workers = [[paragraphs[i] for i in items] for items in held]
    refs = sorted(({"hop": i, "worker": w, "index": j} for w, items in enumerate(held)
                   for j, i in enumerate(items) if i < len(gold)), key=lambda r: r["hop"])
    return workers, refs


def _question(item, dataset, pattern, seed, workers, gold, hops, **extra):
    return dict(id=f"{item['id']}:{pattern}:s{seed}", dataset=dataset, source_id=item["id"], question=item["question"],
                answers=item["answers"], qtype=item["qtype"], hops=hops, placement=pattern,
                crossings=crossings(pattern), seed=seed, workers=workers, gold=gold,
                bridges=extra.get("bridges", []), decomposition=extra.get("decomposition", []))


# ------------------------------------------------------------------------------------------------- MuSiQue

MUSIQUE_LINEAR = {"2hop": 2, "3hop1": 3, "4hop1": 4}


def musique_items(split="validation") -> list[dict]:
    """MuSiQue-Ans, linear chains only (2hop, 3hop1, 4hop1), answerable, one distinct gold paragraph per hop, gold in
    chain order (Paper A musique_gen.py's filter). The non-linear types branch, so crossings has no chain to count."""
    from datasets import load_dataset

    out = []
    for it in load_dataset("dgslibisey/MuSiQue", split=split):
        qtype = it["id"].split("__")[0]
        if not it["answerable"] or qtype not in MUSIQUE_LINEAR:
            continue
        support = [d["paragraph_support_idx"] for d in it["question_decomposition"]]
        by_idx = {p["idx"]: f"{p['title']}: {p['paragraph_text']}" for p in it["paragraphs"]}
        if (len(support) != MUSIQUE_LINEAR[qtype] or None in support or len(set(support)) != len(support)
                or not all(s in by_idx for s in support)):
            continue
        out.append(dict(
            id=it["id"], qtype=qtype, question=it["question"].strip(),
            answers=[a.strip() for a in [it["answer"], *(it["answer_aliases"] or [])] if a and a.strip()],
            gold=[by_idx[s] for s in support], rest=[by_idx[i] for i in sorted(by_idx) if i not in support],
            bridges=[d["answer"] for d in it["question_decomposition"]][:-1],
            decomposition=[d["question"] for d in it["question_decomposition"]]))
    return out


def musique_questions(seed, patterns=MUSIQUE_PATTERNS, n_workers=2, split="validation", limit_per_type=None):
    out, seen = [], defaultdict(int)
    for it in musique_items(split):
        seen[it["qtype"]] += 1
        if limit_per_type and seen[it["qtype"]] > limit_per_type:
            continue
        hops = MUSIQUE_LINEAR[it["qtype"]]
        for pat in patterns.get(hops, []):
            workers, gold = place(it["gold"], it["rest"], pat, n_workers, random.Random(f"{seed}:{it['id']}"))
            out.append(_question(it, "musique", pat, seed, workers, gold, hops,
                                 bridges=it["bridges"], decomposition=it["decomposition"]))
    return out


# ------------------------------------------------------------------------------------------------- HotpotQA

def hotpotqa_items(split="validation") -> list[dict]:
    """HotpotQA distractor, items with exactly 10 paragraphs (a few ship fewer, which can't split 5/5) and 2 gold.
    Most bridge questions are answerable from one paragraph (Min et al. 2019): read HotpotQA only through Part 0."""
    from datasets import load_dataset

    out = []
    for it in load_dataset("hotpotqa/hotpot_qa", "distractor", split=split):
        ctx = it["context"]
        docs = {t: f"{t}: {' '.join(s)}".strip() for t, s in zip(ctx["title"], ctx["sentences"])}
        gold_titles = sorted(set(it["supporting_facts"]["title"]))
        if len(docs) != 10 or len(gold_titles) != 2 or not all(t in docs for t in gold_titles):
            continue
        out.append(dict(id=it["id"], qtype=it["type"], question=it["question"].strip(),
                        answers=[str(it["answer"]).strip()], gold=[docs[t] for t in gold_titles],
                        rest=[d for t, d in docs.items() if t not in gold_titles]))
    return out


def hotpotqa_questions(seed, patterns=HOTPOTQA_PATTERNS, n_workers=2, split="validation", limit_per_type=None):
    """HotpotQA doesn't say which gold paragraph is hop 1, so chain order is random (Paper A hotpot_gen.py). It is
    drawn once per question, the same under every seed, so a seed changes only the layout (as on MuSiQue)."""
    out, seen = [], defaultdict(int)
    for it in hotpotqa_items(split):
        if it["qtype"] not in ("bridge", "comparison"):
            continue
        seen[it["qtype"]] += 1
        if limit_per_type and seen[it["qtype"]] > limit_per_type:
            continue
        gold = list(it["gold"])
        random.Random(f"order:{it['id']}").shuffle(gold)
        for pat in patterns:
            workers, refs = place(gold, it["rest"], pat, n_workers, random.Random(f"{seed}:{it['id']}"))
            out.append(_question(it, "hotpotqa", pat, seed, workers, refs, 2))
    return out


# ------------------------------------------------------------------------------------------------- build

def build(name: str) -> list[dict]:
    """The benchmark as designed (see README, Datasets)."""
    from . import toy

    if name == "toy":       # toy_d2: 2 distractor shards, so AA and BB exist
        return [q for seed in (0, 1, 2) for hop in (0, 1, 2) for q in toy.questions(hop, 500, seed, TOY_PATTERNS, 2)]
    if name == "toy_d0":    # the pilots' toy: 2 shards, no distractors
        return [q for seed in (0, 1, 2) for hop in (0, 1, 2) for q in toy.questions(hop, 500, seed, ["AB", "BA"], 0)]
    if name == "musique":
        return [q for seed in (0, 1, 2) for q in musique_questions(seed)]
    if name == "hotpotqa":
        return [q for seed in (0, 1, 2) for q in hotpotqa_questions(seed)]
    raise ValueError(f"unknown benchmark {name!r}")


def write(qs: list[dict], path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists. Benchmarks are built once; if you really mean to rebuild, delete it AND "
                         "every saved result that used it.")
    if len({q["id"] for q in qs}) != len(qs):
        raise ValueError("duplicate question ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, sort_keys=True, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        qs = build(name)
        write(qs, BENCHMARK / f"{name}.jsonl")
        print(f"benchmark/{name}.jsonl: {len(qs)} questions")
