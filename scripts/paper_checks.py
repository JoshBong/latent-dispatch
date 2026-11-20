"""Checks behind the paper's robustness statements, rebuilt from the archived rows (no GPU).

    python scripts/paper_checks.py                 # every check
    python scripts/paper_checks.py caps split      # only the named ones

Checks: caps, split, levels, position, halves, disagreements, prefix, probes, toy, screen.
F1 points with 95% source-bootstrap intervals (20,000 resamples, seed 0) unless a line says otherwise.
Readers, scorer and intervals are those of scripts/paper_tables.py. JOURNAL.md (2026-11-19) records the output.
"""
import collections
import json
import re
import statistics
import string
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_tables as pt  # noqa: E402
from latent_dispatch import score, stats  # noqa: E402

MASKS = ["blocked", "matched_irrelevant", "gold", "full", "b1", "k0"]
PRIMARY = [("gold", "blocked"), ("gold", "matched_irrelevant"), ("b1", "k0")]   # experiments/wmac_gpu_v3.json
_cache = {}


def source(layout_id):
    return layout_id.split(":")[0]


def main(model, qs):
    if model not in _cache:
        _cache[model] = pt.main_run(model, qs)
    return _cache[model]


def iv(differences, alpha=0.05):
    mean, low, high = stats.bootstrap_ci(differences, n_boot=20_000, alpha=alpha, seed=0)
    return f"{100 * mean:+.1f} ({100 * low:.1f}, {100 * high:.1f})"


# ---- answer-cap hits ---------------------------------------------------------------------------------------------

def caps(qs):
    """Answers cut at the answer cap: how many per condition, and the contrasts without the sources that have one."""
    print("## Answer-cap hits by condition, and contrasts without capped sources\n")
    for model in pt.MODELS:
        scores, sources, _ = main(model, qs)
        cut, f1_of_cut, right = collections.Counter(), collections.defaultdict(list), collections.Counter()
        for (cond, lid), (em, f1, r) in scores.items():
            if r["cut"]:
                cut[cond] += 1
                right[cond] += em
                f1_of_cut[cond].append(f1)
        total = sum(cut.values())
        print(f"{model}, main run: capped answers of 400 per condition {{{', '.join(f'{c}: {cut[c]}' for c in MASKS)}}}, total {total}")
        if not total:
            continue
        any_cut = {source(lid) for (_, lid), (_, _, r) in scores.items() if r["cut"]}
        print(f"  capped answers that are an exact match: {int(sum(right.values()))}; mean F1 of capped answers per condition: "
              + ", ".join(f"{c} {100 * statistics.mean(f1_of_cut[c]):.1f}" for c in MASKS if f1_of_cut[c])
              + f"; sources with a capped answer in some condition: {len(any_cut)}")
        for a, b in [("gold", "blocked"), ("gold", "matched_irrelevant"), ("matched_irrelevant", "blocked"),
                     ("gold", "full"), ("full", "blocked"), ("b1", "k0")]:
            pair_cut = {source(lid) for (cond, lid), (_, _, r) in scores.items() if r["cut"] and cond in (a, b)}
            keep = [s for s in sources if s not in pair_cut]
            clean = [s for s in sources if s not in any_cut]
            zero = {k: ((0.0, 0.0, r) if r["cut"] else (em, f1, r)) for k, (em, f1, r) in scores.items()}
            print(f"  {a} - {b}: all {iv(pt.difference(scores, a, b, sources))} | no cap in either condition (n={len(keep)}) "
                  f"{iv(pt.difference(scores, a, b, keep))} | no cap in any condition (n={len(clean)}) "
                  f"{iv(pt.difference(scores, a, b, clean))} | capped scored 0 {iv(pt.difference(zero, a, b, sources))}")
    print("\nControl layout (both supporting paragraphs in one worker), full - blocked:")
    kept = {}
    for model in pt.TOGETHER_FOLDER:
        together = pt.together_run(model, qs)
        scores, sources, _ = main(model, qs)
        cut = collections.Counter((cond, lid.split(":")[1]) for (cond, lid), (_, _, r) in together.items() if r["cut"])
        capped = {source(lid) for (_, lid), (_, _, r) in together.items() if r["cut"]}
        keep = [s for s in sources if s not in capped]
        kept[model] = (together, scores, sources, keep)
        print(f"{model}: capped answers {sum(cut.values())} of {len(together)} {dict(cut)}; sources with one {len(capped)}")
        for label, subset in (("all", sources), (f"no cap in the control layout (n={len(keep)})", keep)):
            split = pt.difference(scores, "full", "blocked", subset)
            tog = pt.difference(together, "full", "blocked", subset, 1, ("AA", "BB"))
            print(f"  {label}: split {iv(split)} | together {iv(tog)} | split - together {iv([x - y for x, y in zip(split, tog)])}")
    common = [s for s in next(iter(kept.values()))[2] if all(s in set(v[3]) for v in kept.values())]
    for label, subset in (("all", next(iter(kept.values()))[2]), (f"no cap in the control layout (n={len(common)})", common)):
        tog = pt.pooled([pt.difference(v[0], "full", "blocked", subset, 1, ("AA", "BB")) for v in kept.values()])
        split = pt.pooled([pt.difference(v[1], "full", "blocked", subset) for v in kept.values()])
        print(f"two models pooled, {label}: together {iv(tog)} | split - together {iv([x - y for x, y in zip(split, tog)])}")
    print("\nPosition run (A's supporting paragraph first):")
    for model in pt.POSITION:
        position = pt.position_run(model, qs)
        cut = collections.Counter(cond for (cond, _), (_, _, r) in position.items() if r["cut"])
        print(f"{model}: capped answers {sum(cut.values())} of {len(position)} {dict(cut)}")
        if sum(cut.values()) > 1:
            sources = pt.sources_of(position)
            for a, b in [("gold", "matched_late"), ("gold", "blocked"), ("gold", "full")]:
                bad = {source(lid) for (cond, lid), (_, _, r) in position.items() if r["cut"] and cond in (a, b)}
                keep = [s for s in sources if s not in bad]
                print(f"  {a} - {b}: all {iv(pt.difference(position, a, b, sources))} | no cap in either (n={len(keep)}) "
                      f"{iv(pt.difference(position, a, b, keep))}")
    print()


# ---- joint reading minus K0, split through the masks ----------------------------------------------------------------

def split(qs):
    """B1 - K0 = (B1 - full mask) + (full mask - blocked) + (blocked - K0), exactly, source by source."""
    print("## Joint reading minus K0, split into three steps (each step changes one thing)\n")
    steps = [("b1", "k0"), ("b1", "full"), ("full", "blocked"), ("blocked", "k0")]
    per_model = collections.defaultdict(list)
    for model in pt.MODELS:
        scores, sources, _ = main(model, qs)
        for a, b in steps:
            d = pt.difference(scores, a, b, sources)
            per_model[(a, b)].append(d)
            print(f"{model}: {a} - {b} {iv(d)}")
    pooled = {s: pt.pooled(v) for s, v in per_model.items()}
    for a, b in steps:
        print(f"three models pooled: {a} - {b} {iv(pooled[(a, b)])}")
    total = [x + y + z for x, y, z in zip(pooled[("b1", "full")], pooled[("full", "blocked")], pooled[("blocked", "k0")])]
    print("the three steps sum to b1 - k0 for every source:",
          all(abs(t - u) < 1e-9 for t, u in zip(total, pooled[("b1", "k0")])), "\n")


# ---- stricter interval levels ----------------------------------------------------------------------------------------

def levels(qs):
    """The three contrasts the run configuration marks primary, at 95%, 98.3% (three contrasts) and 99%."""
    print("## Primary contrasts at stricter interval levels\n")
    per_model = collections.defaultdict(list)
    for model in pt.MODELS:
        scores, sources, _ = main(model, qs)
        for a, b in PRIMARY + [("matched_irrelevant", "blocked"), ("full", "blocked")]:
            d = pt.difference(scores, a, b, sources)
            per_model[(a, b)].append(d)
            if (a, b) in PRIMARY:
                print(f"{model}: {a} - {b}: 95% {iv(d)} | 98.3% {iv(d, 0.05 / 3)} | 99% {iv(d, 0.01)}")
    for (a, b), v in per_model.items():
        d = pt.pooled(v)
        print(f"three models pooled: {a} - {b}: 95% {iv(d)} | 99% {iv(d, 0.01)}")
    runs = {m: (pt.together_run(m, qs), *main(m, qs)[:2]) for m in pt.TOGETHER_FOLDER}
    together = pt.pooled([pt.difference(t, "full", "blocked", src, 1, ("AA", "BB")) for t, _, src in runs.values()])
    split_effect = pt.pooled([pt.difference(sc, "full", "blocked", src) for _, sc, src in runs.values()])
    inter = [x - y for x, y in zip(split_effect, together)]
    print(f"control layout, two models pooled: together 95% {iv(together)} | 99% {iv(together, 0.01)}; "
          f"split - together 95% {iv(inter)} | 99% {iv(inter, 0.01)}\n")


# ---- position run and control-layout halves --------------------------------------------------------------------------

def position(qs):
    """Position run: A's support first, the equal-token window at the end of A (next to B, after the support)."""
    print("## Position run: the window follows the support and sits next to B\n")
    per_model = collections.defaultdict(list)
    for model in pt.POSITION:
        scores = pt.position_run(model, qs)
        sources = pt.sources_of(scores)
        for a, b in [("gold", "matched_late"), ("matched_late", "blocked"), ("gold", "blocked")]:
            d = pt.difference(scores, a, b, sources)
            per_model[(a, b)].append(d)
            print(f"{model}: {a} - {b} {iv(d)}")
    for (a, b), v in per_model.items():
        print(f"two models pooled: {a} - {b} {iv(pt.pooled(v))}")
    print()


def halves(qs):
    """Control layout by placement: both supports in A (AA) or both in B (BB)."""
    print("## Control layout by placement, full - blocked\n")
    per_layout = collections.defaultdict(list)
    for model in pt.TOGETHER_FOLDER:
        together = pt.together_run(model, qs)
        sources = main(model, qs)[1]
        for layout in ("AA", "BB"):
            d = pt.difference(together, "full", "blocked", sources, 1, (layout,))
            per_layout[layout].append(d)
            print(f"{model}: {layout} {iv(d)}")
    for layout, v in per_layout.items():
        print(f"two models pooled: {layout} {iv(pt.pooled(v))}")
    print()


# ---- gold against full, question by question ---------------------------------------------------------------------------

def _norm(text):
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def disagreements(qs):
    """Layouts where exactly one of gold and full is an exact match, by what the losing answer looks like."""
    print("## Gold against full: exact-match disagreements (400 layouts per model)\n")
    total = collections.Counter()
    for model in pt.MODELS:
        scores, _, _ = main(model, qs)
        kinds = collections.Counter()
        for (cond, lid), (em, f1, r) in scores.items():
            if cond != "gold":
                continue
            em_full, f1_full, r_full = scores[("full", lid)]
            if em == em_full:
                continue
            winner, loser_f1, loser = ("gold", f1_full, r_full) if em > em_full else ("full", f1, r)
            answers = qs[lid]["answers"]
            prediction = _norm(str(score.score(loser["text"], answers)[0]))
            contains = bool(prediction) and any((a and a in prediction) or (prediction in a) for a in map(_norm, answers))
            if loser["cut"]:
                kind = "losing answer cut off"
            elif contains:
                kind = "one answer contains the other"
            elif loser_f1 >= 0.5:
                kind = "partial overlap, F1 >= 0.5"
            elif loser_f1 > 0:
                kind = "partial overlap, F1 < 0.5"
            else:
                kind = "different answer, F1 = 0"
            kinds[(winner, kind)] += 1
            total[(winner, kind)] += 1
        print(f"{model}: gold only {sum(v for (w, _), v in kinds.items() if w == 'gold')}, "
              f"full only {sum(v for (w, _), v in kinds.items() if w == 'full')}")
        for (winner, kind), n in sorted(kinds.items()):
            print(f"  {winner} right, {kind}: {n}")
    print("all models:")
    for (winner, kind), n in sorted(total.items()):
        print(f"  {winner} right, {kind}: {n}")
    print()


# ---- what the equal-token window could have read ------------------------------------------------------------------------

def prefix(qs):
    """Equal-token window: does it precede A's support, and does earlier text mention the answer or the bridge?"""
    print("## Equal-token window: position against the support, and mentions in its earlier context\n")
    clean_everywhere, runs = None, {}
    for model in pt.MODELS:
        scores, sources, _ = main(model, qs)
        flagged, precedes = {}, collections.Counter()
        for (cond, lid), (_, _, r) in scores.items():
            if cond == "matched_irrelevant":
                flagged[lid] = bool(r["mask"]["earlier_prefix_mentions_answer_or_bridge"])
                precedes[r["mask"].get("selected_keys_precede_gold")] += 1
        flagged_sources = {source(lid) for lid, flag in flagged.items() if flag}
        clean = [s for s in sources if s not in flagged_sources]
        runs[model] = (scores, sources)
        clean_everywhere = set(clean) if clean_everywhere is None else clean_everywhere & set(clean)
        print(f"{model}: window precedes the support in {precedes[True]} of {len(flagged)} layouts; earlier text mentions the "
              f"answer or bridge in {sum(flagged.values())} layouts ({len(flagged_sources)} sources)")
        for a, b in [("gold", "matched_irrelevant"), ("matched_irrelevant", "blocked")]:
            print(f"  {a} - {b}: all {iv(pt.difference(scores, a, b, sources))} | without those sources (n={len(clean)}) "
                  f"{iv(pt.difference(scores, a, b, clean))}")
    common = sorted(clean_everywhere)
    for a, b in [("gold", "matched_irrelevant"), ("matched_irrelevant", "blocked")]:
        everyone = pt.pooled([pt.difference(sc, a, b, src) for sc, src in runs.values()])
        clean = pt.pooled([pt.difference(sc, a, b, common) for sc, _ in runs.values()])
        print(f"three models pooled: {a} - {b}: all {iv(everyone)} | clean in all three (n={len(common)}) {iv(clean)}")
    print()


# ---- gold against full on the stricter probe subsets -----------------------------------------------------------------

def probes(qs):
    """Gold - full (and gold - blocked, exact match) on all sources and on the stricter probe subset of each model."""
    print("## Gold - full on the stricter probe subsets\n")
    for model in pt.MODELS:
        scores, sources, cohort = main(model, qs)
        for label, keep in (("all", sources), ("probe subset", cohort)):
            print(f"{model}, {label} (n={len(keep)}): gold - full F1 {iv(pt.difference(scores, 'gold', 'full', keep))} | "
                  f"exact match {iv(pt.difference(scores, 'gold', 'full', keep, 0))} | gold - blocked exact match "
                  f"{iv(pt.difference(scores, 'gold', 'blocked', keep, 0))}")
    print()


# ---- toy: what forced thinking does, answer by answer ------------------------------------------------------------------

def toy(qs):
    """Fresh toy on Qwen3-8B: paired outcomes with thinking off and forced, budget hits, and a repeated phrase."""
    print("## Toy, Qwen3-8B: thinking off against forced, answer by answer\n")
    name = "wmac_gpu_v3/toy-8b/main/rows.jsonl"
    cell = {}
    for r in pt.jsonl(pt.read_tar(pt.A100, [name])[name]):
        chain, order, cities, headings = r["id"].split(":")
        mask, thinking = r["condition"].split("_")
        prediction, em, _ = score.score(r["text"], qs[r["id"]]["answers"])
        cell[(mask, thinking, chain, order, cities, headings)] = (bool(em), r, str(prediction))
    for mask in ("blocked", "gold"):
        for cities in ("split", "together"):
            moves = collections.Counter()
            for (m, thinking, chain, order, c, headings), (em, _, _) in cell.items():
                if (m, thinking, c) == (mask, "off", cities):
                    forced = cell[(m, "forced", chain, order, c, headings)][0]
                    moves[("right" if em else "wrong") + " -> " + ("right" if forced else "wrong")] += 1
            print(f"{mask}, look-alike paragraph {'in the other worker' if cities == 'split' else 'in the same worker'}: {dict(moves)}")
    forced = [(k, v) for k, v in cell.items() if k[1] == "forced"]
    hits = [(k, v) for k, v in forced if v[1]["think_cut"]]
    done = [(k, v) for k, v in forced if not v[1]["think_cut"]]
    print(f"forced answers {len(forced)}; used the whole thinking budget {len(hits)} "
          f"{dict(collections.Counter((k[0], k[4]) for k, _ in hits))}")
    lost = [(k, v) for k, v in forced if (k[0], k[4]) == ("blocked", "split") and not v[0]]
    was_right = [(k, v) for k, v in lost if cell[(k[0], "off", k[2], k[3], k[4], k[5])][0]]
    print(f"blocked, other worker, forced: wrong answers {len(lost)}, extracted as {dict(collections.Counter(v[2] for _, v in lost))}; "
          f"all used the whole budget: {all(v[1]['think_cut'] for _, v in lost)}; right with thinking off: {len(was_right)}")
    exact = [v[1]["text"].count("Let me check again") for _, v in hits]
    anycase = [v[1]["text"].lower().count("let me check again") for _, v in hits]
    anycase_done = [v[1]["text"].lower().count("let me check again") for _, v in done]
    print(f"'Let me check again' four or more times among budget hits: {sum(n >= 4 for n in exact)} of {len(hits)} as written, "
          f"{sum(n >= 4 for n in anycase)} ignoring case; among the {len(done)} that finished: {sum(n >= 4 for n in anycase_done)}")
    print(f"count per budget hit, ignoring case: min {min(anycase)}, median {int(statistics.median(anycase))}, max {max(anycase)}; "
          f"most in an answer that finished: {max(anycase_done)}\n")


# ---- the first screen: how often a question needs both paragraphs ------------------------------------------------------

def screen(_qs):
    """Qwen3-4B, first run (11-13): closed book, joint reading, and each supporting paragraph alone. Exact match.
    Needs benchmark/hotpotqa.jsonl and benchmark/musique.jsonl (`python -m latent_dispatch.data musique hotpotqa`)."""
    print("## First screen (Qwen3-4B, 11-13): how often a question needs both supporting paragraphs\n")
    files = [ROOT / "benchmark" / "hotpotqa.jsonl", ROOT / "benchmark" / "musique.jsonl"]
    if not all(f.exists() for f in files):
        print("benchmark files are not built here: run `python -m latent_dispatch.data musique hotpotqa` first\n")
        return
    bench = {}
    for path in files:
        with open(path) as handle:
            for line in handle:
                item = json.loads(line)
                if item["dataset"] == "hotpotqa" or item["id"].startswith("2hop"):
                    bench[item["id"]] = item
    arms = {}
    with tarfile.open(ROOT / "archive" / "Qwen3-4B-bfloat16-2026-11-13.tar.gz") as tar:
        for arm in ("b0", "b1", "b1_only_hop1", "b1_only_hop2"):
            rows = pt.jsonl(tar.extractfile(f"results/Qwen3-4B-bfloat16/{arm}.jsonl").read())
            arms[arm] = {r["id"]: bool(score.score(r["text"], bench[r["id"]]["answers"])[1]) for r in rows if r["id"] in bench}
    groups = {"HotpotQA bridge": lambda it: it["dataset"] == "hotpotqa" and it["qtype"] == "bridge",
              "HotpotQA comparison": lambda it: it["dataset"] == "hotpotqa" and it["qtype"] == "comparison",
              "MuSiQue 2-hop": lambda it: it["dataset"] != "hotpotqa"}
    for label, keep in groups.items():
        ids = [i for i in arms["b0"] if keep(bench[i]) and all(i in arms[a] for a in arms)]
        rate = lambda arm: 100 * sum(arms[arm][i] for i in ids) / len(ids)      # noqa: E731
        need = 100 * sum(not arms["b0"][i] and not arms["b1_only_hop1"][i] and not arms["b1_only_hop2"][i] for i in ids) / len(ids)
        print(f"{label} (n={len(ids)}): closed book {rate('b0'):.0f} | joint {rate('b1'):.0f} | one supporting paragraph "
              f"{rate('b1_only_hop1'):.0f} / {rate('b1_only_hop2'):.0f} | wrong closed book and with each paragraph alone {need:.0f}%")
    print()


CHECKS = {"caps": caps, "split": split, "levels": levels, "position": position, "halves": halves,
          "disagreements": disagreements, "prefix": prefix, "probes": probes, "toy": toy, "screen": screen}


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(CHECKS)
    unknown = [name for name in wanted if name not in CHECKS]
    if unknown:
        sys.exit(f"unknown check {unknown}; choose from {list(CHECKS)}")
    questions = pt.questions()
    for name in wanted:
        CHECKS[name](questions)
