"""HotpotQA mask run (wmac_hotpot_v1): every table and check, rebuilt from the archived rows (no GPU).

    python scripts/hotpot_tables.py

F1 points with 95% source-bootstrap intervals (20,000 resamples, seed 0) unless a line says otherwise; AB and BA are
averaged within a source; pooled numbers average the three models within a source. Same scorer and intervals as
scripts/paper_tables.py. JOURNAL.md (2026-11-20) records the output.
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

from latent_dispatch import score, stats, wmac_hotpot  # noqa: E402
from latent_dispatch.paper_plan import digest  # noqa: E402
from latent_dispatch.wmac_plan import supports  # noqa: E402

ARCHIVE = ROOT / "archive" / "wmac_hotpot_v1" / "a100-2026-11-20.tar.gz"
MODELS = {"4b": "Qwen3-4B", "8b": "Qwen3-8B", "llama": "Llama 3.1 8B"}
CONDITIONS = list(wmac_hotpot.CONDITIONS)
CONTRASTS = [("gold", "blocked"), ("gold", "matched_irrelevant"), ("matched_irrelevant", "blocked"),
             ("full", "blocked"), ("gold", "full"), ("b1", "k0")]
PRIMARY = CONTRASTS[:2] + [("b1", "k0")]


def iv(differences, alpha=0.05):
    mean, low, high = stats.bootstrap_ci(differences, n_boot=20_000, alpha=alpha, seed=0)
    return f"{100 * mean:+.1f} ({100 * low:.1f}, {100 * high:.1f})"


def load():
    """-> (questions, sources, cells {(model, condition, layout id): (em, f1, row)}, probes {(model, source, probe): (em, f1, cut)})"""
    bundle, questions = wmac_hotpot.load()
    sources = list(dict.fromkeys(id.split(":")[0] for id in bundle["hotpot_ids"]))
    cells, probes = {}, {}
    with tarfile.open(ARCHIVE) as tar:
        for model in MODELS:
            for part in ("main", "probes"):
                rows = [json.loads(line) for line in tar.extractfile(f"wmac_hotpot_v1/{model}/{part}/rows.jsonl").read().splitlines() if line.strip()]
                for r in rows:
                    if r["question_sha256"] != digest(questions[r["id"]]):
                        raise ValueError(f"{model} {part} {r['id']}: row from another question set")
                    _, em, f1 = score.score(r["text"], questions[r["id"]]["answers"])
                    if part == "main":
                        cells[(model, r["condition"].removesuffix("_off"), r["id"])] = (float(em), float(f1), r)
                    else:
                        probes[(model, r["source_id"], r["condition"])] = (float(em), float(f1), bool(r["cut"]))
    if len(cells) != len(MODELS) * len(CONDITIONS) * 2 * len(sources) or len(probes) != len(MODELS) * 3 * len(sources):
        raise ValueError("the archive does not hold every answer exactly once")
    return bundle, questions, sources, cells, probes


def per_source(cells, model, condition, keep, metric=1, layouts=("AB", "BA")):
    return [sum(cells[(model, condition, f"{s}:{layout}:s0")][metric] for layout in layouts) / len(layouts) for s in keep]


def diff(cells, model, a, b, keep, metric=1, layouts=("AB", "BA")):
    return [x - y for x, y in zip(per_source(cells, model, a, keep, metric, layouts), per_source(cells, model, b, keep, metric, layouts))]


def pooled(cells, a, b, keep, layouts=("AB", "BA")):
    return [sum(v) / len(v) for v in zip(*[diff(cells, m, a, b, keep, 1, layouts) for m in MODELS])]


def strict(probes, model, sources):
    return [s for s in sources if all(probes[(model, s, p)][1] < 0.5 and not probes[(model, s, p)][2] for p in wmac_hotpot.PROBES)]


def _norm(text):
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def main():
    bundle, questions, sources, cells, probes = load()
    print(f"## HotpotQA bridge questions, {len(sources)} sources, three models\n")
    rows = [r for _, _, r in cells.values()]
    print("run:", sorted({r["commit"][:7] for r in rows}), sorted({r["device_name"] for r in rows}),
          "answer cap", sorted({r["options"]["answer_tokens"] for r in rows}), "thinking", sorted({r["options"]["thinking"] for r in rows}))

    print("\n### Levels, exact match / F1\n\n| | " + " | ".join(MODELS.values()) + " |\n|---|---|---|---|")
    for c in CONDITIONS:
        print(f"| {c} | " + " | ".join(f"{100 * statistics.mean(per_source(cells, m, c, sources, 0)):.1f} / "
                                        f"{100 * statistics.mean(per_source(cells, m, c, sources)):.1f}" for m in MODELS) + " |")

    print("\n### Paired F1 differences\n\n| | " + " | ".join(MODELS.values()) + " | three models pooled |\n|---|---|---|---|---|")
    for a, b in CONTRASTS + [("b1", "full"), ("blocked", "k0")]:
        print(f"| {a} - {b} | " + " | ".join(iv(diff(cells, m, a, b, sources)) for m in MODELS) + f" | {iv(pooled(cells, a, b, sources))} |")
    total = [x + y + z for x, y, z in zip(pooled(cells, "b1", "full", sources), pooled(cells, "full", "blocked", sources), pooled(cells, "blocked", "k0", sources))]
    print("\nb1 - k0 = (b1 - full) + (full - blocked) + (blocked - k0) for every source:",
          all(abs(t - u) < 1e-9 for t, u in zip(total, pooled(cells, "b1", "k0", sources))))

    print("\n### Primary contrasts at stricter levels\n")
    for a, b in PRIMARY:
        print(f"{a} - {b}: " + "; ".join(f"{MODELS[m]} 98.3% {iv(diff(cells, m, a, b, sources), 0.05 / 3)}, 99% {iv(diff(cells, m, a, b, sources), 0.01)}" for m in MODELS)
              + f"; pooled 99% {iv(pooled(cells, a, b, sources), 0.01)}")

    print("\n### Shortcut probes (AB layouts), exact match, and the stricter subset\n")
    for m in MODELS:
        rates = [100 * statistics.mean(probes[(m, s, p)][0] for s in sources) for p in wmac_hotpot.PROBES]
        joint = 100 * statistics.mean(cells[(m, "b1", f"{s}:AB:s0")][0] for s in sources)
        keep = strict(probes, m, sources)
        print(f"{MODELS[m]}: closed book {rates[0]:.0f}, hop-1 paragraph alone {rates[1]:.0f}, answer paragraph alone {rates[2]:.0f}, joint {joint:.0f}; "
              f"all three probes F1 < 0.5 and none cut: {len(keep)} sources")
        print("  on that subset: " + "; ".join(f"{a} - {b} {iv(diff(cells, m, a, b, keep))}" for a, b in CONTRASTS))

    print("\n### By hop order, three models pooled (AB: hop 1 in A; BA: the answer paragraph in A)\n")
    for layout in ("AB", "BA"):
        print(f"{layout}: " + "; ".join(f"{a} - {b} {iv(pooled(cells, a, b, sources, (layout,)))}" for a, b in CONTRASTS[:4]))
    for a, b in CONTRASTS[:4]:
        d = [x - y for x, y in zip(pooled(cells, a, b, sources, ("BA",)), pooled(cells, a, b, sources, ("AB",)))]
        print(f"BA minus AB, {a} - {b}: {iv(d)}")

    print("\n### Answers cut at the answer cap\n")
    for m in MODELS:
        cut = collections.Counter(c for (mm, c, _), (_, _, r) in cells.items() if mm == m and r["cut"])
        right = sum(em for (mm, _, _), (em, _, r) in cells.items() if mm == m and r["cut"])
        print(f"{MODELS[m]}: of 400 per condition {{{', '.join(f'{c}: {cut[c]}' for c in CONDITIONS)}}}, total {sum(cut.values())}, exact matches among them {int(right)}")
        if sum(cut.values()) > 5:
            for a, b in CONTRASTS:
                bad = {id.split(":")[0] for (mm, c, id), (_, _, r) in cells.items() if mm == m and r["cut"] and c in (a, b)}
                keep = [s for s in sources if s not in bad]
                print(f"  {a} - {b}: all {iv(diff(cells, m, a, b, sources))} | no cut answer in either condition (n={len(keep)}) {iv(diff(cells, m, a, b, keep))}")

    print("\n### Equal-token window: position, and mentions in its earlier context\n")
    clean_all = set(sources)
    for m in MODELS:
        flagged, precede = set(), 0
        for (mm, c, id), (_, _, r) in cells.items():
            if mm == m and c == "matched_irrelevant":
                precede += bool(r["mask"]["selected_keys_precede_gold"])
                if r["mask"]["earlier_prefix_mentions_answer_or_bridge"]:
                    flagged.add(id.split(":")[0])
        keep = [s for s in sources if s not in flagged]
        clean_all &= set(keep)
        print(f"{MODELS[m]}: window precedes the support in {precede} of 400 layouts; earlier text mentions the answer or bridge for {len(flagged)} sources; "
              f"without them (n={len(keep)}): gold - window {iv(diff(cells, m, 'gold', 'matched_irrelevant', keep))}, window - blocked {iv(diff(cells, m, 'matched_irrelevant', 'blocked', keep))}")
    keep = [s for s in sources if s in clean_all]
    print(f"three models pooled, clean in all three (n={len(keep)}): gold - window {iv(pooled(cells, 'gold', 'matched_irrelevant', keep))}, "
          f"window - blocked {iv(pooled(cells, 'matched_irrelevant', 'blocked', keep))}")

    print("\n### Questions that name the answer paragraph's title\n")
    named = [s for s in sources if score.contains_answer(questions[f"{s}:AB:s0"]["question"], [wmac_hotpot.title(supports(questions[f"{s}:AB:s0"])[1])])]
    rest = [s for s in sources if s not in set(named)]
    print(f"{len(named)} of {len(sources)} questions name it. Without them (n={len(rest)}), pooled: "
          + "; ".join(f"{a} - {b} {iv(pooled(cells, a, b, rest))}" for a, b in CONTRASTS[:4]))

    print("\n### Gold against full: exact-match disagreements (400 layouts per model)\n")
    total = collections.Counter()
    for m in MODELS:
        kinds = collections.Counter()
        for (mm, c, id), (em, f1, r) in cells.items():
            if mm != m or c != "gold":
                continue
            em_full, f1_full, r_full = cells[(m, "full", id)]
            if em == em_full:
                continue
            winner, loser_f1, loser = ("gold", f1_full, r_full) if em > em_full else ("full", f1, r)
            answers = questions[id]["answers"]
            prediction = _norm(str(score.score(loser["text"], answers)[0]))
            contains = bool(prediction) and any((a and a in prediction) or (prediction in a) for a in map(_norm, answers))
            kind = ("losing answer cut off" if loser["cut"] else "one answer contains the other" if contains else
                    "partial overlap, F1 >= 0.5" if loser_f1 >= 0.5 else "partial overlap, F1 < 0.5" if loser_f1 > 0 else "different answer, F1 = 0")
            kinds[(winner, kind)] += 1
            total[(winner, kind)] += 1
        print(f"{MODELS[m]}: gold only {sum(v for (w, _), v in kinds.items() if w == 'gold')}, full only {sum(v for (w, _), v in kinds.items() if w == 'full')}")
    for (winner, kind), n in sorted(total.items()):
        print(f"  all models, {winner} right, {kind}: {n}")


if __name__ == "__main__":
    main()
