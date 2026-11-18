"""Rebuild the fresh-source tables from the archived rows. No GPU and no model: it reads the four tarballs in
archive/wmac_gpu_v3/, re-scores every saved answer with latent_dispatch.score, and prints the tables quoted in
JOURNAL.md (the 2026-11-18 entries) as markdown.

    python scripts/paper_tables.py              # every table (a few seconds)
    python scripts/paper_tables.py main toy     # some of: main, position, together, checks, validation, toy

Unit of analysis: the source question. Its layouts (AB and BA; AA and BB in the control layout) are averaged first;
intervals are 95% percentile bootstrap intervals over sources (stats.bootstrap_ci, 20,000 resamples, seed 0). Toy
intervals are over chains.
"""
import collections
import json
import statistics
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_dispatch import score, stats  # noqa: E402

ARCHIVE = ROOT / "archive" / "wmac_gpu_v3"
A100 = ARCHIVE / "a100-no-logits-2026-11-17.tar.gz"      # Qwen3-8B and Llama: main run, validation, toy
CHECKS = ARCHIVE / "a100-checks-2026-11-18.tar.gz"       # second machine: mask checks, support-first position run
L4 = ARCHIVE / "l4-4b-no-logits-2026-11-18.tar.gz"       # Qwen3-4B: main run, validation
TOGETHER = ARCHIVE / "a100-together-2026-11-18.tar.gz"   # control layout: both supporting paragraphs in one worker

# model -> (tarball, folder of the MuSiQue job, folder of the validation job)
MODELS = {
    "Qwen3-8B": (A100, "wmac_gpu_v3/musique-8b", "wmac_gpu_v3/validation"),
    "Llama 3.1 8B": (A100, "wmac_gpu_v3/musique-llama", "wmac_gpu_v3/validation-llama"),
    "Qwen3-4B": (L4, "wmac_gpu_v3_l4/musique-4b", "wmac_gpu_v3_l4/validation-4b"),
}
POSITION = {"Qwen3-8B": "wmac_gpu_v3_checks/8b", "Llama 3.1 8B": "wmac_gpu_v3_checks/llama"}   # in CHECKS
TOGETHER_FOLDER = {"Qwen3-8B": "wmac_gpu_v3_together/8b", "Llama 3.1 8B": "wmac_gpu_v3_together/llama"}   # in TOGETHER

LABEL = {"b1": "joint reading", "full": "full mask", "gold": "gold", "matched_irrelevant": "equal-token window",
         "matched_late": "equal-token window next to B", "blocked": "blocked", "k0": "K0"}
MAIN_CONTRASTS = [("gold", "blocked"), ("gold", "matched_irrelevant"), ("matched_irrelevant", "blocked"),
                  ("gold", "full"), ("full", "blocked"), ("b1", "k0"), ("blocked", "k0")]
POSITION_CONTRASTS = [("gold", "blocked"), ("gold", "matched_late"), ("matched_late", "blocked"),
                      ("gold", "full"), ("full", "blocked")]


# ---- reading and scoring ---------------------------------------------------------------------------------------

def read_tar(path, wanted):
    """{member name: bytes} for the wanted members of one tarball (one pass over it)."""
    with tarfile.open(path) as tar:
        found = {m.name: tar.extractfile(m).read() for m in tar if m.name in wanted}
    missing = set(wanted) - set(found)
    if missing:
        raise FileNotFoundError(f"{path.name} has no {sorted(missing)}")
    return found


def jsonl(blob):
    return [json.loads(line) for line in blob.splitlines() if line.strip()]


def questions():
    """Layouts and accepted answers of every question, keyed by layout id. One bundle, shared by all three runs."""
    return json.loads(read_tar(A100, ["wmac_gpu_v3/prepared.json"])["wmac_gpu_v3/prepared.json"])["questions"]


def layout_scores(rows, qs):
    """{(condition, layout id): (exact match, F1, row)}. Conditions lose their '_off' suffix; '_forced' stays."""
    out = {}
    for r in rows:
        _, em, f1 = score.score(r["text"], qs[r["id"]]["answers"])
        key = (r["condition"].removesuffix("_off"), r["id"])
        if key in out:
            raise ValueError(f"duplicate row {key}")
        out[key] = (float(em), float(f1), r)
    return out


def sources_of(scores):
    return sorted({layout.split(":")[0] for _, layout in scores})


def per_source(scores, condition, metric, sources, layouts=("AB", "BA")):
    """One number per source: the mean over its layouts (AB and BA unless said). metric 0 = exact match, 1 = F1."""
    return [sum(scores[(condition, f"{s}:{layout}:s0")][metric] for layout in layouts) / len(layouts) for s in sources]


def difference(scores, a, b, sources, metric=1, layouts=("AB", "BA")):
    return [x - y for x, y in zip(per_source(scores, a, metric, sources, layouts),
                                  per_source(scores, b, metric, sources, layouts))]


def interval(differences):
    """(mean, low, high) in points."""
    mean, low, high = stats.bootstrap_ci(differences, n_boot=20_000, seed=0)
    return round(100 * mean, 1), round(100 * low, 1), round(100 * high, 1)


def show(triple):
    mean, low, high = (value + 0.0 for value in triple)      # + 0.0 turns a rounded −0.0 into 0.0
    return f"{mean:+.1f} ({low:.1f}, {high:.1f})".replace("-", "−")


def probe_cohort(probe_rows, qs, sources):
    """The runner's stricter set: all three shortcut probes of the source score F1 < 0.5 and none was cut off."""
    seen = collections.defaultdict(dict)
    for r in probe_rows:
        _, _, f1 = score.score(r["text"], qs[r["id"]]["answers"])
        seen[r["id"].split(":")[0]][r["condition"]] = (f1, r["cut"])
    return [s for s in sources if len(seen[s]) == 3 and all(f1 < 0.5 and not cut for f1, cut in seen[s].values())]


def main_run(model, qs):
    """Scores, sources and probe cohort of one model's main run (every supporting paragraph last in its block)."""
    tarball, job, _ = MODELS[model]
    blobs = read_tar(tarball, [f"{job}/main/rows.jsonl", f"{job}/probes/rows.jsonl"])
    scores = layout_scores(jsonl(blobs[f"{job}/main/rows.jsonl"]), qs)
    sources = sources_of(scores)
    return scores, sources, probe_cohort(jsonl(blobs[f"{job}/probes/rows.jsonl"]), qs, sources)


def position_run(model, qs):
    """Scores of the support-first run: A's supporting paragraph moved to the first slot of A, B untouched."""
    name = f"{POSITION[model]}/position/rows.jsonl"
    return layout_scores(jsonl(read_tar(CHECKS, [name])[name]), qs)


def together_run(model, qs):
    """Scores of the control layout: both supporting paragraphs in one worker (AA: in A, BB: in B), blocked and full.
    Its layouts are built from each source's AB layout and are not in the bundle; a row names that layout in base_id,
    and the accepted answers are that layout's."""
    name = f"{TOGETHER_FOLDER[model]}/together/rows.jsonl"
    rows = jsonl(read_tar(TOGETHER, [name])[name])
    return layout_scores(rows, {r["id"]: qs[r["base_id"]] for r in rows})


def pooled(per_model_differences):
    """Average the per-source differences over models (same 200 sources in every run), one number per source."""
    return [sum(values) / len(values) for values in zip(*per_model_differences)]


# ---- tables ----------------------------------------------------------------------------------------------------

def levels(scores, sources, conditions, layouts=("AB", "BA")):
    return {c: (round(100 * statistics.mean(per_source(scores, c, 0, sources, layouts)), 1),
                round(100 * statistics.mean(per_source(scores, c, 1, sources, layouts)), 1)) for c in conditions}


def main_table(qs):
    """Main run, three models: levels, then paired F1 differences on all sources and on each probe cohort."""
    runs = {m: main_run(m, qs) for m in MODELS}
    conditions = ["b1", "full", "gold", "matched_irrelevant", "blocked", "k0"]
    result = {"levels": {m: levels(runs[m][0], runs[m][1], conditions) for m in MODELS}, "contrasts": {}}
    print("### Main run: A's supporting paragraph last. Exact match / F1, 200 sources\n")
    print("| | " + " | ".join(MODELS) + " |\n|---|" + "---|" * len(MODELS))
    for c in conditions:
        print(f"| {LABEL[c]} | " + " | ".join("{:.1f} / {:.1f}".format(*result["levels"][m][c]) for m in MODELS) + " |")
    print("\nPaired F1 differences. Probe cohort sizes: "
          + ", ".join(f"{m} {len(runs[m][2])}" for m in MODELS) + ".\n")
    print("| | " + " | ".join(f"{m}, all 200 | {m}, probe cohort" for m in MODELS) + " | three models pooled |")
    print("|---|" + "---|" * (2 * len(MODELS) + 1))
    for a, b in MAIN_CONTRASTS:
        row = {}
        for m, (scores, sources, cohort) in runs.items():
            row[m] = {"all": interval(difference(scores, a, b, sources)),
                      "probe": interval(difference(scores, a, b, cohort))}
        row["pooled"] = interval(pooled([difference(runs[m][0], a, b, runs[m][1]) for m in MODELS]))
        result["contrasts"][(a, b)] = row
        cells = " | ".join(f"{show(row[m]['all'])} | {show(row[m]['probe'])}" for m in MODELS)
        print(f"| {LABEL[a]} − {LABEL[b]} | {cells} | {show(row['pooled'])} |")
    result["pooled_exact_match"] = {(a, b): interval(pooled([difference(runs[m][0], a, b, runs[m][1], metric=0) for m in MODELS]))
                                    for a, b in [("gold", "blocked"), ("gold", "full")]}
    print("\nExact match, three models pooled: " + "; ".join(
        f"{LABEL[a]} − {LABEL[b]} {show(v)}" for (a, b), v in result["pooled_exact_match"].items()) + "\n")
    return result


def position_table(qs):
    """Support-first run against the main run on the same sources, for the two models that ran both."""
    first = {m: position_run(m, qs) for m in POSITION}
    last = {m: main_run(m, qs) for m in POSITION}
    sources = last["Qwen3-8B"][1]
    conditions = ["b1", "full", "gold", "matched_late", "blocked"]
    result = {"levels": {m: levels(first[m], sources, conditions) for m in POSITION}, "contrasts": {}, "change": {}}
    print("### A's supporting paragraph first. Exact match / F1, 200 sources\n")
    print("| | " + " | ".join(POSITION) + " |\n|---|" + "---|" * len(POSITION))
    for c in conditions:
        print(f"| {LABEL[c]} | " + " | ".join("{:.1f} / {:.1f}".format(*result["levels"][m][c]) for m in POSITION) + " |")
    print("\n| F1 points | " + " | ".join(f"{m}, all 200 | {m}, probe cohort" for m in POSITION) + " | two models pooled |")
    print("|---|" + "---|" * (2 * len(POSITION) + 1))
    for a, b in POSITION_CONTRASTS:
        row = {m: {"all": interval(difference(first[m], a, b, sources)),
                   "probe": interval(difference(first[m], a, b, last[m][2]))} for m in POSITION}
        row["pooled"] = interval(pooled([difference(first[m], a, b, sources) for m in POSITION]))
        result["contrasts"][(a, b)] = row
        cells = " | ".join(f"{show(row[m]['all'])} | {show(row[m]['probe'])}" for m in POSITION)
        print(f"| {LABEL[a]} − {LABEL[b]} | {cells} | {show(row['pooled'])} |")
    print("\nSupport last minus support first, same sources, F1 points:\n")
    print("| | " + " | ".join(POSITION) + " | two models pooled |\n|---|" + "---|" * (len(POSITION) + 1))
    for a, b in [("gold", "full"), ("gold", "blocked"), ("full", "blocked")]:
        per_model = {m: [x - y for x, y in zip(difference(last[m][0], a, b, sources), difference(first[m], a, b, sources))]
                     for m in POSITION}
        row = {m: interval(per_model[m]) for m in POSITION}
        row["pooled"] = interval(pooled(list(per_model.values())))
        result["change"][(a, b)] = row
        print(f"| change in {LABEL[a]} − {LABEL[b]} | " + " | ".join(show(row[m]) for m in POSITION) + f" | {show(row['pooled'])} |")
    print()
    return result


TOGETHER_ROWS = {"split": "one supporting paragraph in each worker (main run, AB and BA)",
                 "together": "both in one worker (AA and BB)", "AA": "both in A (AA)", "BB": "both in B (BB)",
                 "split − together": "first row minus second row"}


def together_table(qs):
    """Control layout against the main run on the same sources: what B's access to A is worth (full mask − blocked)
    when the two supporting paragraphs sit in different workers, and when they sit in the same one."""
    result = {"levels": {}, "contrasts": {"F1": {}, "exact match": {}}}
    per_model, sources = {}, None
    for m in TOGETHER_FOLDER:
        scores = together_run(m, qs)
        main_scores, main_sources, _ = main_run(m, qs)
        sources = sources or main_sources
        if main_sources != sources or sources_of(scores) != sources or len(scores) != 4 * len(sources):
            raise ValueError(f"{m}: the control layout must hold the main run's sources, each in AA and BB, blocked and full")
        result["levels"][m] = {(layout, c): value for layout in ("AA", "BB")
                               for c, value in levels(scores, sources, ["blocked", "full"], (layout,)).items()}
        per_model[m] = {}
        for metric, name in ((1, "F1"), (0, "exact match")):
            d = {"split": difference(main_scores, "full", "blocked", sources, metric),
                 "together": difference(scores, "full", "blocked", sources, metric, ("AA", "BB")),
                 "AA": difference(scores, "full", "blocked", sources, metric, ("AA",)),
                 "BB": difference(scores, "full", "blocked", sources, metric, ("BB",))}
            d["split − together"] = [x - y for x, y in zip(d["split"], d["together"])]
            per_model[m][name] = d
    print(f"### Control layout: both supporting paragraphs in one worker. Exact match / F1, {len(sources)} sources\n")
    print("| | AA, blocked | AA, full mask | BB, blocked | BB, full mask |\n|---|---|---|---|---|")
    for m in TOGETHER_FOLDER:
        print(f"| {m} | " + " | ".join("{:.1f} / {:.1f}".format(*result["levels"][m][(layout, c)])
                                       for layout in ("AA", "BB") for c in ("blocked", "full")) + " |")
    for name in ("F1", "exact match"):
        print(f"\n| full mask − blocked, {name} points | " + " | ".join(TOGETHER_FOLDER) + " | two models pooled |")
        print("|---|" + "---|" * (len(TOGETHER_FOLDER) + 1))
        for key, label in TOGETHER_ROWS.items():
            row = {m: interval(per_model[m][name][key]) for m in TOGETHER_FOLDER}
            row["pooled"] = interval(pooled([per_model[m][name][key] for m in TOGETHER_FOLDER]))
            result["contrasts"][name][key] = row
            print(f"| {label} | " + " | ".join(show(row[m]) for m in TOGETHER_FOLDER) + f" | {show(row['pooled'])} |")
    print()
    return result


def checks_table():
    """Mask checks of the second machine: the swap test and the two pairs that should be the same computation."""
    result = {}
    print("### Mask checks, 48 layouts per model\n")
    print("| | swap under blocked mask: B identical at every layer | swap under full mask: B changed | full mask = joint "
          "reading at every layer, one kernel | same, default kernels | first-token logits, full vs joint, default kernels: "
          "median, largest | blocked mask vs separate blocks: median, largest | same top token |")
    print("|---|---|---|---|---|---|---|---|")
    for model, folder in POSITION.items():
        name = f"{folder}/verify/rows.jsonl"
        rows = jsonl(read_tar(CHECKS, [name])[name])
        default, one = [r["pairs_default_kernels"] for r in rows], [r["pairs_one_kernel"] for r in rows]
        full_gap = [p["joint_vs_full_mask"]["first_logits"]["largest"] for p in default]
        blocked_gap = [p["separate_vs_blocked_mask"]["first_logits"]["largest"] for p in default]
        result[model] = {
            "layouts": len(rows),
            "swap_blocked_identical": sum(r["swap"]["blocked"]["identical_layers"] == r["swap"]["blocked"]["layers"]
                                          and r["swap"]["blocked"]["last_logits_identical"] for r in rows),
            "swap_full_changed": sum(r["swap"]["full"]["largest"] > 0 for r in rows),
            "full_equals_joint_one_kernel": sum(p["joint_vs_full_mask"]["identical_layers"] == p["joint_vs_full_mask"]["layers"]
                                                for p in one),
            "full_equals_joint_default": sum(p["joint_vs_full_mask"]["identical_layers"] == p["joint_vs_full_mask"]["layers"]
                                             for p in default),
            "full_first_logit_gap": (round(statistics.median(full_gap), 2), round(max(full_gap), 2)),
            "blocked_first_logit_gap": (round(statistics.median(blocked_gap), 2), round(max(blocked_gap), 2)),
            "blocked_same_top_token": sum(p["separate_vs_blocked_mask"]["first_logits"]["same_top_token"] for p in default),
        }
        v = result[model]
        print(f"| {model} | {v['swap_blocked_identical']} of {v['layouts']} | {v['swap_full_changed']} of {v['layouts']} | "
              f"{v['full_equals_joint_one_kernel']} of {v['layouts']} | {v['full_equals_joint_default']} of {v['layouts']} | "
              f"{v['full_first_logit_gap'][0]}, {v['full_first_logit_gap'][1]} | "
              f"{v['blocked_first_logit_gap'][0]}, {v['blocked_first_logit_gap'][1]} | {v['blocked_same_top_token']} of {v['layouts']} |")
    print()
    return result


def validation_table():
    """The 24 picked layouts: full mask against joint reading, blocked mask against separately encoded blocks."""
    result = {}
    print("### Endpoint validation on the 24 picked layouts\n")
    print("| | pair | generated tokens identical | exact match differs | same next token when fed the same continuation |")
    print("|---|---|---|---|---|")
    for model, (tarball, _, job) in MODELS.items():
        blobs = read_tar(tarball, [f"{job}/main/summary.json", f"{job}/diagnostics/rows.jsonl"])
        endpoints = json.loads(blobs[f"{job}/main/summary.json"])["endpoint_comparisons"]
        shared = jsonl(blobs[f"{job}/diagnostics/rows.jsonl"])
        result[model] = {}
        for name, mask, reference in [("full", "masked_full", "b1"), ("blocked", "masked_blocked", "b1_mask")]:
            pairs = [e for e in endpoints if (e["a"], e["b"]) == (mask, reference)]
            steps = [s for r in shared if (r["reference"], r["candidate"]) == (reference, mask) for s in r["steps"]]
            result[model][name] = {"pairs": len(pairs), "identical": sum(e["same_generated_tokens"] for e in pairs),
                                   "em_differs": sum(e["em_difference"] != 0 for e in pairs),
                                   "steps": len(steps), "steps_agree": sum(s["argmax_matches"] for s in steps)}
            v = result[model][name]
            print(f"| {model} | {name} mask vs its reference | {v['identical']} of {v['pairs']} | {v['em_differs']} | "
                  f"{v['steps_agree']} of {v['steps']} ({100 * v['steps_agree'] / v['steps']:.1f}%) |")
    print()
    return result


def toy_table(qs):
    """Toy on Qwen3-8B: blocked and gold, thinking off and forced, look-alike Cities paragraph split or together."""
    rows = jsonl(read_tar(A100, ["wmac_gpu_v3/toy-8b/main/rows.jsonl"])["wmac_gpu_v3/toy-8b/main/rows.jsonl"])
    cell = {}      # (mask, thinking, chain, order, cities, headings) -> (exact match, row)
    for r in rows:
        chain, order, cities, headings = r["id"].split(":")
        mask, thinking = r["condition"].split("_")
        _, em, _ = score.score(r["text"], qs[r["id"]]["answers"])
        cell[(mask, thinking, chain, order, cities, headings)] = (float(em), r)
    chains = sorted({k[2] for k in cell})

    def per_chain(mask, thinking, cities=("split", "together")):
        return [statistics.mean(cell[(mask, thinking, chain, o, c, h)][0]
                                for o in ("AB", "BA") for c in cities for h in ("distinct", "identical")) for chain in chains]

    def minus(a, b):
        return interval([x - y for x, y in zip(a, b)])

    result = {"chains": len(chains), "levels": {}, "contrasts": {}}
    print(f"### Toy, Qwen3-8B, {len(chains)} chains. Exact match, AB and BA\n")
    print("| | thinking off, split | thinking off, together | forced, split | forced, together |\n|---|---|---|---|---|")
    for mask in ("blocked", "gold"):
        result["levels"][mask] = {(t, c): round(100 * statistics.mean(per_chain(mask, t, (c,))), 1)
                                  for t in ("off", "forced") for c in ("split", "together")}
        print(f"| {mask} | " + " | ".join(f"{result['levels'][mask][(t, c)]:.0f}"
                                         for t in ("off", "forced") for c in ("split", "together")) + " |")
    result["contrasts"] = {
        "gold − blocked, thinking off, all layouts": minus(per_chain("gold", "off"), per_chain("blocked", "off")),
        "gold − blocked, thinking off, split": minus(per_chain("gold", "off", ("split",)), per_chain("blocked", "off", ("split",))),
        "gold − blocked, forced, split": minus(per_chain("gold", "forced", ("split",)), per_chain("blocked", "forced", ("split",))),
        "gold − blocked, forced, together": minus(per_chain("gold", "forced", ("together",)), per_chain("blocked", "forced", ("together",))),
        "blocked, split: forced − off": minus(per_chain("blocked", "forced", ("split",)), per_chain("blocked", "off", ("split",))),
        "blocked, together: forced − off": minus(per_chain("blocked", "forced", ("together",)), per_chain("blocked", "off", ("together",))),
    }
    print()
    for name, value in result["contrasts"].items():
        print(f"- {name}: {show(value)}")
    capped = [(em, r) for (mask, thinking, _, _, cities, _), (em, r) in cell.items()
              if (mask, thinking, cities) == ("blocked", "forced", "split")]
    result["cap"] = {"answers": len(capped), "thinking_cut": sum(r["think_cut"] for _, r in capped),
                     "cut_and_right": sum(em for em, r in capped if r["think_cut"]),
                     "finished_and_right": sum(em for em, r in capped if not r["think_cut"])}
    v = result["cap"]
    print(f"- blocked, forced, split: thinking used the whole budget in {v['thinking_cut']} of {v['answers']} answers; "
          f"right among those: {v['cut_and_right']:.0f}; right among the "
          f"{v['answers'] - v['thinking_cut']} that finished: {v['finished_and_right']:.0f}\n")
    return result


TABLES = {"main": main_table, "position": position_table, "together": together_table,
          "checks": lambda qs: checks_table(), "validation": lambda qs: validation_table(), "toy": toy_table}

if __name__ == "__main__":
    chosen = sys.argv[1:] or list(TABLES)
    unknown = [name for name in chosen if name not in TABLES]
    if unknown:
        sys.exit(f"unknown table {unknown}; choose from {list(TABLES)}")
    bundle = questions()
    for name in chosen:
        TABLES[name](bundle)
