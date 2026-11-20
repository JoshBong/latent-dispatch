"""HotpotQA bridge questions under the fresh-source mask protocol: the MuSiQue study's method on a second dataset.

Preparation uses tokenizers and never loads model weights. It freezes the questions before any answer exists.

HotpotQA lacks two annotations that the MuSiQue protocol uses. What stands in for each:

  hop order      HotpotQA does not say which supporting paragraph comes first. The supporting paragraph that holds
                 the answer is hop 2 and the other is hop 1. A question whose answer appears in both supporting
                 paragraphs, or in neither, is dropped.
  bridge entity  HotpotQA names no bridge. The bridge is a supporting paragraph's title (a trailing parenthesis
                 removed) that the other supporting paragraph mentions. A question with no such title is dropped,
                 which is the MuSiQue rule "bridge explicit in both supports". As on MuSiQue, the equal-token window
                 is never drawn from a paragraph that mentions the answer or the bridge.

The rest is the MuSiQue protocol as it is (wmac_plan): the grounding flags, one supporting paragraph per worker and
last in its block, AB (hop 1 in A) and BA (hop 2 in A) holding the same distractors, a fixed order that uses no model
output, sources of archived runs left out, and feasibility under all three tokenizers. HotpotQA ships 10 paragraphs
per question, so each worker holds 5 (MuSiQue: 10).
"""
import copy
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from . import prompts
from .paper_plan import ROOT, digest, file_sha, frozen_json
from .score import contains_answer, normalize
from .wmac_plan import grounding_flags, support_key, support_last, supports

PROTOCOL = "wmac_hotpot_v1"
CONFIG = ROOT / "experiments" / "wmac_hotpot_v1.json"
PREPARED = ROOT / "experiments" / "wmac_hotpot_v1" / "prepared.json"
CONDITIONS = ("blocked", "gold", "full", "matched_irrelevant", "b1", "k0")      # the main run's six, thinking off
PROBES = ("closed_book", "keep_hop0", "keep_hop1")                             # AB layouts only, as on MuSiQue
HEAVY = ("a_visible_indices", "a_gold_indices")                                # recomputed at run time, not frozen


# ---- hop order and bridge ------------------------------------------------------------------------------------------

def title(paragraph):
    """'Title (disambiguation): text' -> 'Title'. A title that itself holds ': ' is cut at its first one."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", paragraph.split(": ", 1)[0]).strip()


def chain(q):
    """Give one HotpotQA question the hop order and bridge the MuSiQue protocol needs.
    -> (a copy of q with hop labels and bridges, flags). A flagged question is not used."""
    q = copy.deepcopy(q)
    gold = sorted(q["gold"], key=lambda g: (g["worker"], g["index"]))
    if len(gold) != 2:
        return q, ["invalid_support_pair"]
    if any(normalize(a) in ("yes", "no") for a in q["answers"]):
        return q, ["yes_no_answer"]
    texts = [q["workers"][g["worker"]][g["index"]] for g in gold]
    holds = [contains_answer(text, q["answers"]) for text in texts]
    if all(holds):
        return q, ["answer_in_both_supports"]
    if not any(holds):
        return q, ["answer_absent_from_supports"]
    second = holds.index(True)                                  # the supporting paragraph with the answer is hop 2
    gold[second]["hop"], gold[1 - second]["hop"] = 1, 0
    q["gold"] = sorted(gold, key=lambda g: g["hop"])
    bridges = []
    for own, other in ((texts[second], texts[1 - second]), (texts[1 - second], texts[second])):
        name = title(own)
        if normalize(name) and contains_answer(other, [name]) and name not in bridges:
            bridges.append(name)
    q["bridges"] = bridges
    q["hop_rule"], q["bridge_rule"] = "answer paragraph is hop 2", "supporting title mentioned by the other support"
    return q, []


# ---- layouts ---------------------------------------------------------------------------------------------------------

def layouts(q):
    """AB (hop 1 in A, hop 2 in B) and BA (the reverse) of one source, each supporting paragraph last in its block.
    Both keep each worker's distractors in their order, so the two differ only in where the supports sit (MuSiQue's
    AB and BA have the same relation). The input needs hop labels (chain) and one supporting paragraph per worker."""
    hops = sorted(q["gold"], key=lambda g: g["hop"])
    if [g["hop"] for g in hops] != [0, 1] or sorted(g["worker"] for g in hops) != [0, 1] or len(q["workers"]) != 2:
        raise ValueError(f"{q['id']}: need two workers, one supporting paragraph in each, with hop labels 0 and 1")
    first, second = (q["workers"][g["worker"]][g["index"]] for g in hops)
    rest = [[p for i, p in enumerate(worker) if not any(g["worker"] == w and g["index"] == i for g in hops)]
            for w, worker in enumerate(q["workers"])]
    made = []
    for placement, in_a, in_b in (("AB", first, second), ("BA", second, first)):
        item = copy.deepcopy(q)
        item["workers"] = [rest[0] + [in_a], rest[1] + [in_b]]
        hop_in_a = 0 if placement == "AB" else 1
        item["gold"] = sorted([{"hop": hop_in_a, "worker": 0, "index": len(rest[0])},
                               {"hop": 1 - hop_in_a, "worker": 1, "index": len(rest[1])}], key=lambda g: g["hop"])
        item.update(id=f"{q['source_id']}:{placement}:s0", placement=placement, crossings=1)
        item = support_last(item)                               # already last: this only stamps the layout protocol
        if item["workers"] != [rest[0] + [in_a], rest[1] + [in_b]]:
            raise ValueError(f"{q['id']}: support_last moved a paragraph")
        made.append(item)
    return made


# ---- selection -------------------------------------------------------------------------------------------------------

def historical_sources(folder, known):
    """Sources among `known` that already have rows in an archived run. Scans the tarballs at the top of the folder,
    the same ones the MuSiQue planner scans. -> (sources, {tarball name: sha256})"""
    used, archives = set(), {}
    for path in sorted(Path(folder).glob("*.tar.gz")):
        archives[path.name] = file_sha(path)
        with tarfile.open(path) as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".jsonl") or Path(member.name).name.startswith("._"):
                    continue
                for line in tar.extractfile(member):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    source = str(row.get("id", "")).split(":")[0] if isinstance(row, dict) else ""
                    if source in known:
                        used.add(source)
    return used, archives


def select_sources(questions, excluded, n, seed, ineligible=None):
    """wmac_plan.select_sources for HotpotQA bridge questions: walk the sources in a fixed order that uses no model
    output, keep the first n that pass every check, and say why each of the others was dropped.
    `ineligible(layout)` returns a flag for a layout that cannot be run, or None.
    -> (2n layouts: AB then BA of each kept source, audit rows in the order walked)"""
    groups = {q["source_id"]: q for q in questions
              if q["dataset"] == "hotpotqa" and q["qtype"] == "bridge" and q["seed"] == 0 and q["placement"] == "AB"}
    used_supports = {support_key(p) for s in excluded if s in groups for p in supports(groups[s])}
    used_questions = {normalize(groups[s]["question"]) for s in excluded if s in groups}
    selected, audit = [], []
    for source in sorted(groups, key=lambda s: digest([seed, PROTOCOL, s])):
        q, flags = chain(groups[source])
        flags = flags or grounding_flags(q)
        if source in excluded:
            flags.append("historical_source")
        keys = {support_key(p) for p in supports(q)}
        if keys & used_supports:
            flags.append("shared_support_text")
        if normalize(q["question"]) in used_questions:
            flags.append("duplicate_question")
        prepared = []
        if not flags:
            prepared = layouts(q)
            flags = sorted({flag for item in prepared for flag in [ineligible(item) if ineligible else None] if flag})
        audit.append(dict(source_id=source, flags=sorted(set(flags))))
        if flags:
            continue
        selected.extend(prepared)
        used_supports.update(keys)
        used_questions.add(normalize(q["question"]))
        if len(selected) == 2 * n:
            break
    if len(selected) != 2 * n:
        raise ValueError(f"only {len(selected) // 2} eligible independent sources; need {n}")
    return selected, audit


def feasibility(adapters, max_tokens):
    """A layout is usable only if every model's tokenizer can build the equal-token window and the context fits."""
    from .wmac_experiments import matched_indices

    def ineligible(q):
        for adapter in adapters.values():
            try:
                matched_indices(adapter, q)
            except ValueError as error:
                if "insufficient earlier irrelevant tokens" in str(error):
                    return "input_token_control_ineligible"
                return "input_tokenization_ineligible"          # e.g. a token that spans two paragraphs
            length = len(adapter.ids(prompts.prefix(adapter.fmt, q["question"]))) + sum(
                len(adapter.ids(prompts.block(worker))) for worker in q["workers"])
            if length > max_tokens:
                return "context_too_long"
        return None
    return ineligible


# ---- the frozen question set -----------------------------------------------------------------------------------------

def adapter(tokenizer, fmt):
    """Enough of a Model for the layout and tokenizer checks: no weights."""
    return SimpleNamespace(tok=tokenizer, fmt=prompts.FORMATS[fmt],
                           ids=lambda text, tok=tokenizer: tok(text, add_special_tokens=False).input_ids)


def load_adapters(cfg, local_files_only=False):
    """The three tokenizers at the revisions the config freezes; each must be the tokenizer of the MuSiQue run."""
    from transformers import AutoTokenizer
    adapters = {}
    for key, model in cfg["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(model["name"], revision=model["revision"],
                                                  local_files_only=local_files_only)
        found = digest(tokenizer.backend_tokenizer.to_str())
        if found != model["tokenizer_sha256"]:
            raise ValueError(f"{model['name']}: tokenizer {found[:12]} is not the MuSiQue run's {model['tokenizer_sha256'][:12]}")
        adapters[key] = adapter(tokenizer, model["format"])
    return adapters


def read_benchmark(folder):
    """The seed-0 AB layout of every question in benchmark/hotpotqa.jsonl (one per source question; the file holds
    three layouts under three seeds), after checking the file against SHA256SUMS. -> (questions, {file name: sha256})"""
    expected = dict(reversed(line.split()) for line in (ROOT / "benchmark" / "SHA256SUMS").read_text().splitlines())
    path = Path(folder) / "hotpotqa.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path}: first build it with python -m latent_dispatch.data hotpotqa")
    sha = file_sha(path)
    if sha != expected["hotpotqa.jsonl"]:
        raise ValueError(f"benchmark checksum mismatch: {path}")
    questions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            q = json.loads(line)
            if q["seed"] == 0 and q["placement"] == "AB":
                questions.append(q)
    if len({q["source_id"] for q in questions}) != len(questions):
        raise ValueError("a source question has more than one seed-0 AB layout")
    return questions, {"hotpotqa.jsonl": sha}


def build_bundle(cfg, questions, excluded, adapters, extra=None):
    """Select the questions and gather everything that is frozen with them. No file is read or written here."""
    from .wmac_experiments import check_prompt_template, preflight
    if cfg["protocol"] != PROTOCOL:
        raise ValueError(f"not a {PROTOCOL} config")
    fresh, audit = select_sources(questions, excluded, cfg["hotpot_sources"], cfg["seed"],
                                  feasibility(adapters, cfg["max_context_tokens"]))
    chosen = {q["id"]: q for q in fresh}
    if len(chosen) != len(fresh):
        raise ValueError("duplicate layout ids")
    checked = {}
    for key, stand_in in adapters.items():
        lengths = preflight(stand_in, chosen, list(chosen), cfg["max_context_tokens"])
        checked[key] = {id: {k: v for k, v in row.items() if k not in HEAVY} for id, row in lengths.items()}
    return dict(config=cfg, **(extra or {}), excluded_historical_sources=sorted(excluded), selection_audit=audit,
                exclusion_counts=dict(Counter(flag for row in audit for flag in row["flags"])),
                questions=chosen, question_sha256={id: digest(q) for id, q in chosen.items()},
                hotpot_ids=[q["id"] for q in fresh], preflight=checked,
                prompt_template_checks={key: check_prompt_template(a, fresh[0]) for key, a in adapters.items()},
                tokenizer_sha256={key: digest(a.tok.backend_tokenizer.to_str()) for key, a in adapters.items()})


def prepare(benchmark_dir=ROOT / "benchmark", destination=PREPARED, config_path=CONFIG, adapters=None,
            archive=ROOT / "archive"):
    """Select and freeze the questions. Run once, before inference; an existing file is loaded, never rebuilt."""
    destination = Path(destination)
    if destination.exists():
        return load(destination)
    cfg = json.loads(Path(config_path).read_text())
    adapters = adapters or load_adapters(cfg)
    questions, hashes = read_benchmark(benchmark_dir)
    excluded, archives = historical_sources(archive, {q["source_id"] for q in questions})
    bundle = build_bundle(cfg, questions, excluded, adapters,
                          dict(config_sha256=file_sha(config_path), benchmark_sha256=hashes, archive_sha256=archives))
    frozen_json(destination, bundle)
    return bundle, bundle["questions"]


def load(path=PREPARED):
    """The frozen question set, with its hashes and its layout rules checked. -> (bundle, {layout id: question})"""
    bundle = json.loads(Path(path).read_text())
    if bundle["config"]["protocol"] != PROTOCOL:
        raise ValueError(f"{path}: not a {PROTOCOL} question set")
    questions, ids = bundle["questions"], bundle["hotpot_ids"]
    if {id: digest(q) for id, q in questions.items()} != bundle["question_sha256"]:
        raise ValueError("prepared question hash mismatch")
    if len(ids) != len(set(ids)) or set(ids) != set(questions) or len(ids) != 2 * bundle["config"]["hotpot_sources"]:
        raise ValueError("prepared ids are inconsistent")
    for ab, ba in zip(ids[::2], ids[1::2]):
        first, second = questions[ab], questions[ba]
        if (first["source_id"] != second["source_id"] or (first["placement"], second["placement"]) != ("AB", "BA")
                or sorted(p for w in first["workers"] for p in w) != sorted(p for w in second["workers"] for p in w)):
            raise ValueError(f"{ab}: a source needs its AB layout then its BA layout, over the same paragraphs")
        for q in (first, second):
            if grounding_flags(q) or support_last(q)["workers"] != q["workers"] or q.get("layout_protocol") != "support-last-v1":
                raise ValueError(f"{q['id']}: the layout breaks the protocol")
    return bundle, questions
