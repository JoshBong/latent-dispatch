"""A-side visibility experiment: full recomputation of the unchanged B1 token sequence.

Only B queries' access to A keys changes. The shared prefix, causal attention within each block, and the answer
tail's access to the entire cache are unchanged. This is a masked recompute, not an efficient KV-cache repair.
"""
import hashlib

import torch

from . import prompts

CONDITIONS = ("blocked", "gold", "distractor", "except_gold", "full")


def paragraph_tokens(model, paragraphs):
    """Tokenize the whole existing block once, and map each paragraph to its token indices.

    Tokenizing paragraphs separately changes BPE boundaries. Offsets from the same full-block tokenization preserve
    B1's IDs. A token spanning two paragraphs is ambiguous and is rejected rather than silently leaking content.
    """
    if not paragraphs or any(not isinstance(p, str) or not p for p in paragraphs):
        raise ValueError("each worker needs nonempty string paragraphs")
    text = prompts.block(paragraphs)
    try:
        encoded = model.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError) as e:
        raise ValueError("paragraph masking requires a fast tokenizer with offset mappings") from e
    ids, offsets = encoded.input_ids, encoded.offset_mapping
    if ids != model.ids(text) or len(ids) != len(offsets):
        raise ValueError("offset tokenization does not match B1 block tokenization")
    ranges, cursor = [], 0
    for p in paragraphs:
        ranges.append((cursor, cursor + len(p)))
        cursor += len(p) + 2  # prompts.block joins paragraphs with two newlines
    groups = [[] for _ in paragraphs]
    for i, (start, end) in enumerate(offsets):
        hits = [j for j, (lo, hi) in enumerate(ranges) if start < hi and end > lo]
        if len(hits) > 1:
            raise ValueError(f"token {i} crosses a paragraph boundary")
        if hits:
            groups[hits[0]].append(i)
    if any(not group for group in groups):
        raise ValueError("a paragraph has no identifiable tokens")
    return ids, groups


def layout(model, q, condition, distractor_seed=0):
    """Return B1 ids, additive 4D causal mask, block sizes, and the chosen A paragraph indices.

    'gold' exposes all A gold paragraphs, 'distractor' exposes one deterministically chosen A distractor, and
    'except_gold' exposes all of A except its gold paragraphs. 'full' also exposes separators; 'blocked' exposes
    no A tokens. A control with no A gold is necessarily identical to blocked (gold) or full (except_gold).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, not {condition!r}")
    if len(q["workers"]) != 2 or any(not w for w in q["workers"]):
        raise ValueError("the A-side masking experiment requires exactly two nonempty workers")
    gold = set()
    for ref in q["gold"]:
        w, i = ref["worker"], ref["index"]
        if not isinstance(w, int) or not isinstance(i, int) or w not in (0, 1) or not 0 <= i < len(q["workers"][w]):
            raise ValueError(f"invalid gold paragraph reference: {ref!r}")
        if w == 0:
            gold.add(i)
    s = model.ids(prompts.prefix(model.fmt, q["question"]))
    a, spans = paragraph_tokens(model, q["workers"][0])
    b = model.ids(prompts.block(q["workers"][1]))
    distractors = [i for i in range(len(spans)) if i not in gold]
    if condition == "distractor" and not distractors:
        raise ValueError("worker A has no distractor paragraph for the distractor control")
    if condition == "gold":
        chosen = sorted(gold)
    elif condition == "distractor":
        # Stable across processes and Python versions, and independent of arm execution order.
        digest = hashlib.sha256(f"{distractor_seed}:{q['id']}".encode()).digest()
        chosen = [distractors[int.from_bytes(digest, "big") % len(distractors)]]
    elif condition in ("except_gold", "full"):
        chosen = distractors if condition == "except_gold" else list(range(len(spans)))
    else:
        chosen = []

    visible = torch.zeros(len(a), dtype=torch.bool, device=model.device)
    if condition in ("full", "except_gold"):
        visible[:] = True  # include separators, so except_gold is the literal complement of gold tokens
    for i in (sorted(gold) if condition == "except_gold" else chosen):
        visible[spans[i]] = condition != "except_gold"
    ids = s + a + b
    n, b_start = len(ids), len(s) + len(a)
    allow = torch.ones((n, n), dtype=torch.bool, device=model.device).tril()
    allow[b_start:, len(s):b_start] = visible
    dtype = next(model.hf.parameters()).dtype
    mask = torch.zeros((n, n), dtype=dtype, device=model.device)
    mask.masked_fill_(~allow, torch.finfo(dtype).min)
    meta = {"condition": condition, "a_paragraphs": chosen, "a_gold_paragraphs": sorted(gold),
            "a_visible_tokens": int(visible.sum().item()), "a_tokens": len(a), "distractor_seed": distractor_seed}
    return ids, mask[None, None], [len(a), len(b)], meta


def context(model, q, condition, distractor_seed=0):
    ids, mask, sizes, meta = layout(model, q, condition, distractor_seed)
    out = model.forward(ids, attention_mask=mask)
    return out.past_key_values, len(ids), sizes, meta
