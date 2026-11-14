"""Every arm, as a plain function:  arm(model, q, **options) -> dict with the raw output text and token counts.

`q` is one benchmark question (a dict from data.load). Every arm builds the orchestrator's context in the same prompt
layout (prompts.py) and answers through the same `answer` step, so two arms differ only in how that context is built.
Each *_context function returns (cache, pos, worker_tokens): the orchestrator's KV cache, the RoPE position of the
next token, and the payload size per worker. The tests compare these contexts directly.

Options every arm takes (passed through to `answer`):
    thinking = "off" | "forced"    forced prefills "<think>\\n" so the model must reason first (Qwen3 only)
    think_budget = 1024            thinking tokens allowed before "</think>" is forced (H4's variable)
    max_answer_tokens = 256        a cap that should never bind: every row records `cut`, and the audit reports how
                                   often it did (analysis.audit). Raise it if the smoke shows cuts.

The latent-step code (k1) follows JoshBong/LatentMAS models.py (fork of Gen-Verse/LatentMAS, Apache-2.0):
https://github.com/JoshBong/LatentMAS/blob/main/models.py. The scan and text-report arms port Paper A's
scan_judge.py and text_report_run.py.
"""
import torch

from . import kv, masking, prompts


def answer(model, cache, pos, thinking="off", think_budget=1024, max_answer_tokens=256, pre=""):
    """Feed the answer tail (format line + assistant header + thinking prefix) on top of `cache` and decode."""
    if thinking not in ("off", "forced"):
        raise ValueError(f"thinking must be 'off' or 'forced', not {thinking!r}")
    tail = model.ids(prompts.tail(model.fmt, thinking, pre))
    g = model.greedy(tail, cache, pos, max_new=max_answer_tokens,
                     think_budget=think_budget if thinking == "forced" else None)
    return {"text": g.text, "think_tokens": g.think_tokens, "gen_tokens": g.tokens, "cut": g.cut,
            "think_cut": g.think_cut}


# ---------------------------------------------------------------------------------------------------- B0, B1

def b0(model, q, **answer_options):
    """Closed book: the question, no paragraphs."""
    ids = model.ids(prompts.prefix(model.fmt, q["question"]))
    cache, _ = model.prefill(ids)
    return answer(model, cache, len(ids), **answer_options)


def visible_paragraphs(q, keep_gold_hops=None):
    """Each worker's paragraphs, minus the gold paragraphs of every hop not in keep_gold_hops (distractors stay)."""
    if keep_gold_hops is None:
        return q["workers"]
    drop = {(g["worker"], g["index"]) for g in q["gold"] if g["hop"] not in keep_gold_hops}
    return [[p for j, p in enumerate(w) if (wi, j) not in drop] for wi, w in enumerate(q["workers"])]


def b1_context(model, q, keep_gold_hops=None):
    blocks = [model.ids(prompts.block(w)) for w in visible_paragraphs(q, keep_gold_hops) if w]
    ids = model.ids(prompts.prefix(model.fmt, q["question"])) + [t for b in blocks for t in b]
    cache, _ = model.prefill(ids)
    return cache, len(ids), [len(b) for b in blocks]


def b1(model, q, keep_gold_hops=None, **answer_options):
    """All paragraphs as text in one context, in the same order K0 stitches them. The ceiling.
    keep_gold_hops = the Part 0 probe: e.g. [0] keeps only the hop-1 gold paragraph (plus every distractor)."""
    cache, pos, sizes = b1_context(model, q, keep_gold_hops)
    return {**answer(model, cache, pos, **answer_options), "worker_tokens": sizes}


# ---------------------------------------------------------------------------------------------------- K0, B1-mask

def encode_paragraphs(model, paragraphs, cache, start):
    """One worker's block on top of its copy of S, at RoPE positions starting at `start`. -> (cache, n positions)."""
    ids = model.ids(prompts.block(paragraphs))
    cache, _ = model.prefill(ids, cache, start)
    return cache, len(ids)


def k0_context(model, q, positions="local", merge="seq", visible=None, encode=encode_paragraphs):
    """Each worker encodes its paragraphs on top of a copy of the shared prefix S; the orchestrator gets S + every
    worker's block, stitched.

    positions="local": every worker encodes right after S, then its keys are shifted into place in the stitched
                       sequence (merge="seq"), or left overlapping at the same positions (merge="shared").
    positions="true":  every worker encodes at its final position. Equal to B1 with a block-diagonal mask (blocks
                       can't see each other): that's B1-mask.
    visible:           which workers the orchestrator gets (None = all). [0] alone = the select-one check.
    encode:            how one worker's block is built (K1 swaps in paragraphs + latent steps).
    """
    if positions not in ("local", "true") or merge not in ("seq", "shared"):
        raise ValueError(f"positions must be 'local'/'true' and merge 'seq'/'shared', not {positions!r}/{merge!r}")
    if positions == "true" and merge == "shared":
        raise ValueError("merge='shared' only makes sense with positions='local'")
    s_ids = model.ids(prompts.prefix(model.fmt, q["question"]))
    s_cache, _ = model.prefill(s_ids)
    n_s = len(s_ids)
    blocks, sizes, offset = [], [], 0
    for w in (visible if visible is not None else range(len(q["workers"]))):
        start = n_s + offset if positions == "true" else n_s
        cache, n = encode(model, q["workers"][w], kv.copy(s_cache), start)
        block = kv.suffix(cache, n_s)
        if positions == "local" and merge == "seq" and offset:
            block = kv.shift(block, offset, model.inv_freq)
        blocks.append(block)
        sizes.append(n)
        offset += n
    # seq / true: blocks fill n_s .. n_s + offset.  shared: every block starts at n_s, so the tail follows the longest.
    pos = n_s + (max(sizes, default=0) if merge == "shared" else offset)
    return kv.concat([s_cache] + blocks), pos, sizes


def k0(model, q, positions="local", merge="seq", visible=None, **answer_options):
    """Paragraph KV from each worker, stitched (Paper A's `merge`, LATENT0=1)."""
    cache, pos, sizes = k0_context(model, q, positions, merge, visible)
    return {**answer(model, cache, pos, **answer_options), "worker_tokens": sizes}


def b1_mask(model, q, **answer_options):
    """B1 with a block-diagonal attention mask and true positions = K0 with positions="true"."""
    return k0(model, q, positions="true", **answer_options)


def masked_recompute(model, q, condition, distractor_seed=0, score_gold=True, **answer_options):
    """A-side visibility experiment, with the same B1 tokens and true positions under every condition.

    By default also teacher-force the first gold answer, formatted as \\boxed{answer}, immediately after the
    thinking-off answer tail. This is a conditional sequence log-probability, not EM or a sum over aliases.
    """
    thinking = answer_options.get("thinking", "off")
    if score_gold and (thinking != "off" or answer_options.get("pre", "")):
        raise ValueError("gold likelihood requires thinking='off' and no extra answer preamble")
    cache, pos, sizes, meta = masking.context(model, q, condition, distractor_seed)
    likelihood = {}
    if score_gold:
        gold = next((a for a in q["answers"] if a and a.strip()), None)
        if gold is None:
            raise ValueError("instance has no gold answer")
        target = model.ids("\\boxed{" + gold + "}")
        tail = model.ids(prompts.tail(model.fmt, "off"))
        if not target or not tail:
            raise ValueError("empty tokenized answer or answer prompt")
        out = model.forward(tail + target, kv.copy(cache), pos)
        logits = out.logits[0, len(tail) - 1:len(tail) + len(target) - 1].float()
        indices = torch.tensor(target, device=logits.device).unsqueeze(1)
        logp = torch.log_softmax(logits, dim=-1).gather(1, indices).sum().item()
        likelihood = {"gold_answer": gold, "gold_logprob": logp, "gold_token_count": len(target),
                      "gold_mean_logprob": logp / len(target)}
    return {**answer(model, cache, pos, **answer_options), "worker_tokens": sizes, "mask": meta, **likelihood}


# ---------------------------------------------------------------------------------------------------- K1

def k1_context(model, q, m, realign=False, visible=None):
    """K0, but each worker's block is: its paragraphs, an instruction, its own assistant turn holding m latent
    thoughts (LatentMAS), and the close of that turn. With m = 0 the wrapper is still there, so the H2 contrast is
    K1(m) vs K1(0), not vs bare K0. realign=True applies LatentMAS's W_a (that's K2; on Qwen3-4B W_a ≈ I)."""
    close = model.ids(prompts.worker_close(model.fmt))

    def encode(model, paragraphs, cache, start):
        head = model.ids(prompts.block(paragraphs) + prompts.worker_open(model.fmt))
        cache, h = model.prefill(head, cache, start)
        cache, _ = model.latent_steps(cache, h, m, start + len(head), realign)
        cache, _ = model.prefill(close, cache, start + len(head) + m)
        return cache, len(head) + m + len(close)

    return k0_context(model, q, visible=visible, encode=encode)


def k1(model, q, m, realign=False, visible=None, **answer_options):
    cache, pos, sizes = k1_context(model, q, m, realign, visible)
    return {**answer(model, cache, pos, **answer_options), "worker_tokens": sizes}


# ---------------------------------------------------------------------------------------------------- T1

def t1(model, q, report_tokens=1024, **answer_options):
    """Each worker reads only its own paragraphs and writes a text report (at most report_tokens); the orchestrator
    reads the reports as text, in the same layout."""
    reports, sizes, truncated = [], [], []
    for w in q["workers"]:
        g = model.greedy(model.ids(prompts.report_prompt(model.fmt, q["question"], w)), max_new=report_tokens)
        reports.append(g.text.strip())
        sizes.append(g.tokens)
        truncated.append(g.cut)
    ids = model.ids(prompts.prefix(model.fmt, q["question"]))
    ids += [t for j, r in enumerate(reports) for t in model.ids(prompts.report_block(j, r))]
    cache, _ = model.prefill(ids)
    return {**answer(model, cache, len(ids), **answer_options), "worker_tokens": sizes, "reports": reports,
            "truncated": truncated}


# ---------------------------------------------------------------------------------------------------- scan

def _note(text, note_style, note_chars):
    text = text.strip()
    if note_style == "line":
        return next((ln.strip()[:300] for ln in text.splitlines() if ln.strip()), "")
    return text[:note_chars] if note_chars > 0 else text


def scan(model, q, rounds=1, note_style="line", note_tokens=None, note_chars=0, **answer_options):
    """The orchestrator reads the saved worker caches one at a time and carries text notes forward; it answers on
    the last block. Each worker block is encoded once, right after S, and reused on every pass. Each step sees
    [S | one worker block] + the notes so far, so information crosses workers only through the notes.
    rounds = R reads the blocks R times in stitching order.
    note_style "line": one line (48 tokens, first line, at most 300 characters). Being one line is what this channel
    is, so a longer note is not a cut. "scratch": a four-line scratchpad with room to finish (1024 tokens, no
    character cap unless note_chars > 0). `notes_cut` records every note that hit its token cap before finishing."""
    if rounds < 1 or note_style not in ("line", "scratch"):
        raise ValueError(f"need rounds >= 1 and note_style 'line' or 'scratch', not {rounds!r}/{note_style!r}")
    budget = note_tokens or (48 if note_style == "line" else 1024)
    s_ids = model.ids(prompts.prefix(model.fmt, q["question"]))
    s_cache, _ = model.prefill(s_ids)
    n_s = len(s_ids)
    blocks = []
    for w in q["workers"]:
        cache, n = encode_paragraphs(model, w, kv.copy(s_cache), n_s)
        blocks.append((kv.suffix(cache, n_s), n))
    notes, notes_cut = [], []
    order = list(range(len(blocks))) * rounds
    for step, w in enumerate(order):
        ctx = kv.concat([s_cache, blocks[w][0]])
        pos = n_s + blocks[w][1]
        if step == len(order) - 1:
            out = answer(model, ctx, pos, pre=prompts.notes_block(notes) + "\n\n", **answer_options)
            return {**out, "worker_tokens": [n for _, n in blocks], "notes": notes, "notes_cut": notes_cut,
                    "passes": len(order)}
        g = model.greedy(model.ids(prompts.note_tail(model.fmt, notes, note_style)), ctx, pos, max_new=budget)
        notes.append(_note(g.text, note_style, note_chars))
        # a line note is cut only if the cap hit before its first line ended
        notes_cut.append(g.cut and (note_style == "scratch" or "\n" not in g.text.strip()))
