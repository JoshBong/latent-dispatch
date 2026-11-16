"""GPU operations for the three frozen paper ablations; imported only on execution."""
from pathlib import Path

import numpy as np
import torch

from . import arms, kv, masking, prompts
from .paper_plan import digest


def generation_fields(g):
    return {'text':g.text, 'gen_tokens':g.tokens, 'think_tokens':g.think_tokens,
            'answer_tokens':g.answer_tokens, 'cut':g.cut, 'think_cut':g.think_cut,
            'think_eos':g.think_eos}


def context_tokens(model, q):
    prefix = model.ids(prompts.prefix(model.fmt, q['question']))
    blocks = [model.ids(prompts.block(w)) for w in q['workers']]
    ids = prefix + [token for block in blocks for token in block]
    return ids, [len(b) for b in blocks], len(prefix)


def endpoint(model, q, variant, answer_tokens, output_dir):
    """One context build and one generation. Diagnostics do not alter the answer cache."""
    expected_ids, sizes, prefix_size = context_tokens(model, q)
    if variant == 'b1':
        cache, pos, actual_sizes = arms.b1_context(model, q)
    elif variant == 'b1_mask':
        cache, pos, actual_sizes = arms.k0_context(model, q, positions='true')
    else:
        condition = {'masked_full':'full', 'masked_blocked':'blocked'}[variant]
        ids, mask, actual_sizes, _ = masking.layout(model, q, condition)
        if ids != expected_ids:
            raise ValueError('masked and baseline context tokens differ')
        out = model.forward(ids, attention_mask=mask)
        cache, pos = out.past_key_values, len(ids)
        del out, mask
    if actual_sizes != sizes or pos != len(expected_ids) or kv.length(cache) != pos:
        raise ValueError('context/cache length or stitched positions differ')
    tail = model.ids(prompts.tail(model.fmt, 'off'))
    out = model.forward(tail, kv.copy(cache), pos)
    first = out.logits[0,-1].float().cpu().numpy().copy()
    del out
    if not np.isfinite(first).all():
        raise ValueError('non-finite endpoint logits')
    name = 'logits/' + digest([q['id'],variant])[:24] + '.npz'
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, first_logits=first)
    temporary.replace(path)
    g = model.greedy(tail, cache, pos, max_new=answer_tokens, trace_topk=5)
    return {**generation_fields(g), 'token_ids':g.token_ids, 'trace':g.trace,
            'first_logits_file':name, 'context_tokens_sha256':digest(expected_ids),
            'answer_tail_sha256':digest(tail), 'position_ids_sha256':digest(list(range(pos))),
            'context_length':pos, 'prefix_tokens':prefix_size, 'worker_tokens':sizes}


def thinking(model, q, condition, mode, think_budget, answer_tokens):
    cache, pos, sizes, meta = masking.context(model, q, condition)
    tail = model.ids(prompts.tail(model.fmt, mode))
    g = model.greedy(tail, cache, pos, max_new=answer_tokens,
                     think_budget=think_budget if mode == 'forced' else None)
    return {**generation_fields(g), 'worker_tokens':sizes, 'mask':meta}


def boxed_parts(model, answer):
    """Partition the canonical boxed tokenization, refusing a boundary-changing prefix.

This preserves the original likelihood target exactly while separating probability
of the format prefix, the answer tokens conditional on that prefix, and the close.
"""
    prefix = model.ids('\\boxed{')
    close = model.ids('}')
    full = model.ids('\\boxed{' + answer + '}')
    if not prefix or not close or full[:len(prefix)] != prefix or full[-len(close):] != close:
        raise ValueError('boxed prefix/closing boundary changes tokenization; cannot isolate answer tokens')
    answer_ids = full[len(prefix):-len(close)]
    if not answer_ids or model.decode(answer_ids) != answer:
        raise ValueError('answer tokens do not decode to the exact candidate')
    return full, len(prefix), len(answer_ids)


def likelihood(model, q, condition, candidate_spec):
    cache, pos, sizes, meta = masking.context(model, q, condition, distractor_seed=0)
    if condition == 'distractor':
        chosen = meta['a_paragraphs']
        if chosen != [candidate_spec['a_distractor_index']] or chosen[0] >= min(meta['a_gold_paragraphs']):
            raise ValueError('the control is not the frozen pre-gold distractor')
    tail = model.ids(prompts.tail(model.fmt, 'off'))
    scores = []
    for answer in candidate_spec['answers']:
        target, prefix_n, answer_n = boxed_parts(model, answer)
        out = model.forward(tail + target, kv.copy(cache), pos)
        logits = out.logits[0,len(tail)-1:len(tail)+len(target)-1].float()
        ids = torch.tensor(target, device=logits.device).unsqueeze(1)
        logp = torch.log_softmax(logits, -1).gather(1, ids).squeeze(1)
        if not torch.isfinite(logp).all():
            raise ValueError('non-finite candidate likelihood')
        values = logp.cpu().tolist()
        conditional = sum(values[prefix_n:prefix_n+answer_n])
        scores.append({'answer':answer, 'target_token_ids':target, 'token_logprobs':values,
                       'answer_token_count':answer_n, 'prefix_token_count':prefix_n,
                       'boxed_logprob':sum(values), 'prefix_logprob':sum(values[:prefix_n]),
                       'answer_logprob':conditional, 'answer_mean_logprob':conditional/answer_n,
                       'closing_logprob':sum(values[prefix_n+answer_n:])})
        del out, logits, logp
    gold, wrong = scores[0], scores[1:]
    return {'worker_tokens':sizes, 'mask':meta, 'candidates':scores,
            'gold_boxed_logprob':gold['boxed_logprob'], 'gold_prefix_logprob':gold['prefix_logprob'],
            'gold_answer_logprob':gold['answer_logprob'], 'gold_answer_mean_logprob':gold['answer_mean_logprob'],
            'gold_closing_logprob':gold['closing_logprob'],
            'gold_minus_best_wrong':gold['answer_logprob']-max(r['answer_logprob'] for r in wrong),
            'gold_minus_best_wrong_mean':gold['answer_mean_logprob']-max(r['answer_mean_logprob'] for r in wrong),
            'archived':candidate_spec['archived'][condition],
            'boxed_change_from_archive':gold['boxed_logprob']-candidate_spec['archived'][condition]['gold_logprob']}


def compare_endpoints(a, b, output_dir):
    """Compare same-environment paths; numerical differences are diagnostics, not a pass/fail claim."""
    for key in ['context_tokens_sha256','answer_tail_sha256','position_ids_sha256','worker_tokens']:
        if a[key] != b[key]:
            raise ValueError(f'endpoint structural mismatch: {key}')
    def read(row):
        path = (Path(output_dir) / row['first_logits_file']).resolve()
        if not path.is_relative_to(Path(output_dir).resolve()):
            raise ValueError('logit artifact outside run directory')
        with np.load(path, allow_pickle=False) as f:
            return f['first_logits'].astype(np.float64)
    x, y = read(a), read(b)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid endpoint logit vectors')
    lx, ly = x-np.logaddexp.reduce(x), y-np.logaddexp.reduce(y)
    ta, tb = a['token_ids'], b['token_ids']
    difference = next((i for i,(u,v) in enumerate(zip(ta,tb)) if u != v), None)
    if difference is None and len(ta) != len(tb):
        difference = min(len(ta),len(tb))
    traces = []
    if difference is not None:
        traces = [next((t for t in row['trace'] if t['token_index']==difference), None) for row in (a,b)]
    return {'same_generated_tokens':ta==tb, 'first_divergence_index':difference,
            'traces_at_first_divergence':traces, 'first_argmax_matches':int(x.argmax())==int(y.argmax()),
            'first_argmax_ids':[int(x.argmax()),int(y.argmax())],
            'first_top2_margins':[float(np.sort(z)[-1]-np.sort(z)[-2]) for z in (x,y)],
            'first_logits_max_abs_delta':float(np.max(np.abs(x-y))),
            'first_logits_mean_abs_delta':float(np.mean(np.abs(x-y))),
            'first_distribution_kl_a_to_b':float(np.sum(np.exp(lx)*(lx-ly)))}
