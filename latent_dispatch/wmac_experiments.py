"""Inference and numerical diagnostics, called only by an explicit runner command."""
from pathlib import Path

import numpy as np
import torch

from . import arms, kv, masking, paper_experiments, prompts
from .paper_plan import digest
from .score import contains_answer


def matched_indices(model,q):
    """Exact gold-size window over earlier non-support token indices.

    May span portions of several paragraphs. It matches the number of exposed
    keys, not their positions or all semantic information in their causal prefix.
    """
    _,spans=masking.paragraph_tokens(model,q['workers'][0])
    gold=sorted(g['index'] for g in q['gold'] if g['worker']==0)
    if len(gold)!=1: raise ValueError('matched access requires exactly one A support')
    n=len(spans[gold[0]])
    candidates=[i for i in range(gold[0])
                if not contains_answer(q['workers'][0][i],q['answers']+q.get('bridges',[]))]
    eligible=[t for i in candidates for t in spans[i]]
    if len(eligible)<n:
        raise ValueError(f'{q["id"]}: insufficient earlier irrelevant tokens ({len(eligible)} < {n})')
    start=int(digest([q['id'],'matched-access-v1']),16)%(len(eligible)-n+1)
    chosen=eligible[start:start+n]
    chosen_set=set(chosen)
    touched=[i for i in candidates if set(spans[i])&chosen_set]
    earlier='\n'.join(q['workers'][0][:max(touched)+1])
    return chosen,dict(a_visible_tokens=n,a_gold_tokens=n,a_paragraphs=touched,a_visible_indices=chosen,
                       a_gold_indices=spans[gold[0]],selected_keys_precede_gold=True,
                       earlier_prefix_mentions_answer_or_bridge=contains_answer(earlier,q['answers']+q.get('bridges',[])),
                       control='equal-token earlier non-support window; positions unmatched')


def matched_layout(model,q):
    selected,extra=matched_indices(model,q)
    ids,mask,sizes,meta=masking.layout(model,q,'gold')
    prefix=len(ids)-sum(sizes); b=prefix+sizes[0]
    mask[0,0,b:,prefix:b]=torch.finfo(mask.dtype).min
    mask[0,0,b:,[prefix+i for i in selected]]=0
    return ids,mask,sizes,{**meta,**extra,'condition':'matched_irrelevant'}


def generate(model,q,condition,mode,think_budget,answer_tokens):
    if condition=='b1':
        cache,pos,sizes=arms.b1_context(model,q)
        meta=dict(condition='b1',implementation='joint_prefill',encoding_positions='true',key_shift=False)
    elif condition=='k0':
        cache,pos,sizes=arms.k0_context(model,q,positions='local',merge='seq')
        meta=dict(condition='k0',implementation='separate_cache',encoding_positions='local',merge='seq',key_shift=True)
    elif condition=='matched_irrelevant':
        ids,mask,sizes,meta=matched_layout(model,q)
        out=model.forward(ids,attention_mask=mask)
        cache,pos=out.past_key_values,len(ids)
        del out,mask
    else:
        cache,pos,sizes,meta=masking.context(model,q,condition)
    expected,expected_sizes,_=paper_experiments.context_tokens(model,q)
    if sizes!=expected_sizes or pos!=len(expected) or kv.length(cache)!=pos:
        raise ValueError('encoding condition changed final token/cache positions')
    _,groups=masking.paragraph_tokens(model,q['workers'][0])
    meta['a_gold_tokens']=sum(len(groups[g['index']]) for g in q['gold'] if g['worker']==0)
    tail=model.ids(prompts.tail(model.fmt,mode))
    gen=model.greedy(tail,cache,pos,max_new=answer_tokens,think_budget=think_budget if mode=='forced' else None)
    return {**paper_experiments.generation_fields(gen),'worker_tokens':sizes,'mask':meta,
            'context_tokens_sha256':digest(expected),'final_answer_position':pos}


def probe(model,q,name,answer_tokens):
    if name=='closed_book':
        ids=model.ids(prompts.prefix(model.fmt,q['question']))
        cache,_=model.prefill(ids); pos,sizes=len(ids),[]
    else:
        hop={'keep_hop0':0,'keep_hop1':1}[name]
        cache,pos,sizes=arms.b1_context(model,q,keep_gold_hops=[hop])
    gen=model.greedy(model.ids(prompts.tail(model.fmt,'off')),cache,pos,max_new=answer_tokens)
    return {**paper_experiments.generation_fields(gen),'worker_tokens':sizes,
            'probe_definition':'closed book or one support plus all distractors, AB order only'}


def check_prompt_template(model,q):
    """Compare our text to the native template, using the repo's fixed date."""
    paragraphs=[p for w in q['workers'] for p in w]
    user=f'Target Question: {q["question"]}\n\n{prompts.INTRO}\n\n'+prompts.block(paragraphs)+prompts.FMT
    messages=[{'role':'system','content':prompts.SYSTEM},{'role':'user','content':user}]
    native=model.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                                         enable_thinking=False,date_string='26 Jul 2024')
    ours=prompts.prefix(model.fmt,q['question'])+prompts.block(paragraphs)+prompts.tail(model.fmt,'off')
    if native!=ours: raise ValueError(f'{model.fmt.name}: prompt differs from native chat template')
    return dict(format=model.fmt.name,prompt_sha256=digest(ours),native_template_matches=True,
                dedicated_thinking_disabled=bool(model.fmt.thinking))


def check_rope_shift(model):
    """Small forward-only cache check; no generated answers or accuracy criterion.

    Matches the existing real-model regression: compare shifted layer-0 keys to
    native encoding at the shifted positions. This does not assert K0 == blocked.
    """
    ids=model.ids('The mayor of Velmora is Bram Holt. The river flows through Irvale.')
    original,_=model.prefill(ids,start_pos=10)
    native,_=model.prefill(ids,start_pos=1210)
    shifted_keys=kv.layers(kv.shift(original,1200,model.inv_freq))[0][0].float()
    native_keys=kv.layers(native)[0][0].float()
    relative=((shifted_keys-native_keys).norm()/native_keys.norm().clamp_min(1e-12)).item()
    bound={'bfloat16':6e-3,'float16':1e-3,'float32':1e-4}[model.dtype_name]
    if not np.isfinite(relative) or relative>=bound:
        raise ValueError(f'RoPE shift differs from native layer-0 keys: relative L2 {relative}, bound {bound}')
    return dict(relative_l2=relative,bound=bound,offset=1200,layer=0,
                scope='rotation implementation only; not equivalence of full K0 and true-position encoding')


def preflight(model,questions,ids,max_tokens):
    """Check all inputs without filtering runtime failures or truncating context."""
    lengths,factorial={},{}
    for id in ids:
        q=questions[id]
        tokens,sizes,_=paper_experiments.context_tokens(model,q)
        segments=[prompts.prefix(model.fmt,q['question']),*[prompts.block(w) for w in q['workers']],
                  prompts.tail(model.fmt,'off')]
        if [t for segment in segments for t in model.ids(segment)]!=model.ids(''.join(segments)):
            raise ValueError(f'{id}: segmented tokenization differs from complete prompt tokenization')
        if len(tokens)>max_tokens:
            raise ValueError(f'{id}: context length {len(tokens)} exceeds {max_tokens}; no silent truncation')
        lengths[id]=dict(context_tokens=len(tokens),worker_tokens=sizes,input_ids_sha256=digest(tokens),
                         segmented_tokenization_matches=True)
        if q.get('layout_protocol')=='support-last-v1':
            _,extra=matched_indices(model,q); lengths[id].update(extra)
        if q['dataset']=='wmac_toy':
            key=(q['source_id'],q['placement'])
            if factorial.setdefault(key,sizes)!=sizes:
                raise ValueError(f'{id}: worker lengths changed across the factorial conditions')
            positions=[]; offset=len(tokens)-sum(sizes)
            for w in range(2):
                _,groups=masking.paragraph_tokens(model,q['workers'][w])
                for g in q['gold']:
                    if g['worker']==w:
                        positions.append((g['hop'],offset+groups[g['index']][0],len(groups[g['index']])))
                offset+=sizes[w]
            if factorial.setdefault((*key,'gold'),sorted(positions))!=sorted(positions):
                raise ValueError(f'{id}: gold positions changed across the factorial conditions')
    return lengths


def forced_logits(model,q,variant,continuation):
    if variant=='b1': cache,pos,_=arms.b1_context(model,q)
    elif variant=='b1_mask': cache,pos,_=arms.k0_context(model,q,positions='true')
    else:
        cache,pos,_,_=masking.context(model,q,{'masked_full':'full','masked_blocked':'blocked'}[variant])
    tail=model.ids(prompts.tail(model.fmt,'off'))
    out=model.forward(tail+continuation,cache,pos)
    values=out.logits[0,len(tail)-1:len(tail)+len(continuation)].float().cpu().numpy().copy()
    if not np.isfinite(values).all(): raise ValueError('non-finite teacher-forced logits')
    return values


def compare_same_prefix(model,q,reference,candidate,continuation,folder):
    """Feed both paths the reference's same prefix, including after free-generation divergence."""
    x=forced_logits(model,q,reference,continuation)
    y=forced_logits(model,q,candidate,continuation)
    if x.shape!=y.shape: raise ValueError('teacher-forcing shape mismatch')
    positions={0,len(continuation),*[p for p in (1,2,4,8,16,32,64,128) if p<=len(continuation)]}
    steps=[]
    for i,(a,b) in enumerate(zip(x,y)):
        la,lb=a-np.logaddexp.reduce(a),b-np.logaddexp.reduce(b)
        ta,tb=np.argpartition(a,-2)[-2:],np.argpartition(b,-2)[-2:]
        steps.append(dict(prefix_tokens=i,argmax_matches=bool(a.argmax()==b.argmax()),
            argmax_ids=[int(a.argmax()),int(b.argmax())],max_abs_delta=float(np.max(np.abs(a-b))),
            mean_abs_delta=float(np.mean(np.abs(a-b))),kl_reference_to_candidate=float(np.sum(np.exp(la)*(la-lb))),
            top2_margins=[float(abs(a[ta[0]]-a[ta[1]])),float(abs(b[tb[0]]-b[tb[1]]))]))
    first=next((s['prefix_tokens'] for s in steps if not s['argmax_matches']),None)
    if first is not None: positions.add(first)
    positions=sorted(positions)
    name='logits/common-prefix-'+digest([q['id'],reference,candidate])[:20]+'.npz'
    path=Path(folder)/name; path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,prefix_positions=np.asarray(positions),reference=x[positions],candidate=y[positions])
    temporary.replace(path)
    return dict(reference=reference,candidate=candidate,continuation_token_ids=continuation,
                continuation_sha256=digest(continuation),steps=steps,logits_file=name,
                interpretation='Shared-prefix numerical diagnostics; not an automatic parity verdict.')
