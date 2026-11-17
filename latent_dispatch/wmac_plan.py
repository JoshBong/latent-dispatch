"""Input preparation for the WMAC follow-ups. Nothing executes on import.

Preparation uses tokenizers but never loads model weights. It freezes selected
questions before inference. Literal grounding checks are not human adjudication.
"""
import copy
import json
import random
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from . import prompts, toy
from .paper_plan import ROOT, digest, file_sha, frozen_json
from .score import contains_answer, normalize

CONFIG = ROOT/'experiments/wmac_gpu_v3.json'
JOBS = ('validation', 'validation-llama', 'validation-4b', 'musique-8b', 'musique-llama', 'musique-4b', 'toy-8b')
MASKS = ('blocked', 'gold', 'full', 'matched_irrelevant')
CONDITIONS = (*MASKS, 'b1', 'k0')


def is_validation(job):
    return job in ('validation','validation-llama','validation-4b')


def model_key(job):
    if job not in JOBS: raise ValueError(f'unknown job: {job}')
    return 'llama' if job.endswith('-llama') else '4b' if job.endswith('-4b') else '8b'


def gpu_profile(job,cfg):
    matches=[(name,profile) for name,profile in cfg['gpu_jobs'].items() if job in profile['jobs']]
    if len(matches)!=1: raise ValueError(f'{job}: expected exactly one GPU assignment')
    return matches[0]


def supports(q):
    return [q['workers'][g['worker']][g['index']] for g in sorted(q['gold'],key=lambda g:g['hop'])]


def support_key(text):
    return digest(normalize(text.split(': ',1)[-1]))


def historical_sources(folder):
    sources, archives = set(), {}
    for path in sorted(Path(folder).glob('*.tar.gz')):
        archives[path.name] = file_sha(path)
        with tarfile.open(path) as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith('.jsonl') or Path(m.name).name.startswith('._'):
                    continue
                for line in tar.extractfile(m):
                    source = json.loads(line).get('id','').split(':')[0]
                    if source.startswith('2hop__'):
                        sources.add(source)
    return sources, archives


def support_last(q):
    """Identical layout in all masks; earlier control keys cannot encode later A gold.

    This deliberately differs from historical random layouts. Key positions and
    distance are NOT matched between gold and irrelevant access.
    """
    q = copy.deepcopy(q)
    for w, paragraphs in enumerate(q['workers']):
        refs = [g for g in q['gold'] if g['worker']==w]
        if len(refs)!=1:
            raise ValueError('need one support per worker')
        index = refs[0]['index']
        q['workers'][w] = [p for i,p in enumerate(paragraphs) if i!=index]+[paragraphs[index]]
        refs[0]['index'] = len(paragraphs)-1
    q['layout_protocol'] = 'support-last-v1'
    return q


def grounding_flags(q):
    gold = supports(q)
    if len(gold)!=2 or len(set(map(support_key,gold)))!=2:
        return ['invalid_support_pair']
    flags = []
    if not contains_answer(gold[1],q['answers']):
        flags.append('answer_absent_from_second_support')
    if not q.get('bridges') or not all(contains_answer(p,q['bridges']) for p in gold):
        flags.append('bridge_not_explicit_in_both_supports')
    if contains_answer(gold[0],q['answers']):
        flags.append('answer_explicit_in_first_support')
    return flags


def select_sources(questions, excluded, n, seed, reviews=None, eligible=None):
    reviews = reviews or {}
    groups = {}
    for q in questions:
        if q['seed']==0 and q['qtype']=='2hop' and q['placement'] in ('AB','BA'):
            groups.setdefault(q['source_id'],{})[q['placement']] = q
    if not isinstance(reviews,dict) or set(reviews)-groups.keys():
        raise ValueError('review file must map known source IDs to review decisions')
    used_supports = {support_key(p) for s in excluded if s in groups
                     for p in supports(next(iter(groups[s].values())))}
    used_questions = {normalize(next(iter(groups[s].values()))['question']) for s in excluded if s in groups}
    selected, audit = [], []
    # Retain V1's input-only source ordering; V2 adds feasibility for all three tokenizers.
    for source in sorted(groups,key=lambda s:digest([seed,'wmac_a100_v1',s])):
        pair = groups[source]
        q = copy.deepcopy(next(iter(pair.values())))
        review = reviews.get(source,{})
        if not isinstance(review,dict) or (review and (not review.get('reviewer') or not review.get('reason'))):
            raise ValueError(f'{source}: review needs reviewer and reason')
        aliases = review.get('aliases',[])
        if not isinstance(aliases,list) or not all(isinstance(a,str) and a.strip() for a in aliases):
            raise ValueError('aliases must be a list of nonempty strings')
        q['answers'] = list(dict.fromkeys(q['answers']+aliases))
        flags = grounding_flags(q)
        if source in excluded: flags.append('historical_source')
        if set(pair)!={'AB','BA'}: flags.append('missing_order')
        if review.get('exclude'): flags.append('human_exclusion')
        keys = set(map(support_key,supports(q)))
        if keys & used_supports: flags.append('shared_support_text')
        if normalize(q['question']) in used_questions: flags.append('duplicate_question')
        prepared = []
        if not flags:
            for order in ('AB','BA'):
                item = support_last(pair[order]); item['answers'] = q['answers']
                prepared.append(item)
                if eligible and not eligible(item): flags.append('input_token_control_ineligible')
        audit.append(dict(source_id=source,flags=sorted(set(flags)),human_review=review or None))
        if flags: continue
        selected.extend(prepared)
        used_supports.update(keys); used_questions.add(normalize(q['question']))
        if len(selected)==2*n: break
    if len(selected)!=2*n:
        raise ValueError(f'only {len(selected)//2} eligible independent sources; need {n}')
    return selected,audit


def factorial_toy(n,seed,tokenizer):
    """Keep People placement and gold positions fixed; swap only a distractor and filler."""
    out = []
    ids = lambda text:tokenizer(text,add_special_tokens=False).input_ids
    words = ['unused','paper','empty','folders','spare','envelopes','blank','labels']
    for it in toy.items(1,n,seed,n_distract=2):
        source = f'wmac-toy-s{seed}-i{it["i"]}'
        labels = random.Random(source).sample(['A','B'],2)
        real = it['shards'][1].replace('Cities:',f'Cities {labels[0]}:',1)
        fake = it['distract'][1].replace('Cities:',f'Cities {labels[0]}:',1)
        target = len(ids(prompts.block([fake])))
        for count in range(1,1024):
            filler = 'Records: '+' '.join(words[i%len(words)] for i in range(count))+'.'
            if len(ids(prompts.block([filler])))==target: break
        else:
            raise ValueError(f'cannot balance filler for {source}')
        for order in ('AB','BA'):
            for cities in ('together','split'):
                for headings in ('identical','distinct'):
                    label = labels[0] if headings=='identical' else labels[1]
                    distractor = it['distract'][1].replace('Cities:',f'Cities {label}:',1)
                    people = [it['shards'][0], 'Records: The inventory contains blank notebooks and unused envelopes.',
                              filler if cities=='together' else distractor]
                    city = [it['distract'][0], real, distractor if cities=='together' else filler]
                    pw,cw = (0,1) if order=='AB' else (1,0)
                    out.append(dict(id=f'{source}:{order}:{cities}:{headings}',source_id=source,
                        dataset='wmac_toy',qtype='hop1',hops=2,crossings=1,seed=seed,placement=order,
                        question=it['question'],answers=[it['gold']],workers=[people,city] if order=='AB' else [city,people],
                        gold=[{'hop':0,'worker':pw,'index':0},{'hop':1,'worker':cw,'index':1}],
                        bridges=it['chain'][1:-1],decomposition=[],cities=cities,headings=headings,
                        title_labels=labels,layout_protocol='factorial-people-fixed-v1'))
    return out


def read_benchmarks(folder):
    expected = {name:sha for sha,name in (line.split() for line in (ROOT/'benchmark/SHA256SUMS').read_text().splitlines())}
    questions,hashes = {},{}
    for name in ('musique.jsonl','toy.jsonl'):
        path = Path(folder)/name
        if not path.exists():
            raise FileNotFoundError(f'{path}: first build with python -m latent_dispatch.data toy musique')
        hashes[name] = file_sha(path)
        if hashes[name]!=expected[name]: raise ValueError(f'benchmark checksum mismatch: {path}')
        with path.open() as f:
            for line in f:
                q=json.loads(line)
                if q['id'] in questions: raise ValueError('duplicate benchmark ID')
                questions[q['id']]=q
    return questions,hashes


def reused_sources(prior,n,eligible):
    """Preserve existing source text, order, and aliases; never substitute failures."""
    fresh=[copy.deepcopy(prior['questions'][id]) for id in prior['musique_ids']]
    groups={}
    for q in fresh: groups.setdefault(q['source_id'],[]).append(q['placement'])
    if len(fresh)!=2*n or len(groups)!=n or any(sorted(v)!=['AB','BA'] for v in groups.values()):
        raise ValueError(f'source reuse needs the same {n} paired source questions')
    for q in fresh:
        if not eligible(q): raise ValueError(f'{q["id"]}: reused input fails tokenizer feasibility; no question was replaced')
    return fresh


def prepare(folder, destination, reviews=None, config_path=CONFIG, reuse_prepared=None):
    """Josh runs this before inference; existing prepared inputs are never regenerated."""
    destination=Path(destination)
    if destination.exists():
        if reviews is not None or reuse_prepared is not None:
            raise ValueError('human revisions or source reuse need a NEW prepared path and run name')
        return load(destination)
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    from .wmac_experiments import preflight, matched_indices, check_prompt_template
    cfg=json.loads(Path(config_path).read_text())
    if reuse_prepared is not None and reviews is not None:
        raise ValueError('source reuse preserves answers; do not combine it with new review decisions')
    prior=_read_bundle(reuse_prepared,('wmac_a100_v1','wmac_a100_v2','wmac_gpu_v3'))[0] if reuse_prepared else None
    adapters={}
    for key,model in cfg['models'].items():
        if prior is not None:
            old=prior['config']['models'].get(key,{})
            if old.get('name')==model['name'] and isinstance(old.get('revision'),str) and len(old['revision'])==40:
                model['revision']=old['revision']
        model['requested_revision']=model['revision']
        model['revision']=HfApi().model_info(model['name'],revision=model['revision'] or 'main').sha
        if not isinstance(model['revision'],str) or len(model['revision'])!=40:
            raise ValueError(f'{model["name"]}: could not resolve an immutable model revision')
        tokenizer=AutoTokenizer.from_pretrained(model['name'],revision=model['revision'])
        adapters[key]=SimpleNamespace(tok=tokenizer,fmt=prompts.FORMATS[model['format']],
                            ids=lambda text,tok=tokenizer:tok(text,add_special_tokens=False).input_ids)
    questions,hashes=read_benchmarks(folder)
    excluded,archives=historical_sources(ROOT/'archive')
    def eligible(q):
        for adapter in adapters.values():
            try: matched_indices(adapter,q)
            except ValueError as error:
                if 'insufficient earlier irrelevant tokens' in str(error): return False
                raise
            if (len(adapter.ids(prompts.prefix(adapter.fmt,q['question'])))+
                    sum(len(adapter.ids(prompts.block(w))) for w in q['workers']))>cfg['max_context_tokens']:
                return False
        return True
    if reuse_prepared is None:
        fresh,audit=select_sources(questions.values(),excluded,cfg['musique_sources'],cfg['seed'],reviews,eligible)
    else:
        if prior['benchmark_sha256']!=hashes: raise ValueError('source-reuse benchmark mismatch')
        fresh=reused_sources(prior,cfg['musique_sources'],eligible)
        reviews=prior['reviews'];audit=prior['selection_audit']
    synthetic=factorial_toy(cfg['toy_sources'],cfg['seed'],adapters['8b'].tok)
    old=json.loads((ROOT/'experiments/paper_ablations_v1.json').read_text())
    diagnostic=[questions[id] for id in old['validation']['ids']]
    chosen={q['id']:q for q in diagnostic+fresh+synthetic}
    checked={};templates={}
    for key,adapter in adapters.items():
        checked_ids=list(chosen) if key=='8b' else [q['id'] for q in diagnostic+fresh]
        checked[key]=preflight(adapter,chosen,checked_ids,cfg['max_context_tokens'])
        templates[key]=check_prompt_template(adapter,fresh[0])
    bundle=dict(config=cfg,config_sha256=file_sha(config_path),benchmark_sha256=hashes,
                archive_sha256=archives,excluded_historical_sources=sorted(excluded),reviews=reviews or {},
                selection_audit=audit,exclusion_counts=dict(Counter(flag for row in audit for flag in row['flags'])),
                questions=chosen,question_sha256={id:digest(q) for id,q in chosen.items()},
                validation_ids=[q['id'] for q in diagnostic],musique_ids=[q['id'] for q in fresh],
                toy_ids=[q['id'] for q in synthetic],preflight=checked,
                prompt_template_checks=templates,
                reused_prepared_sha256=file_sha(reuse_prepared) if reuse_prepared else None,
                tokenizer_sha256={key:digest(a.tok.backend_tokenizer.to_str()) for key,a in adapters.items()})
    frozen_json(destination,bundle)
    review_packet=[dict(source_id=q['source_id'],question=q['question'],answers=q['answers'],
                        supports=supports(q),bridges=q['bridges'],decomposition=q['decomposition'],
                        human_review=(reviews or {}).get(q['source_id'])) for q in fresh[::2]]
    frozen_json(destination.with_name(destination.stem+'-review.json'),review_packet)
    return bundle,chosen


def _read_bundle(path,allowed):
    bundle=json.loads(Path(path).read_text())
    if bundle['config']['protocol'] not in allowed:
        raise ValueError('prepared protocol mismatch; prepare v3 or use the original pinned code for earlier runs')
    qs=bundle['questions']
    if {id:digest(q) for id,q in qs.items()}!=bundle['question_sha256']:
        raise ValueError('prepared question hash mismatch')
    ids=bundle['validation_ids']+bundle['musique_ids']+bundle['toy_ids']
    if len(ids)!=len(set(ids)) or set(ids)!=set(qs): raise ValueError('prepared IDs are inconsistent')
    return bundle,qs


def load(path):
    bundle,qs=_read_bundle(path,('wmac_gpu_v3',))
    if set(bundle['config']['models'])!={'8b','4b','llama'}:
        raise ValueError('v3 requires Qwen3-8B, Qwen3-4B, and Llama')
    for model in bundle['config']['models'].values():
        if not isinstance(model['revision'],str) or len(model['revision'])!=40:
            raise ValueError('v3 requires frozen model revisions from preparation')
    profiles=bundle['config']['gpu_jobs']
    expected={'t4':('float16',{'validation-4b','musique-4b'}),
              'a100':('bfloat16',set(JOBS)-{'validation-4b','musique-4b'})}
    if set(profiles)!=set(expected): raise ValueError('v3 needs T4 and A100 jobs')
    for gpu,(dtype,jobs) in expected.items():
        if profiles[gpu]['dtype']!=dtype or set(profiles[gpu]['jobs'])!=jobs or len(profiles[gpu]['jobs'])!=len(jobs):
            raise ValueError(f'{gpu}: invalid precision or component assignment')
    return bundle,qs


def job_ids(job,bundle):
    return bundle['validation_ids' if is_validation(job) else 'musique_ids' if job.startswith('musique') else 'toy_ids']


def specifications(job,cfg):
    if is_validation(job):
        return {a:{'variant':a,'thinking':'off','answer_tokens':cfg['validation_answer_tokens']}
                for a in ('b1','masked_full','b1_mask','masked_blocked')}
    masks=CONDITIONS if job.startswith('musique') else ('blocked','gold')
    modes=('off',) if job.startswith('musique') else ('off','forced')
    return {c+'_'+m:{'condition':c,'thinking':m,'answer_tokens':cfg['answer_tokens'],'think_budget':cfg['think_budget']}
            for m in modes for c in masks}
