#!/usr/bin/env python3
"""Three independent, resumable jobs. --plan-only never loads a model."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from latent_dispatch.paper_plan import (PLAN, SEED, RowStore, atomic_json, digest, file_sha,
                                        frozen_json, load_plan, specifications, stable_order)


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',choices=['validation','thinking','likelihood'])
    p.add_argument('--run-name',default='paper_v1')
    p.add_argument('--benchmark-dir',type=Path,default=ROOT/'benchmark')
    p.add_argument('--output-root',type=Path,default=ROOT/'results')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--max-hours',type=float,default=8.0,help='Soft limit checked between evaluations; resume with the same command.')
    p.add_argument('--device',default='cuda')
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16',help='float32 is only for a separate validation diagnostic.')
    p.add_argument('--revision',default=None,help='Optional Hugging Face commit; actual resolved model/tokenizer are recorded.')
    p.add_argument('--attention',choices=['sdpa','eager'],default='sdpa')
    args=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name):
        p.error('--run-name must contain only letters, digits, underscores, and hyphens')
    if not 0 < args.max_hours < float('inf'):
        p.error('--max-hours must be finite and positive')
    if args.dtype!='bfloat16' and args.job!='validation':
        p.error('the frozen thinking and likelihood protocols require bfloat16')
    if sum([args.plan_only,args.smoke_only,args.summarize])>1:
        p.error('choose only one of --plan-only, --smoke-only, --summarize')
    return args


def code_state():
    files=sorted((ROOT/'latent_dispatch').glob('*.py'))+[Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)):file_sha(p) for p in files}


def commit():
    return subprocess.check_output(['git','-c','core.fsmonitor=false','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def smoke_ids(job,ids,questions):
    # Deterministic examples already in the frozen sample; stored separately from main results.
    by_cell={}
    for id in ids:
        q=questions[id]
        by_cell.setdefault((q['dataset'],q['placement']),id)
    return [by_cell[k] for k in sorted(by_cell)]


def complete_box(text,forced=False):
    if forced:
        if '</think>' not in text:
            return False
        text=text.rsplit('</think>',1)[1]
    start=text.find('\\boxed{')
    if start<0:
        return False
    start+=len('\\boxed{');depth=1
    for i in range(start,len(text)):
        depth+=(text[i]=='{')-(text[i]=='}')
        if depth==0:
            return bool(text[start:i].strip())
    return False


def check_smoke(store,specs,job,expected):
    if len(store.rows)!=expected:
        raise ValueError('smoke is incomplete; rerun to finish it before full inference')
    if job=='likelihood':
        return
    for (id,name),row in store.rows.items():
        if job=='validation':
            # Truncation is itself an endpoint diagnostic at the archived 256-token cap.
            if not row['text'].strip():
                raise ValueError(f'empty endpoint smoke output: {id} / {name}')
            continue
        if row['cut'] or not complete_box(row['text'],specs[name]['thinking']=='forced'):
            raise ValueError(f'smoke failed on {id} / {name}: inspect smoke/rows.jsonl; no main inference started')


def schedule(ids,specs):
    # Complete every condition of each layout, then its paired placement. No arm-major overnight partials.
    return [(id,name) for id in ids for name in stable_order(list(specs),'schedule:'+id)]


def verify_saved_rows(store,specs,questions):
    for (id,name),row in store.rows.items():
        if row.get('options')!=specs[name] or row.get('question_sha256')!=digest(questions[id]):
            raise ValueError(f'saved row settings or question changed: {id} / {name}')


def execute_phase(job,ids,specs,questions,store,evaluate,stop_requested,deadline,commit_id):
    total=len(ids)*len(specs)
    for id,name in schedule(ids,specs):
        if (id,name) in store.rows:
            continue
        if stop_requested() or time.monotonic()>=deadline:
            return False
        print(f'{job} {len(store.rows)+1}/{total} {id} {name}',flush=True)
        start=time.perf_counter()
        out=evaluate(questions[id],name,specs[name])
        row={'id':id,'condition':name,'options':specs[name],'commit':commit_id,
             'question_sha256':digest(questions[id]),'seconds':time.perf_counter()-start,**out}
        store.append(row)
        details={k:row[k] for k in ['seconds','gen_tokens','think_tokens','answer_tokens','cut','think_cut','think_eos'] if k in row}
        print(json.dumps(details),flush=True)
    return True


def runtime_metadata(model):
    import torch
    return {'python':platform.python_version(),
            'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','numpy']},
            'cuda':torch.version.cuda,'device':str(model.device),
            'device_name':torch.cuda.get_device_name(model.device) if model.device.type=='cuda' else str(model.device),
            'model_revision':getattr(model.hf.config,'_commit_hash',None),
            'model_config_sha256':digest(model.hf.config.to_dict()),
            'tokenizer_sha256':digest(model.tok.backend_tokenizer.to_str()),
            'attention_implementation':getattr(model.hf.config,'_attn_implementation',None),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_deterministic':torch.backends.cudnn.deterministic}


def main(argv=None):
    args=arguments(argv)
    plan,questions=load_plan(args.benchmark_dir)
    specs=specifications(args.job,plan);ids=plan[args.job]['ids']
    if len(ids)*len(specs)!=plan[args.job]['expected_rows']:
        raise ValueError('frozen evaluation count changed')
    folder=args.output_root/args.run_name/args.job
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest={'protocol':plan['protocol'],'job':args.job,'plan_sha256':file_sha(PLAN),
              'code_sha256':code_state(),'model':plan['model'],'dtype':args.dtype,
              'revision_requested':args.revision,'attention_requested':args.attention,
              'ids':ids,'specifications':specs,'question_sha256':{id:digest(questions[id]) for id in ids},
              'sampling_seed':SEED,'expected_rows':len(ids)*len(specs)}
    frozen_json(folder/'manifest.json',manifest)
    fingerprint=digest(manifest)
    print(f'{args.job}: {len(ids)} layouts, {len(specs)} conditions, {len(ids)*len(specs)} main evaluations',flush=True)
    print(f'Manifest: {folder / "manifest.json"}',flush=True)
    if args.plan_only:
        return 0
    from latent_dispatch.paper_summary import summarize
    main_folder=folder/'main'
    main_store=RowStore(main_folder/'rows.jsonl',fingerprint,schedule(ids,specs))
    smoke=smoke_ids(args.job,ids,questions);smoke_folder=folder/'smoke'
    smoke_store=RowStore(smoke_folder/'rows.jsonl',fingerprint,schedule(smoke,specs))
    verify_saved_rows(main_store,specs,questions)
    verify_saved_rows(smoke_store,specs,questions)
    if (main_store.rows or smoke_store.rows) and not (folder/'runtime.json').exists():
        raise ValueError('saved responses require their original runtime.json; restore the complete bundle')
    if args.summarize or len(main_store.rows)==len(ids)*len(specs):
        summary=summarize(args.job,ids,specs,questions,main_store,main_folder)
        print(f'Saved {summary["saved_rows"]}/{summary["expected_rows"]}; complete={summary["complete"]}')
        return 0
    from latent_dispatch import paper_experiments as experiments
    from latent_dispatch.model import Model
    import torch
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use an A100-class GPU for these 8B runs')
        if torch.cuda.get_device_properties(torch.device(args.device)).total_memory<30*2**30:
            raise RuntimeError('Use a GPU with at least 30 GiB for this 8B protocol')
    torch.manual_seed(SEED)
    start=time.monotonic();deadline=start+args.max_hours*3600
    model=Model.load(plan['model'],dtype=args.dtype,device=args.device,revision=args.revision,
                     attn_implementation=args.attention)
    frozen_json(folder/'runtime.json',runtime_metadata(model))
    stopped=[False]
    old_handlers={}
    def signal_stop(signum,frame):
        stopped[0]=True
        print('Stop requested; checkpointing after the current evaluation.',flush=True)
    for sig in [signal.SIGTERM,signal.SIGINT]:
        old_handlers[sig]=signal.signal(sig,signal_stop)
    def evaluator(where):
        def evaluate(q,name,options):
            if args.job=='validation':
                return experiments.endpoint(model,q,options['variant'],options['answer_tokens'],where)
            if args.job=='thinking':
                return experiments.thinking(model,q,options['condition'],options['thinking'],
                                             options['think_budget'],options['answer_tokens'])
            return experiments.likelihood(model,q,options['condition'],plan['likelihood']['candidates'][q['id']])
        return evaluate
    status='interrupted'
    try:
        finished=execute_phase(args.job,smoke,specs,questions,smoke_store,evaluator(smoke_folder),
                               lambda:stopped[0],deadline,commit())
        summarize(args.job,smoke,specs,questions,smoke_store,smoke_folder)
        if not finished:
            return 2
        check_smoke(smoke_store,specs,args.job,len(smoke)*len(specs))
        atomic_json(folder/'smoke_passed.json',{'fingerprint':fingerprint,'rows':len(smoke_store.rows)})
        if args.smoke_only:
            status='smoke_passed'
            return 0
        finished=execute_phase(args.job,ids,specs,questions,main_store,evaluator(main_folder),
                               lambda:stopped[0],deadline,commit())
        status='complete' if finished else 'interrupted'
        return 0 if finished else 2
    except BaseException as error:
        status='failed'
        atomic_json(folder/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)
        summary=summarize(args.job,ids,specs,questions,main_store,main_folder)
        atomic_json(folder/'status.json',{'status':status,'saved_rows':summary['saved_rows'],
                    'expected_rows':summary['expected_rows'],'seconds':time.monotonic()-start})
        # Bundle just this job, including logits and manifests, for inspection by teammates.
        archive=shutil.make_archive(str(folder),'gztar',root_dir=folder.parent,base_dir=folder.name)
        print(f'{status}: {summary["saved_rows"]}/{summary["expected_rows"]}; bundle {archive}',flush=True)


if __name__=='__main__':
    raise SystemExit(main())
