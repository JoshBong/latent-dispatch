#!/usr/bin/env python3
"""Explicit preparation and resumable A100 jobs. No execution on import."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from latent_dispatch import wmac_plan
from latent_dispatch.paper_plan import RowStore,atomic_json,digest,file_sha,frozen_json
from scripts.run_paper_ablations import code_state,commit,execute_phase,runtime_metadata,schedule,verify_saved_rows


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',choices=['prepare',*wmac_plan.JOBS])
    p.add_argument('--prepared',type=Path,help='Frozen prepared.json; defaults to output-root/run-name/prepared.json')
    p.add_argument('--benchmark-dir',type=Path,default=ROOT/'benchmark')
    p.add_argument('--output-root',type=Path,default=ROOT/'results')
    p.add_argument('--run-name',default='wmac_a100_v2')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--smoke-only',action='store_true')
    mode.add_argument('--summarize',action='store_true')
    p.add_argument('--max-hours',type=float,default=24)
    p.add_argument('--device',default='cuda')
    p.add_argument('--dtype',choices=['bfloat16','float32'],default='bfloat16')
    p.add_argument('--attention',choices=['sdpa','eager'],default='sdpa')
    p.add_argument('--revision',help='HF revision override; actual revision is frozen on resume')
    p.add_argument('--validation-limit',type=int,help='Separate numerical diagnostic on N shortest frozen layouts')
    p.add_argument('--reviews',type=Path,help='prepare only: pre-output human exclusions/aliases, with reviewer and reason')
    p.add_argument('--reuse-prepared',type=Path,help='prepare only: preserve an existing 200-question cohort and answers exactly')
    args=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name): p.error('invalid run name')
    if not 0<args.max_hours<float('inf'): p.error('--max-hours must be finite and positive')
    if args.job!='prepare' and not wmac_plan.is_validation(args.job) and (args.dtype!='bfloat16' or args.attention!='sdpa'):
        p.error('main experiments require BF16 + SDPA')
    if args.validation_limit is not None and (not wmac_plan.is_validation(args.job) or not 1<=args.validation_limit<=24):
        p.error('--validation-limit is only for validation and must be between 1 and 24')
    if args.reviews and args.job!='prepare': p.error('human reviews must be frozen during preparation')
    if args.reuse_prepared and args.job!='prepare': p.error('--reuse-prepared applies only to preparation')
    if args.reuse_prepared and args.reviews: p.error('source reuse preserves aliases; do not also supply new reviews')
    if args.job=='prepare' and (args.plan_only or args.smoke_only or args.summarize):
        p.error('prepare is tokenizer/data preparation; these modes apply only to inference jobs')
    if args.prepared is None: args.prepared=args.output_root/args.run_name/'prepared.json'
    return args


def smoke_ids(ids,questions):
    cells={}
    for id in ids:
        q=questions[id]
        cells.setdefault((q['dataset'],q['placement'],q.get('cities',''),q.get('headings','')),id)
    return list(cells.values())


def check_smoke(store,specs):
    # Natural wrong answers, malformed boxes, or cap hits remain outcomes, never exclusions.
    for (id,name),row in store.rows.items():
        if not row['text'].strip(): raise ValueError(f'empty smoke answer: {id}/{name}')
        if specs[name]['thinking']=='forced' and '</think>' not in row['text']:
            raise ValueError(f'forced thinking did not close: {id}/{name}')
        print(json.dumps(dict(smoke_id=id,condition=name,text=row['text'],think_cut=row['think_cut'],cut=row['cut']),
                         ensure_ascii=False),flush=True)


def store_for(folder,phase,fingerprint,ids,specs):
    return RowStore(folder/phase/'rows.jsonl',fingerprint,schedule(ids,specs))


def numerical_diagnostics(model,ids,questions,main,diagnostics,folder,stop,deadline):
    from latent_dispatch.wmac_experiments import compare_same_prefix
    for id in ids:
        for reference,candidate in [('b1','masked_full'),('b1_mask','masked_blocked')]:
            if (id,candidate) in diagnostics.rows: continue
            if stop() or time.monotonic()>=deadline: return False
            start=time.perf_counter()
            out=compare_same_prefix(model,questions[id],reference,candidate,main.rows[id,reference]['token_ids'],folder/'diagnostics')
            diagnostics.append(dict(id=id,condition=candidate,question_sha256=digest(questions[id]),
                                    seconds=time.perf_counter()-start,**out))
            print(f'same-prefix diagnostics {len(diagnostics.rows)}/{2*len(ids)}',flush=True)
    return True


def main(argv=None):
    args=arguments(argv)
    if args.job=='prepare':
        reviews=json.loads(args.reviews.read_text()) if args.reviews else None
        args.prepared.parent.mkdir(parents=True,exist_ok=True)
        with args.prepared.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            bundle,_=wmac_plan.prepare(args.benchmark_dir,args.prepared,reviews,reuse_prepared=args.reuse_prepared)
        print(json.dumps(dict(prepared=str(args.prepared),main=bundle['config']['expected_main_rows'],
                              probes=bundle['config']['expected_probe_rows'],screening=bundle['config']['screening']),indent=2))
        return 0
    if not args.prepared.exists(): raise FileNotFoundError(f'{args.prepared}: run the prepare command first')
    bundle,questions=wmac_plan.load(args.prepared)
    ids=list(wmac_plan.job_ids(args.job,bundle)); cfg=bundle['config']
    if args.validation_limit:
        ids=sorted(ids,key=lambda id:sum(len(p) for w in questions[id]['workers'] for p in w))[:args.validation_limit]
    specs=wmac_plan.specifications(args.job,cfg)
    if not args.validation_limit and len(ids)*len(specs)!=cfg['expected_main_rows'][args.job]:
        raise ValueError('frozen row count mismatch')
    folder=args.output_root/args.run_name/args.job
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run(args,bundle,questions,ids,specs,folder)


def run(args,bundle,questions,ids,specs,folder):
    cfg=bundle['config']
    family=wmac_plan.model_key(args.job)
    model_cfg=cfg['models'][family]
    if args.revision and args.revision!=model_cfg['revision']:
        raise ValueError('model revision differs from the prepared protocol; prepare a separate run')
    revision=args.revision or model_cfg['revision']
    files=code_state();files[str(Path(__file__).relative_to(ROOT))]=file_sha(__file__)
    manifest=dict(protocol=cfg['protocol'],job=args.job,prepared_sha256=file_sha(args.prepared),code_sha256=files,
        model=model_cfg['name'],model_format=model_cfg['format'],revision_requested=revision,dtype=args.dtype,attention=args.attention,
        ids=ids,specifications=specs,question_sha256={id:digest(questions[id]) for id in ids},
        expected_main_rows=len(ids)*len(specs),screening=cfg['screening'])
    frozen_json(folder/'manifest.json',manifest)
    frozen_json(folder/'prepared.json',bundle)
    fingerprint=digest(manifest)
    print(f'{args.job}: {len(ids)*len(specs)} main generations; {folder/"manifest.json"}',flush=True)
    if args.plan_only: return 0
    smoke=smoke_ids(ids,questions)
    main_store=store_for(folder,'main',fingerprint,ids,specs)
    smoke_store=store_for(folder,'smoke',fingerprint,smoke,specs)
    probe_ids=[id for id in ids if questions[id]['placement']=='AB'] if args.job.startswith('musique') else []
    probe_specs={name:{'thinking':'off','answer_tokens':cfg['probe_answer_tokens']} for name in ('closed_book','keep_hop0','keep_hop1')}
    probes=store_for(folder,'probes',fingerprint,probe_ids,probe_specs)
    diagnostics=RowStore(folder/'diagnostics/rows.jsonl',fingerprint,[(id,c) for id in ids for c in ('masked_full','masked_blocked')])
    for store,options in [(main_store,specs),(smoke_store,specs),(probes,probe_specs)]:
        verify_saved_rows(store,options,questions)
    for (id,_),row in diagnostics.rows.items():
        if row['question_sha256']!=digest(questions[id]): raise ValueError('diagnostic question changed')
    if (main_store.rows or smoke_store.rows or probes.rows or diagnostics.rows) and not (folder/'runtime.json').exists():
        raise ValueError('saved responses lack runtime metadata; restore the complete bundle')
    for phase,store in [('main',main_store),('smoke',smoke_store),('diagnostics',diagnostics)]:
        for row in store.rows.values():
            for key in ('first_logits_file','logits_file'):
                if key in row:
                    path=(folder/phase/row[key]).resolve()
                    if not path.is_relative_to((folder/phase).resolve()) or not path.is_file():
                        raise ValueError(f'missing or invalid numerical artifact: {row[key]}')
    def summarize():
        if wmac_plan.is_validation(args.job):
            from latent_dispatch.paper_summary import summarize as old_summary
            summary=old_summary('validation',ids,specs,questions,main_store,folder/'main')
            summary['job']=args.job
            summary['common_prefix_diagnostics']=dict(saved=len(diagnostics.rows),expected=2*len(ids),file='../diagnostics/rows.jsonl')
            summary['complete']=summary['complete'] and len(diagnostics.rows)==2*len(ids)
            atomic_json(folder/'main/summary.json',summary)
            return summary
        from latent_dispatch.wmac_summary import summarize as new_summary
        return new_summary(args.job,ids,specs,questions,main_store,folder/'main',cfg,probes)
    finished=(len(main_store.rows)==len(ids)*len(specs) and len(probes.rows)==len(probe_ids)*len(probe_specs)
              and (not wmac_plan.is_validation(args.job) or len(diagnostics.rows)==2*len(ids)))
    if args.summarize or finished:
        summary=summarize()
        archive=shutil.make_archive(str(folder),'gztar',root_dir=folder.parent,base_dir=folder.name)
        print(json.dumps(dict(complete=finished,saved_rows=summary['saved_rows'],archive=archive),indent=2))
        return 0 if finished else 2
    from latent_dispatch.model import Model
    from latent_dispatch import paper_experiments,wmac_experiments
    import torch
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; run this job on the A100')
        if family in ('8b','llama') and torch.cuda.get_device_properties(torch.device(args.device)).total_memory<30*2**30:
            raise RuntimeError('8B protocol requires at least 30 GiB GPU memory')
    torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();deadline=start+args.max_hours*3600;stopped=[False]
    def stop(signum,frame):
        stopped[0]=True
        print('Stop requested; saving the current evaluation before exit.',flush=True)
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    status='interrupted'
    try:
        if revision is None and (folder/'runtime.json').exists():
            revision=json.loads((folder/'runtime.json').read_text())['model_revision']
        model=Model.load(model_cfg['name'],device=args.device,dtype=args.dtype,fmt=model_cfg['format'],
                         revision=revision,attn_implementation=args.attention)
        frozen_json(folder/'runtime.json',runtime_metadata(model))
        frozen_json(folder/'prompt_template_check.json',wmac_experiments.check_prompt_template(model,questions[ids[0]]))
        frozen_json(folder/'rope_shift_check.json',wmac_experiments.check_rope_shift(model))
        frozen_json(folder/'preflight.json',wmac_experiments.preflight(model,questions,ids,cfg['max_context_tokens']))
        def evaluator(where):
            def evaluate(q,name,options):
                if wmac_plan.is_validation(args.job):
                    return paper_experiments.endpoint(model,q,options['variant'],options['answer_tokens'],where)
                return wmac_experiments.generate(model,q,options['condition'],options['thinking'],options['think_budget'],options['answer_tokens'])
            return evaluate
        if not execute_phase(args.job,smoke,specs,questions,smoke_store,evaluator(folder/'smoke'),lambda:stopped[0],deadline,commit()):
            return 2
        check_smoke(smoke_store,specs)
        frozen_json(folder/'smoke_passed.json',dict(fingerprint=fingerprint,rows=len(smoke_store.rows)))
        if args.smoke_only:
            status='smoke_passed';return 0
        if probe_ids and not execute_phase('shortcut probes',probe_ids,probe_specs,questions,probes,
                lambda q,name,opts:wmac_experiments.probe(model,q,name,opts['answer_tokens']),lambda:stopped[0],deadline,commit()):
            return 2
        if not execute_phase(args.job,ids,specs,questions,main_store,evaluator(folder/'main'),lambda:stopped[0],deadline,commit()):
            return 2
        if wmac_plan.is_validation(args.job) and not numerical_diagnostics(model,ids,questions,main_store,diagnostics,folder,lambda:stopped[0],deadline):
            return 2
        status='complete';return 0
    except BaseException as error:
        status='failed';atomic_json(folder/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        for sig,handler in handlers.items(): signal.signal(sig,handler)
        try: summarize()
        except Exception as error:
            atomic_json(folder/'summary_failure.json',dict(type=type(error).__name__,message=str(error)))
            if status=='complete': status='analysis_failed'
        atomic_json(folder/'status.json',dict(status=status,saved_main_rows=len(main_store.rows),expected_main_rows=len(ids)*len(specs),
            saved_probe_rows=len(probes.rows),expected_probe_rows=len(probe_ids)*len(probe_specs),
            saved_diagnostic_pairs=len(diagnostics.rows),seconds=time.monotonic()-start))
        archive=shutil.make_archive(str(folder),'gztar',root_dir=folder.parent,base_dir=folder.name)
        print(f'{status}: {archive}',flush=True)
        if status=='analysis_failed':
            raise RuntimeError('inference finished but summary failed; inspect summary_failure.json and rerun --summarize')


if __name__=='__main__':
    raise SystemExit(main())
