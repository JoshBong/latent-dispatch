#!/usr/bin/env python3
"""Two explicit GPU jobs: T4 (4B FP16) and A100 (8B BF16). Nothing runs on import."""
import argparse
import fcntl
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from latent_dispatch import wmac_plan
from latent_dispatch.paper_plan import atomic_json,file_sha,frozen_json


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('gpu',choices=['t4','a100'])
    p.add_argument('--prepared',type=Path,help='The SAME frozen v3 prepared.json on both machines')
    p.add_argument('--output-root',type=Path,default=ROOT/'results')
    p.add_argument('--run-name',default='wmac_gpu_v3')
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-hours',type=float,default=24,help='Soft time limit for the whole GPU job, excluding final packaging')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--smoke-only',action='store_true')
    mode.add_argument('--plan-only',action='store_true')
    mode.add_argument('--summarize',action='store_true')
    args=p.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name): p.error('invalid run name')
    if not 0<args.max_hours<float('inf'): p.error('--max-hours must be finite and positive')
    args.output_root=args.output_root.resolve()
    args.prepared=(args.prepared or args.output_root/args.run_name/'prepared.json').resolve()
    return args


def commands(args,cfg):
    """Each component gets a separate process, releasing GPU weights before the next."""
    for job in cfg['gpu_jobs'][args.gpu]['jobs']:
        cmd=[sys.executable,str(ROOT/'scripts/run_wmac_a100.py'),job,
             '--prepared',str(args.prepared),'--output-root',str(args.output_root),
             '--run-name',args.run_name,'--device',args.device]
        for flag in ('smoke_only','plan_only','summarize'):
            if getattr(args,flag): cmd.append('--'+flag.replace('_','-'))
        yield job,cmd


def archive(args,jobs):
    """Only this GPU's components; the other machine can still be running."""
    root=args.output_root/args.run_name
    destination=args.output_root/f'{args.run_name}-{args.gpu}.tar.gz'
    temporary=destination.with_suffix('.tmp')
    paths=[root/f'gpu-{args.gpu}',*[root/job for job in jobs]]
    with tarfile.open(temporary,'w:gz') as tar:
        tar.add(root/f'gpu-{args.gpu}'/'prepared.json',arcname=f'{args.run_name}/prepared.json')
        for path in paths:
            if path.exists(): tar.add(path,arcname=str(path.relative_to(args.output_root)))
    temporary.replace(destination)
    return destination


def run(args,bundle):
    cfg=bundle['config'];jobs=cfg['gpu_jobs'][args.gpu]['jobs']
    root=args.output_root/args.run_name;meta=root/f'gpu-{args.gpu}'
    meta.mkdir(parents=True,exist_ok=True)
    with (meta/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen_json(meta/'prepared.json',bundle)
        frozen_json(meta/'manifest.json',dict(protocol=cfg['protocol'],gpu=args.gpu,
            profile=cfg['gpu_jobs'][args.gpu],prepared_sha256=file_sha(args.prepared),
            launcher_sha256=file_sha(__file__),component_runner_sha256=file_sha(ROOT/'scripts/run_wmac_a100.py')))
        start=time.monotonic();deadline=start+args.max_hours*3600
        active=None;stopped=False;exit_code=0;completed=[];status='incomplete'
        def stop(signum,frame):
            nonlocal stopped
            stopped=True
            if active is not None and active.poll() is None: active.send_signal(signum)
        handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            for job,cmd in commands(args,cfg):
                remaining=(deadline-time.monotonic())/3600
                if stopped or remaining<=0:
                    exit_code=2;break
                cmd+=['--max-hours',str(remaining)]
                print(f'[{args.gpu}] {shlex.join(cmd)}',flush=True)
                active=subprocess.Popen(cmd,cwd=ROOT)
                code=active.wait();active=None
                if not code: completed.append(job)
                atomic_json(meta/'progress.json',dict(completed=completed,last_component=job,last_exit_code=code))
                if code:
                    exit_code=code if code>0 else 128-code
                    if not (args.summarize and code==2): break
            if exit_code==0 and not stopped:
                status='planned' if args.plan_only else 'smoke_passed' if args.smoke_only else 'complete'
            elif exit_code not in (0,2): status='failed'
            if stopped and exit_code==0: exit_code=2
            return exit_code
        except BaseException as error:
            status='failed'
            atomic_json(meta/'failure.json',dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            for sig,handler in handlers.items(): signal.signal(sig,handler)
            atomic_json(meta/'status.json',dict(status=status,completed=completed,expected=jobs,
                        seconds=time.monotonic()-start))
            if not args.plan_only:
                print(f'[{args.gpu}] {status}: {archive(args,jobs)}',flush=True)


def main(argv=None):
    args=arguments(argv)
    if not args.prepared.is_file():
        raise FileNotFoundError(f'{args.prepared}: prepare ONCE, then copy that same bundle to both machines')
    bundle,_=wmac_plan.load(args.prepared)
    return run(args,bundle)


if __name__=='__main__':
    raise SystemExit(main())
