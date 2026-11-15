"""350-question T4 pilot: B1/K0/T1, thinking off/1024; resumes per arm."""
import collections
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_dispatch import arms, data, results
from latent_dispatch.model import Model
from latent_dispatch.score import score

MODEL, DTYPE = 'Qwen/Qwen3-4B', 'float16'
qs = data.per_cell(
    data.load('toy', seed=0, qtype=['hop1', 'hop2'], placement=['AB', 'BA'])
    + data.load('musique', seed=0, qtype='2hop', placement=['AB', 'BA', 'BB']), 50)
assert len(qs) == len({q['id'] for q in qs}) == 350
folder = results.results_path(MODEL, DTYPE, 'x').parent
folder.mkdir(parents=True, exist_ok=True)
# Separate result names prevent mixing with older thinking implementations.
specs = {}
for thinking in ('forced', 'off'):
    for name in ('b1', 'k0', 't1'):
        options = dict(thinking=thinking, think_budget=1024, max_answer_tokens=256)
        if name == 't1':
            options['report_tokens'] = 1024
        specs[f'thinking_pilot_{name}_{thinking}'] = (getattr(arms, name), options)
manifest = dict(model=MODEL, dtype=DTYPE, commit=results._commit(), seed=0,
                per_cell=50, count=len(qs), question_ids=[q['id'] for q in qs],
                cells=dict(collections.Counter(f"{q['dataset']}/{q['qtype']}/{q['placement']}" for q in qs)),
                arms={n:o for n,(_,o) in specs.items()},
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(folder / 'thinking-pilot-launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2), flush=True)

def pack():
    bundle = ROOT / 'results-Qwen3-4B-float16-thinking.tar.gz'
    temporary = bundle.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as tar:
        for p in sorted(folder.glob('thinking*')):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
        tar.add(__file__, arcname='scripts/run_thinking_pilot.py')
    temporary.replace(bundle)
    print(f'Checkpoint bundle: {bundle}', flush=True)

try:
    model = Model.load(MODEL, dtype=DTYPE)
    smoke = [next(q for q in qs if q['dataset']=='toy'),
             next(q for q in qs if q['dataset']=='musique')]
    for name, (arm, opts) in specs.items():
        smoke_name = name + '_smoke'
        results.run(model, smoke, smoke_name, arm, **opts)
        saved = results._read(results.results_path(MODEL, DTYPE, smoke_name))
        for q in smoke:
            r = saved[q['id']]
            print('SMOKE', name, q['id'], json.dumps(r), flush=True)
            assert not r['cut'], f'Answer cap hit: {name}, inspect before full run'
            assert '\\boxed{' in r['text'], f'Missing boxed answer: {name}'
            if opts['thinking']=='forced':
                assert '</think>' in r['text'], f'Unclosed thinking: {name}'
                assert '\\boxed{' in r['text'].split('</think>')[-1], f'No final answer: {name}'
    print('SMOKE PASSED: all six arms, toy and MuSiQue.', flush=True)
    pack()
    for name, (arm, opts) in specs.items():
        started = time.monotonic()
        results.run(model, qs, name, arm, **opts)
        rows = list(results._read(results.results_path(MODEL, DTYPE, name)).values())
        mine = [r for r in rows if r['id'] in set(manifest['question_ids'])]
        print('ARM_DONE', name, 'rows', len(mine), 'seconds', time.monotonic()-started,
              'mean_think_tokens', sum(r['think_tokens'] for r in mine)/len(mine),
              'answer_cuts', sum(r['cut'] for r in mine), flush=True)
        pack()
    df = results.load_results(qs, specs, MODEL, DTYPE)
    df.groupby(['arm','dataset','qtype','placement'])[['em','think_tokens','gen_tokens','seconds','cut']].mean().to_csv(folder/'thinking-pilot-summary.csv')
    assert len(df)==2100
    print('COMPLETED: 2100 main rows.', flush=True)
finally:
    pack()
