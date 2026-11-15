"""Run only A-side masking on the existing pilot scope, with its necessary composition probes."""
import hashlib
import json
import os
import pathlib
import subprocess
import tarfile
import time

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import papermill as pm

from latent_dispatch import arms, data, results
from latent_dispatch.model import Model

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
MODEL, DTYPE, SEEDS, N_PER_CELL = 'Qwen/Qwen3-8B', 'float16', [0], 100
qs = data.per_cell(data.load('toy', seed=SEEDS, qtype='hop1', placement='AB')
                   + data.load('musique', seed=SEEDS, qtype='2hop', placement='AB'), N_PER_CELL)
assert len(qs) == 200 and len({(q['dataset'], q['source_id']) for q in qs}) == 200
assert all(len(q['gold']) == 2 for q in qs)
PROBES = {'b0': (arms.b0, {}),
          'b1_only_hop1': (arms.b1, {'keep_gold_hops': [0]}),
          'b1_only_hop2': (arms.b1, {'keep_gold_hops': [1]})}
folder = results.results_path(MODEL, DTYPE, 'x').parent
folder.mkdir(parents=True, exist_ok=True)
plan = []
for name, (arm, options) in PROBES.items():
    path = results.results_path(MODEL, DTYPE, name)
    saved = results._read(path)
    results._check(path, saved, {'arm': arm.__name__, 'options': options})
    present = sum(q['id'] in saved for q in qs)
    plan.append({'arm': name, 'total': len(qs), 'saved': present, 'remaining': len(qs)-present})
manifest = {'scope': 'masking_only_pilot', 'model': MODEL, 'dtype': DTYPE,
            'seeds': SEEDS, 'n_per_cell': N_PER_CELL, 'commit': results._commit(),
            'source_questions': len(qs), 'probes': plan,
            'script_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
(folder / 'masking-only-launch.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2), flush=True)
if any(item['remaining'] for item in plan):
    model = Model.load(MODEL, dtype=DTYPE)
    for name, (arm, options) in PROBES.items():
        results.run(model, qs, name, arm, **options)
    del model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

df = results.load_results(qs, PROBES, MODEL, DTYPE)
labels = {}
for q in qs:
    found = df[df.id == q['id']]
    assert len(found) == len(PROBES), f"Missing composition probe for {q['id']}"
    labels[f"{q['dataset']}:{q['source_id']}"] = bool((found.em == 0).all())
label_file = folder / 'part0_requires_composition.json'
if label_file.exists():
    backup = folder / f'part0_requires_composition-before-masking-only-{int(time.time())}.json'
    backup.write_bytes(label_file.read_bytes())
label_file.write_text(json.dumps({'seeds': SEEDS, 'smoke': False, 'scope': 'masking_only_pilot',
                                  'n_per_cell': N_PER_CELL, 'labels': labels}, indent=2)+'\n')
print(f'Composition labels ready: {sum(labels.values())}/{len(labels)} eligible sources.', flush=True)
output = ROOT / 'notebooks/runs/Qwen3-8B-float16-h1_masking-only.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)
pm.execute_notebook(str(ROOT / 'notebooks/h1_masking.ipynb'), str(output), kernel_name='python3',
                    cwd=str(ROOT), log_output=True,
                    parameters={'MODEL': MODEL, 'DTYPE': DTYPE, 'SMOKE': False,
                                'SEEDS': SEEDS, 'N_PER_CELL': N_PER_CELL, 'SKIP': []})
bundle = ROOT / 'results/Qwen3-8B-float16-masking-only.tar.gz'
with tarfile.open(bundle, 'w:gz') as tar:
    tar.add(folder, arcname=str(folder.relative_to(ROOT)))
    tar.add(output, arcname=str(output.relative_to(ROOT)))
    tar.add(__file__, arcname='run-masking-only.py')

print(f'Masking-only experiment completed. Bundle: {bundle}', flush=True)
