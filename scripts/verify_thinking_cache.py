"""Validate saved cache against the completed 25-per-cell archive before continuation."""
import argparse,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--allow-appends',action='store_true');args=p.parse_args()
baseline=json.loads((root/'reports/2026-11-16-t4-thinking-50/resume-baseline.json').read_text())
folder=root/'results/Qwen3-4B-float16'
for name,info in baseline['files'].items():
    raw=(folder/name).read_bytes()
    compared=raw[:info['bytes']] if args.allow_appends else raw
    assert hashlib.sha256(compared).hexdigest()==info['sha256'],f'Saved bytes changed: {name}'
    assert not raw or raw.endswith(b'\n'),f'Incomplete row: {name}'
    rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
    assert len(rows)==len({r['id'] for r in rows}),f'Duplicate IDs: {name}'
for c in baseline['conditions']:
    rows=[json.loads(l) for l in (folder/(c['name']+'.jsonl')).read_text().splitlines() if l.strip()]
    assert all(r['options']==c['options'] and r['model']==baseline['model'] and r['dtype']==baseline['dtype'] for r in rows),c['name']
    print(f"{c['name']}: {c['saved']} baseline cached, {c['new']} initially missing",flush=True)
print('Saved cache verified; original bytes preserved.',flush=True)
