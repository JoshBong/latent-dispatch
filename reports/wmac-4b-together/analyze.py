"""Audit the 4B same-worker control and compare its paired gap with the main run."""
import collections,hashlib,importlib.util,json,pathlib,sys,tarfile
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from latent_dispatch.score import score
from latent_dispatch.stats import bootstrap_ci
s=importlib.util.spec_from_file_location('together',HERE/'wmac_together.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
archive=HERE/'results-and-logs.tar.gz'
with tarfile.open(archive) as t:
    read=lambda n:json.load(t.extractfile('wmac-4b-together/4b/together/'+n))
    plan,status=read('plan.json'),read('status.json')
    rows=[json.loads(l) for l in t.extractfile('wmac-4b-together/4b/together/rows.jsonl') if l.strip()]
assert status['complete'] and status['saved_rows']==status['expected_rows']==800
assert status['layouts']==400 and status['dropped']==0 and not plan['dropped']
assert status['runtime']['dtype']==plan['dtype']=='float16' and 'L4' in status['runtime']['device_name']
assert plan['conditions']==['blocked','full']
mainfile=ROOT/'archive/wmac_gpu_v3/l4-4b-no-logits-2026-11-18.tar.gz'
with tarfile.open(mainfile) as t:
    prefix='wmac_gpu_v3_l4/'
    raw=t.extractfile(prefix+'prepared.json').read();assert hashlib.sha256(raw).hexdigest()==m.PREPARED_SHA256
    bundle=json.loads(raw)
    main=[json.loads(l) for l in t.extractfile(prefix+'musique-4b/main/rows.jsonl') if l.strip()]
assert len(main)==2400
assert status['runtime']['revision']==bundle['config']['models']['4b']['revision']
questions={}
for id in m.base_ids(bundle):
    for p in ('AA','BB'):
        q=m.together(bundle['questions'][id],p);questions[q['id']]=q
assert plan['question_sha256']=={id:m.digest(q) for id,q in questions.items()}
control={(r['id'],r['condition']):r for r in rows};assert len(control)==len(rows)==800
assert set(control)=={(id,c+'_off') for id in questions for c in ('blocked','full')}
for (id,c),r in control.items():
    assert r['key']==id+'|'+c.removesuffix('_off') and r['question_sha256']==m.digest(questions[id])
    assert r['options']==dict(condition=c.removesuffix('_off'),thinking='off',answer_tokens=1024,think_budget=1024)
    assert r['think_tokens']==0 and r['source_id']==questions[id]['source_id']
mainrows={(r['id'],r['condition']):r for r in main}
cs={k:score(r['text'],questions[k[0]]['answers']) for k,r in control.items()}
ms={k:score(r['text'],bundle['questions'][k[0]]['answers']) for k,r in mainrows.items()}
cm=collections.defaultdict(dict);mm=collections.defaultdict(dict)
for id,q in questions.items():cm[q['source_id']][q['placement']]=id
for id in bundle['musique_ids']:
    q=bundle['questions'][id];mm[q['source_id']][q['placement']]=id
assert cm.keys()==mm.keys() and len(cm)==200
sources=sorted(cm)
def estimate(values):
    mean,lo,hi=bootstrap_ci(values,n_boot=20000,seed=bundle['config']['seed'])
    return dict(mean=mean,ci95=[lo,hi],sources=len(values))
contrasts=[]
for metric,index in [('f1',2),('em',1)]:
    gaps={}
    for label,orders,data,scores in [('control_AA_BB',('AA','BB'),cm,cs),('control_AA',('AA',),cm,cs),
        ('control_BB',('BB',),cm,cs),('main_AB_BA',('AB','BA'),mm,ms)]:
        values=[float(np.mean([float(scores[data[s][p],'full_off'][index])-float(scores[data[s][p],'blocked_off'][index]) for p in orders])) for s in sources]
        gaps[label]=values;contrasts.append(dict(contrast=label+'_full_minus_blocked',metric=metric,**estimate(values)))
    contrasts.append(dict(contrast='split_gap_minus_together_gap',metric=metric,
        **estimate(np.asarray(gaps['main_AB_BA'])-np.asarray(gaps['control_AA_BB']))))
cells=[]
for place in ('AA','BB','AA+BB'):
    for cond in ('blocked_off','full_off'):
        chosen=[r for r in rows if r['condition']==cond and (place=='AA+BB' or r['placement']==place)]
        vals=[cs[r['id'],cond] for r in chosen]
        cells.append(dict(placement=place,condition=cond,rows=len(chosen),em=float(np.mean([v[1] for v in vals])),f1=float(np.mean([v[2] for v in vals])),
            answer_cuts=sum(r['cut'] for r in chosen),missing_boxes=sum('\\boxed{' not in r['text'] for r in chosen),mean_seconds=float(np.mean([r['seconds'] for r in chosen])) ))
output=dict(verified=True,sources=200,layouts=400,main_answers=800,runtime=status['runtime'],cells=cells,contrasts=contrasts,
 statistics=dict(unit='source question, averaging AA/BB or AB/BA within source',resamples=20000,seed=bundle['config']['seed'],ci='95% unadjusted paired percentile',
 caveat='A confidence interval covering zero does not prove equivalence; layout interaction is not a unique mechanism test. Fixed accepted answers retain known ambiguity.'))
(HERE/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
lines=['# Completed Qwen3-4B same-worker control','', 'Verified all 800 control answers on the same 200 sources, FP16 on L4.','',
 '| Layout | Condition | F1 (0–100) | EM (%) |','|---|---|---:|---:|']
for c in cells:lines.append(f"| {c['placement']} | {c['condition']} | {100*c['f1']:.2f} | {100*c['em']:.2f} |")
lines+=['','Paired F1 differences (percentage points); 95% unadjusted source-bootstrap intervals:','']
for c in contrasts:
    if c['metric']=='f1':lines.append(f"- {c['contrast']}: {100*c['mean']:+.2f} [{100*c['ci95'][0]:+.2f}, {100*c['ci95'][1]:+.2f}]")
lines+=['','The interaction compares split and same-worker gaps on matched sources. A near-zero control estimate is not by itself an equivalence claim. Annotation ambiguity remains; no answers were manually changed.']
(HERE/'FINAL.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(output,indent=2))
