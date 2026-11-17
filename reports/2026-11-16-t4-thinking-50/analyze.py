"""Validate and summarize the completed 50-per-cell pilot from its archived raw rows."""
import collections,csv,hashlib,json,pathlib,sys,tarfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from latent_dispatch.score import score
HERE=pathlib.Path(__file__).resolve().parent
archive=HERE/'results-and-logs.tar.gz'
with tarfile.open(archive) as tar:
    members={m.name:m for m in tar.getmembers() if m.isfile()}
    prefix='latent-dispatch-thinking/results/Qwen3-4B-float16/'
    manifest=json.load(tar.extractfile(members[prefix+'thinking-pilot-launch.json']))
    status=tar.extractfile(members['thinking-experiment.status']).read().decode()
    assert 'exit_code=0' in status,status
    assert manifest['per_cell']==50 and manifest['count']==350
    ids=set(manifest['question_ids']);assert len(ids)==350
    questions={q['id']:q for q in json.loads((HERE/'question-metadata.json').read_text())}
    assert set(questions)==ids
    checksum_name='latent-dispatch-thinking/benchmark/SHA256SUMS'
    if checksum_name in members:
        assert tar.extractfile(members[checksum_name]).read()==(ROOT/'benchmark/SHA256SUMS').read_bytes()
    baseline=json.loads((HERE/'resume-baseline.json').read_text())
    for name,record in baseline['files'].items():
        raw=tar.extractfile(members[prefix+name]).read()
        assert hashlib.sha256(raw[:record['bytes']]).hexdigest()==record['sha256'],name
    byarm={};audits={};scored=[]
    for name,opts in manifest['arms'].items():
        raw=tar.extractfile(members[prefix+name+'.jsonl']).read()
        assert raw.endswith(b'\n')
        rows=[json.loads(l) for l in raw.splitlines() if l.strip()]
        assert len(rows)==len({r['id'] for r in rows})
        method=name.removeprefix('thinking_pilot_v2_').rsplit('_',1)[0]
        assert all(r['options']==opts and r['arm']==method and r['model']=='Qwen/Qwen3-4B' and r['dtype']=='float16' for r in rows)
        selected=[r for r in rows if r['id'] in ids];assert len(selected)==350
        byarm[name]={}
        for r in selected:
            q=questions[r['id']];pred,em,f1=score(r['text'],q['answers'])
            if opts['thinking']=='forced': assert '</think>' in r['text']
            row=dict(id=q['id'],source_id=q['source_id'],dataset=q['dataset'],qtype=q['qtype'],placement=q['placement'],arm=name,em=int(em),f1=f1,seconds=r['seconds'],think_tokens=r['think_tokens'],gen_tokens=r['gen_tokens'],cut=r['cut'],think_cut=r.get('think_cut',False),think_eos=r.get('think_eos',False),boxed='\\boxed{' in r['text'],report_cuts=sum(r.get('truncated',[])),pred=pred)
            byarm[name][r['id']]=row;scored.append(row)
        audits[name]=dict(selected=len(selected),all_saved=len(rows),outside_sample=len(rows)-350,answer_cuts=sum(r['cut'] for r in selected),unboxed=sum('\\boxed{' not in r['text'] for r in selected),thinking_cap=sum(r.get('think_cut',False) for r in selected),mean_think_tokens=float(np.mean([r['think_tokens'] for r in selected])),mean_seconds=float(np.mean([r['seconds'] for r in selected])),report_cuts=sum(sum(r.get('truncated',[])) for r in selected),think_eos=sum(r.get('think_eos',False) for r in selected))
    assert len(scored)==2100
    (HERE/'thinking-experiment.status').write_text(status)
    (HERE/'thinking-pilot-launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/'thinking-experiment.log').write_bytes(tar.extractfile(members['thinking-experiment.log']).read())
with (HERE/'scored-responses.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(scored[0]));w.writeheader();w.writerows(scored)
groups={
 'Toy two-fact AB+BA':[q for q in questions.values() if q['dataset']=='toy_d2' and q['qtype']=='hop1'],
 'Toy three-fact AB+BA':[q for q in questions.values() if q['dataset']=='toy_d2' and q['qtype']=='hop2'],
 'MuSiQue 2-hop AB+BA':[q for q in questions.values() if q['dataset']=='musique' and q['placement'] in ('AB','BA')],
 'MuSiQue 2-hop BB control':[q for q in questions.values() if q['dataset']=='musique' and q['placement']=='BB'],
}
rows=[];pairs=[];rng=np.random.default_rng(20261116)
for group,qs in groups.items():
    row={'group':group,'layouts':len(qs),'sources':len({q['source_id'] for q in qs})}
    for method in ('b1','k0','t1'):
        for mode in ('off','forced'):
            row[method+'_'+mode]=round(100*np.mean([byarm['thinking_pilot_v2_'+method+'_'+mode][q['id']]['em'] for q in qs]),2)
    rows.append(row)
    # Paired bootstrap over source questions keeps AB/BA of the same source together.
    for label,terms in {
        'B1 thinking gain':[(1,'b1_forced'),(-1,'b1_off')],
        'K0 thinking gain':[(1,'k0_forced'),(-1,'k0_off')],
        'T1 thinking gain':[(1,'t1_forced'),(-1,'t1_off')],
        'B1-K0 off':[(1,'b1_off'),(-1,'k0_off')],
        'B1-K0 on':[(1,'b1_forced'),(-1,'k0_forced')],
        'gap closure with thinking':[(1,'b1_off'),(-1,'k0_off'),(-1,'b1_forced'),(1,'k0_forced')],
    }.items():
        clustered=collections.defaultdict(list)
        for q in qs: clustered[q['source_id']].append(sum(weight*byarm['thinking_pilot_v2_'+arm][q['id']]['em'] for weight,arm in terms))
        a=np.array([np.mean(v) for v in clustered.values()]);boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)*100
        lo,hi=np.quantile(boot,[.025,.975]);pairs.append(dict(group=group,comparison=label,points=round(float(a.mean()*100),2),ci95=[round(float(lo),2),round(float(hi),2)]))
summary=dict(cache_prefixes_verified=len(baseline['files']),status=status,model=manifest['model'],dtype=manifest['dtype'],per_cell=50,total_selected=2100,accuracy=rows,audits=audits,paired_bootstrap=pairs)
(HERE/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
