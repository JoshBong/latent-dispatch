"""Audit archived WMAC rows and reproduce source-paired scores without GPU inference."""
import argparse,collections,csv,hashlib,io,json,pathlib,sys,tarfile,tempfile
from types import SimpleNamespace
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from latent_dispatch.paper_plan import digest
from latent_dispatch.score import score
from latent_dispatch import wmac_plan,wmac_summary

parser=argparse.ArgumentParser();parser.add_argument('--archive',type=pathlib.Path)
args=parser.parse_args()
with tempfile.TemporaryDirectory() as temp:
    archive=args.archive or HERE/'raw-results.tar.gz'
    if not archive.exists():
        parts=sorted(HERE.glob('raw-results.part-*'))
        if not parts:raise FileNotFoundError('Supply --archive or restore the raw-results parts')
        archive=pathlib.Path(temp)/'raw-results.tar.gz'
        with archive.open('wb') as out:
            for part in parts:
                with part.open('rb') as f:
                    for block in iter(lambda:f.read(2**20),b''):out.write(block)
    sha=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):sha.update(block)
    expected=json.loads((HERE/'archive-provenance.json').read_text())
    assert sha.hexdigest()==expected['raw_archive_sha256']
    data={};first_logits={};shared_logits={};numeric_files=0
    with tarfile.open(archive,'r|gz') as tar:
        for m in tar:
            if not m.isfile():continue
            if m.name.endswith(('.json','.jsonl','.csv')):data[m.name]=tar.extractfile(m).read()
            elif m.name.endswith('.npz'):
                with np.load(io.BytesIO(tar.extractfile(m).read()),allow_pickle=False) as arrays:
                    assert all(np.isfinite(arrays[k]).all() for k in arrays.files),m.name
                    if 'first_logits' in arrays.files:first_logits[m.name]=arrays['first_logits'].copy()
                    elif 'prefix_positions' in arrays.files:
                        x,y=arrays['reference'],arrays['candidate']
                        shared_logits[m.name]=dict(positions=arrays['prefix_positions'].tolist(),
                            matches=(x.argmax(-1)==y.argmax(-1)).tolist(),
                            max_abs=np.abs(x-y).max(-1).tolist())
                    else:raise AssertionError(m.name)
                numeric_files+=1
    prefix='wmac_gpu_v3_l4/'
    def obj(name):return json.loads(data[prefix+name])
    bundle=obj('prepared.json');questions=bundle['questions'];cfg=bundle['config']
    assert hashlib.sha256(data[prefix+'prepared.json']).hexdigest()=='8abf29425611b1f3b65f5016fd40651ef0281bdb6de9852e1df2940f33953058'
    assert obj('gpu-t4/status.json')['status']=='complete'
    assert all(digest(q)==bundle['question_sha256'][id] for id,q in questions.items())
    stores={};phase_audit={};manifests={}
    for job in ('validation-4b','musique-4b'):
        manifests[job]=manifest=obj(job+'/manifest.json')
        assert manifest['dtype']=='float16' and manifest['model']=='Qwen/Qwen3-4B'
        assert manifest['revision_requested']==cfg['models']['4b']['revision']
        assert obj(job+'/status.json')['status']=='complete'
        runtime=obj(job+'/runtime.json');assert 'L4' in runtime['device_name']
        for file,h in manifest['code_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
        for phase in ('main','smoke','probes','diagnostics'):
            name=prefix+job+'/'+phase+'/rows.jsonl'
            if name not in data:continue
            raw=data[name];assert raw.endswith(b'\n')
            rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
            keyed={(r['id'],r['condition']):r for r in rows};assert len(keyed)==len(rows)
            assert all(r['fingerprint']==digest(manifest) and r['question_sha256']==digest(questions[r['id']]) for r in rows)
            if phase in ('main','smoke'):
                assert all(r['options']==manifest['specifications'][r['condition']] for r in rows)
                assert all(r['think_tokens']==0 for r in rows)
            phase_audit[job+'/'+phase]=dict(rows=len(rows),conditions=dict(collections.Counter(r['condition'] for r in rows)),
                answer_cuts=sum(r.get('cut',False) for r in rows),
                missing_boxes=sum('\\boxed{' not in r.get('text','') for r in rows) if phase!='diagnostics' else None)
            stores[job,phase]=SimpleNamespace(rows=keyed)
        ids=wmac_plan.job_ids(job,bundle)
        assert set(stores[job,'main'].rows)=={(id,c) for id in ids for c in manifest['specifications']}
    expected_counts={'validation-4b/main':96,'validation-4b/smoke':16,'validation-4b/diagnostics':48,
                     'musique-4b/main':2400,'musique-4b/smoke':12,'musique-4b/probes':600}
    assert {k:v['rows'] for k,v in phase_audit.items()}==expected_counts
    ids=bundle['musique_ids'];specs=wmac_plan.specifications('musique-4b',cfg)
    probes=stores['musique-4b','probes'];main=stores['musique-4b','main']
    ab=[id for id in ids if questions[id]['placement']=='AB']
    assert set(probes.rows)=={(id,c) for id in ab for c in ('closed_book','keep_hop0','keep_hop1')}
    assert all(r['options']=={'thinking':'off','answer_tokens':cfg['probe_answer_tokens']} for r in probes.rows.values())
    for id in ids:
        rows=[main.rows[id,c] for c in specs]
        assert len({r['context_tokens_sha256'] for r in rows})==1
        assert all(r['final_answer_position']==bundle['preflight']['4b'][id]['context_tokens'] for r in rows)
        control=main.rows[id,'matched_irrelevant_off']['mask']
        assert control['a_visible_tokens']==control['a_gold_tokens']
        assert control['selected_keys_precede_gold']
    out=HERE/'recomputed';out.mkdir(exist_ok=True)
    summary=wmac_summary.summarize('musique-4b',ids,specs,questions,main,out,cfg,probes)
    saved=obj('musique-4b/main/summary.json')
    def equal(a,b,path=''):
        if isinstance(a,dict):
            assert a.keys()==b.keys(),path
            for k in a:equal(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            assert len(a)==len(b),path
            for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
        elif isinstance(a,float):assert np.isclose(a,b,atol=1e-12,rtol=1e-12),path
        else:assert a==b,path
    for key in ('saved_rows','expected_rows','fully_paired_sources','complete','probes','paired_absolute_scores','contrasts','cells'):
        equal(summary[key],saved[key],key)
    for file in out.glob('*.csv'):file.write_bytes(file.read_bytes().replace(b'\r\n',b'\n'))
    numerical=[]
    vm=stores['validation-4b','main'].rows
    for id in bundle['validation_ids']:
        for reference,candidate in [('b1','masked_full'),('b1_mask','masked_blocked')]:
            a,b=vm[id,reference],vm[id,candidate]
            assert all(a[k]==b[k] for k in ('context_tokens_sha256','answer_tail_sha256','position_ids_sha256','worker_tokens'))
            x=first_logits[prefix+'validation-4b/main/'+a['first_logits_file']].astype(float)
            y=first_logits[prefix+'validation-4b/main/'+b['first_logits_file']].astype(float)
            d=stores['validation-4b','diagnostics'].rows[id,candidate]
            sampled=shared_logits[prefix+'validation-4b/diagnostics/'+d['logits_file']]
            for i,match,delta in zip(sampled['positions'],sampled['matches'],sampled['max_abs'],strict=True):
                assert match==d['steps'][i]['argmax_matches']
                assert np.isclose(delta,d['steps'][i]['max_abs_delta'],atol=1e-7)
            numerical.append(dict(id=id,reference=reference,candidate=candidate,
                first_argmax_equal=bool(x.argmax()==y.argmax()),max_abs=float(np.abs(x-y).max()),
                relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-12)),
                generated_tokens_equal=a['token_ids']==b['token_ids'],
                predictions_equal=score(a['text'],questions[id]['answers'])[0]==score(b['text'],questions[id]['answers'])[0],
                same_prefix_positions=len(d['steps']),same_prefix_argmax_mismatches=sum(not s['argmax_matches'] for s in d['steps'])))
    (out/'numerical-diagnostics.json').write_text(json.dumps(numerical,indent=2)+'\n')
    extra={}
    for c in specs:
        rows=[main.rows[id,c] for id in ids]
        scores=[score(r['text'],questions[r['id']]['answers']) for r in rows]
        extra[c]=dict(rows=len(rows),em=sum(s[1] for s in scores)/len(rows),f1=float(np.mean([s[2] for s in scores])),
            answer_cuts=sum(r['cut'] for r in rows),missing_boxes=sum('\\boxed{' not in r['text'] for r in rows),
            mean_seconds=float(np.mean([r['seconds'] for r in rows])),
            mean_answer_tokens=float(np.mean([r['answer_tokens'] for r in rows])),
            partial_f1_but_not_em=sum(s[2]>0 and not s[1] for s in scores))
    supports=collections.defaultdict(set)
    for id in ab:
        q=questions[id]
        for g in q['gold']:
            text=' '.join(q['workers'][g['worker']][g['index']].lower().split())
            supports[text].add(q['source_id'])
    duplicated=[sorted(v) for v in supports.values() if len(v)>1]
    audit=dict(verified=True,generated_answers=sum(v['rows'] for k,v in phase_audit.items() if not k.endswith('/diagnostics')),
        numerical_pairs=48,numeric_files_verified=numeric_files,phases=phase_audit,conditions=extra,
        prepared_sha256=hashlib.sha256(data[prefix+'prepared.json']).hexdigest(),
        runtime_commits=sorted({r['commit'] for r in main.rows.values()}),
        sources=summary['fully_paired_sources'],strict_sources=summary['probes']['strict_paired_sources'],
        duplicated_support_paragraph_groups=duplicated,human_review_records=len(bundle.get('reviews',{})),
        numerical_summary={c:dict(pairs=sum(n['candidate']==c for n in numerical),
            first_argmax_matches=sum(n['first_argmax_equal'] for n in numerical if n['candidate']==c),
            same_generated_tokens=sum(n['generated_tokens_equal'] for n in numerical if n['candidate']==c),
            same_prediction=sum(n['predictions_equal'] for n in numerical if n['candidate']==c),
            same_prefix_argmax_mismatches=sum(n['same_prefix_argmax_mismatches'] for n in numerical if n['candidate']==c),
            same_prefix_positions=sum(n['same_prefix_positions'] for n in numerical if n['candidate']==c),
            maximum_first_relative_l2=max(n['relative_l2'] for n in numerical if n['candidate']==c)) for c in ('masked_full','masked_blocked')})
    (HERE/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
