"""Descriptive paired transitions, cap sensitivity, and illustrative raw answers."""
import argparse,collections,json,pathlib,sys,tarfile
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]))
from latent_dispatch.score import score
from latent_dispatch.stats import bootstrap_ci
p=argparse.ArgumentParser();p.add_argument('archive',type=pathlib.Path);args=p.parse_args()
with tarfile.open(args.archive) as t:
    bundle=json.load(t.extractfile('wmac_gpu_v3_l4/prepared.json'));q=bundle['questions']
    rows=[json.loads(x) for x in t.extractfile('wmac_gpu_v3_l4/musique-4b/main/rows.jsonl') if x.strip()]
by={(r['id'],r['condition']):r for r in rows}
scores={k:score(r['text'],q[r['id']]['answers']) for k,r in by.items()}
ids=bundle['musique_ids'];caps={q[r['id']]['source_id'] for r in rows if r['cut']}
extra={};examples=[]
for a,b in [('gold_off','blocked_off'),('gold_off','matched_irrelevant_off'),('gold_off','full_off'),('b1_off','k0_off'),('b1_off','full_off')]:
    gain=[id for id in ids if scores[id,a][1] and not scores[id,b][1]]
    loss=[id for id in ids if scores[id,b][1] and not scores[id,a][1]]
    extra[a+'-'+b]=dict(em_fixed_layouts=len(gain),em_broken_layouts=len(loss),
        identical_generated_text=sum(by[id,a]['text']==by[id,b]['text'] for id in ids),
        identical_em_outcome=sum(scores[id,a][1]==scores[id,b][1] for id in ids),
        identical_f1_score=int(sum(np.isclose(scores[id,a][2],scores[id,b][2]) for id in ids)),
        f1_difference_excluding_any_capped_source=float(np.mean([scores[id,a][2]-scores[id,b][2] for id in ids if q[id]['source_id'] not in caps])))
    for id in gain[:2]:
        examples.append(dict(comparison=a+' vs '+b,id=id,question=q[id]['question'],accepted_answers=q[id]['answers'],
            answers={c:by[id,c]['text'] for c in (a,b)},
            support_paragraphs=[q[id]['workers'][g['worker']][g['index']] for g in q[id]['gold']]))
groups=collections.defaultdict(dict)
for id in ids:groups[q[id]['source_id']][q[id]['placement']]=id
order=[]
for layouts in groups.values():
    ab,ba=layouts['AB'],layouts['BA']
    order.append((scores[ba,'b1_off'][2]-scores[ba,'k0_off'][2])-(scores[ab,'b1_off'][2]-scores[ab,'k0_off'][2]))
mean,lo,hi=bootstrap_ci(order,n_boot=20000,seed=bundle['config']['seed'])
output=dict(capped_sources=sorted(caps),capped_sources_count=len(caps),uncapped_sources=200-len(caps),
    paired_transitions=extra,exploratory_gap_order_difference_BA_minus_AB=dict(mean=mean,ci95=[lo,hi],unit='paired source',adjusted=False))
(HERE/'supplemental.json').write_text(json.dumps(output,indent=2)+'\n')
(HERE/'illustrative-answers.json').write_text(json.dumps(examples,indent=2)+'\n')
print(json.dumps(output,indent=2))
