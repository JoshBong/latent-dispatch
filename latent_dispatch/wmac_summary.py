"""Source-paired summaries from saved answers, without loading model weights."""
from collections import defaultdict
import csv

import numpy as np

from . import stats
from .paper_plan import atomic_json
from .score import score


def estimate(values,n_boot,seed):
    if not values: return dict(sources=0,mean=None,ci95=None)
    mean,lo,hi=stats.bootstrap_ci(values,n_boot=n_boot,seed=seed)
    return dict(sources=len(values),mean=mean,ci95=[lo,hi])


def scored(store,questions):
    rows={}
    for key,row in store.rows.items():
        q=questions[row['id']]
        pred,em,f1=score(row['text'],q['answers'])
        rows[key]={**row,'prediction':pred,'em':float(em),'f1':f1,'source_id':q['source_id'],
                   'placement':q['placement'],'cities':q.get('cities',''),'headings':q.get('headings','')}
    return rows


def summarize(job,ids,specs,questions,store,folder,cfg,probe_store=None):
    rows=scored(store,questions)
    groups=defaultdict(list)
    for id in ids: groups[questions[id]['source_id']].append(id)
    paired={s:g for s,g in groups.items() if all((id,c) in rows for id in g for c in specs)}
    cells=defaultdict(list)
    for row in rows.values():
        cells[row['condition'],row['placement'],row['cities'],row['headings']].append(row)
    metrics=('em','f1','seconds','think_tokens','answer_tokens','gen_tokens','cut','think_cut','think_eos')
    table=[]
    for (condition,order,cities,headings),cell in sorted(cells.items()):
        table.append(dict(condition=condition,placement=order,cities=cities,headings=headings,
                          rows=len(cell),sources=len({r['source_id'] for r in cell}),
                          **{m:float(np.mean([r[m] for r in cell])) for m in metrics}))
    result=dict(job=job,saved_rows=len(rows),expected_rows=len(ids)*len(specs),
                main_complete=len(rows)==len(ids)*len(specs),complete=len(rows)==len(ids)*len(specs),
                fully_paired_sources=len(paired),cells=table,contrasts=[],
                statistics=dict(unit='source, averaging orders within source',seed=cfg['seed'],
                    bootstrap_resamples=cfg['bootstrap_resamples'],ci='unadjusted paired percentile',
                    caution='Exploratory slices; no family-wise significance claim. Cap subsets are not causal.'))
    def contrast(name,metric,values,**labels):
        result['contrasts'].append(dict(contrast=name,metric=metric,**labels,
            **estimate(values,cfg['bootstrap_resamples'],cfg['seed'])))
    result['paired_absolute_scores']=[dict(condition=c,metric=m,
        **estimate([float(np.mean([rows[id,c][m] for id in group])) for group in paired.values()],
                   cfg['bootstrap_resamples'],cfg['seed'])) for c in specs for m in ('f1','em')]
    if job.startswith('musique'):
        probes=scored(probe_store,questions) if probe_store is not None else {}
        strict=set()
        for source,group in paired.items():
            ab=next(id for id in group if questions[id]['placement']=='AB')
            keys=[(ab,p) for p in ('closed_book','keep_hop0','keep_hop1')]
            if all(k in probes and probes[k]['f1']<.5 and not probes[k]['cut'] for k in keys): strict.add(source)
        probe_complete=len(probes)==len(groups)*3
        result['complete']=result['complete'] and probe_complete
        result['probes']=dict(saved_rows=len(probes),expected_rows=len(groups)*3,complete=probe_complete,
                              strict_paired_sources=len(strict),definition=cfg['strict_sensitivity'])
        for cohort,selected in [('all',paired),('strict',{s:g for s,g in paired.items() if s in strict})]:
            for order in ('AB+BA','AB','BA'):
                for metric in ('f1','em'):
                    for a,b in [('gold','blocked'),('gold','matched_irrelevant'),('full','blocked'),('gold','full')]:
                        values=[]
                        for group in selected.values():
                            within=[id for id in group if order=='AB+BA' or questions[id]['placement']==order]
                            values.append(float(np.mean([rows[id,a+'_off'][metric]-rows[id,b+'_off'][metric] for id in within])))
                        contrast(a+'-'+b,metric,values,cohort=cohort,placement=order,
                            primary=cohort=='all' and order=='AB+BA' and metric=='f1' and a+'-'+b in cfg['primary_musique_contrasts'])
        bins=defaultdict(list)
        limits=cfg['support_length_bins']
        for group in paired.values():
            n=float(np.mean([rows[id,'gold_off']['mask']['a_gold_tokens'] for id in group]))
            label=next((f'<= {limit}' if i==0 else f'({limits[i-1]}, {limit}]'
                        for i,limit in enumerate(limits) if n<=limit),f'> {limits[-1]}')
            bins[label].append(group)
        for label,grouped in sorted(bins.items()):
            for a,b in [('gold','blocked'),('gold','matched_irrelevant')]:
                values=[float(np.mean([rows[id,a+'_off']['f1']-rows[id,b+'_off']['f1'] for id in group])) for group in grouped]
                contrast(a+'-'+b,'f1',values,cohort='all',support_length_bucket=label,primary=False)
    else:
        def mean(source,condition,metric,cities,headings):
            group=[id for id in paired[source] if questions[id]['cities']==cities and questions[id]['headings']==headings]
            return float(np.mean([rows[id,condition][metric] for id in group]))
        for mode in ('off','forced'):
            for metric in ('em','f1'):
                for cities in ('together','split'):
                    for headings in ('identical','distinct'):
                        values=[mean(s,'gold_'+mode,metric,cities,headings)-mean(s,'blocked_'+mode,metric,cities,headings) for s in paired]
                        contrast('gold-blocked',metric,values,thinking=mode,cities=cities,headings=headings,primary=False)
                for mask in ('blocked','gold'):
                    values=[(mean(s,mask+'_'+mode,metric,'split','distinct')-mean(s,mask+'_'+mode,metric,'split','identical'))
                            -(mean(s,mask+'_'+mode,metric,'together','distinct')-mean(s,mask+'_'+mode,metric,'together','identical'))
                            for s in paired]
                    contrast('heading_effect_split_minus_together',metric,values,thinking=mode,mask=mask,
                             primary=mode=='forced' and mask=='blocked' and metric=='em')
        transitions=[]
        for cities in ('together','split'):
            for headings in ('identical','distinct'):
                for mask in ('blocked','gold'):
                    changes=[(rows[id,mask+'_off']['em'],rows[id,mask+'_forced']['em'])
                             for group in paired.values() for id in group
                             if questions[id]['cities']==cities and questions[id]['headings']==headings]
                    transitions.append(dict(cities=cities,headings=headings,mask=mask,layouts=len(changes),
                                             fixed=sum(a==0 and b==1 for a,b in changes),broken=sum(a==1 and b==0 for a,b in changes)))
        result['thinking_transitions_descriptive']=transitions
    cap_cells=[]
    for condition in specs:
        for cap in (False,True):
            group=[r for r in rows.values() if r['condition']==condition and bool(r['think_cut'])==cap]
            if group: cap_cells.append(dict(condition=condition,think_cut=cap,rows=len(group),
                                            em=float(np.mean([r['em'] for r in group])),f1=float(np.mean([r['f1'] for r in group]))))
    result['cap_subsets_descriptive']=cap_cells
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'summary.json',result)
    if table:
        with (folder/'cells.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    with (folder/'answers.csv').open('w',newline='') as f:
        fields=['id','source_id','condition','placement','cities','headings','prediction','em','f1','think_tokens','think_cut','cut']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows.values())
    return result
