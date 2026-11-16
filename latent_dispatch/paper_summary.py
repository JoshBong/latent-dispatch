"""Summaries from saved rows, with source questions as the statistical unit."""
from collections import defaultdict
import csv

import numpy as np

from . import stats
from .paper_plan import SEED, atomic_json
from .score import score


def estimate(values):
    values = list(values)
    if not values:
        return {'sources':0, 'mean':None, 'ci95':None, 'p_unadjusted':None}
    mean, lo, hi = stats.bootstrap_ci(values, n_boot=20000, seed=SEED)
    return {'sources':len(values), 'mean':mean, 'ci95':[lo,hi],
            'p_unadjusted':stats.sign_flip(values, seed=SEED)}


def complete_sources(ids, questions, records, conditions):
    groups = defaultdict(list)
    for id in ids:
        q=questions[id]
        groups[q['dataset'],q['source_id']].append(id)
    return {key:qs for key,qs in groups.items() if all((id,c) in records for id in qs for c in conditions)}


def summarize(job, ids, specs, questions, store, folder):
    rows, cells = {}, []
    for key,row in store.rows.items():
        q=questions[row['id']]
        r={**row, 'dataset':q['dataset'], 'source_id':q['source_id'], 'placement':q['placement']}
        if 'text' in r:
            pred,em,f1=score(r['text'],q['answers'])
            r.update(prediction=pred,em=float(em),f1=f1)
        rows[key]=r
    buckets=defaultdict(list)
    for row in rows.values():
        buckets[row['dataset'],row['placement'],row['condition']].append(row)
    generation_metrics=['em','f1','seconds','gen_tokens','think_tokens','answer_tokens','cut','think_cut','think_eos']
    likelihood_metrics=['seconds','gold_boxed_logprob','gold_prefix_logprob','gold_answer_logprob',
                        'gold_answer_mean_logprob','gold_closing_logprob','gold_minus_best_wrong',
                        'gold_minus_best_wrong_mean','boxed_change_from_archive']
    metrics=likelihood_metrics if job=='likelihood' else generation_metrics
    for (dataset,placement,condition),group in sorted(buckets.items()):
        cells.append({'dataset':dataset,'placement':placement,'condition':condition,'n':len(group),
                      **{m:float(np.mean([r[m] for r in group])) for m in metrics}})
    paired=complete_sources(ids,questions,rows,specs)
    output={'job':job,'expected_rows':len(ids)*len(specs),'saved_rows':len(rows),
            'complete':len(rows)==len(ids)*len(specs),'cells':cells,
            'fully_paired_sources':dict((d,sum(k[0]==d for k in paired)) for d in ['toy_d2','musique']),
            'statistics':{'unit':'source question; average placements within source',
                          'bootstrap_resamples':20000,'seed':SEED,'ci':'percentile, unadjusted',
                          'caution':'A zero-width interval on unanimous samples is not proof of equivalence.'}}
    if job=='validation':
        from .paper_experiments import compare_endpoints
        comparisons=[]
        for id in ids:
            for a,b in [('masked_full','b1'),('masked_blocked','b1_mask')]:
                if (id,a) in rows and (id,b) in rows:
                    comparisons.append({'id':id,'a':a,'b':b,
                        'em_difference':rows[id,a]['em']-rows[id,b]['em'],
                        **compare_endpoints(rows[id,a],rows[id,b],folder)})
        output['endpoint_comparisons']=comparisons
        output['interpretation']='Purposively selected diagnostics; not an unbiased accuracy sample or automatic parity approval.'
    else:
        contrasts=[]
        datasets=sorted({k[0] for k in paired})
        if job=='thinking':
            for dataset in datasets:
                groups=[v for k,v in paired.items() if k[0]==dataset]
                for metric in ['em','f1']:
                    for a,b in [('full','blocked'),('gold','blocked'),('gold','full')]:
                        gaps={}
                        for mode in ['off','forced']:
                            values=[float(np.mean([rows[id,a+'_'+mode][metric]-rows[id,b+'_'+mode][metric]
                                                   for id in group])) for group in groups]
                            gaps[mode]=values
                            contrasts.append({'dataset':dataset,'metric':metric,'contrast':a+'-'+b,
                                              'thinking':mode,'primary':False,**estimate(values)})
                        values=np.asarray(gaps['forced'])-np.asarray(gaps['off'])
                        contrasts.append({'dataset':dataset,'metric':metric,'contrast':a+'-'+b,
                                          'thinking':'forced_gap_minus_off_gap',
                                          'primary':metric=='em' and b=='blocked',**estimate(values)})
            primary=[r for r in contrasts if r['primary']]
            # Keep the prespecified family of four even in partial checkpoints.
            pvalues=[r['p_unadjusted'] for r in primary]+[1.0]*(4-len(primary))
            for row,p in zip(primary,stats.holm(pvalues)):
                row['p_holm_primary_interactions']=p
            output['primary_test_family']='Four EM interactions: two datasets x (full-blocked, gold-blocked); Holm adjusted.'
            output['interaction_sign']='Negative means thinking reduced the attention gap; token budgets are matched, actual compute is not.'
        else:
            for dataset in datasets:
                for placement in ['AB','BA','AB+BA']:
                    groups=[[id for id in group if placement=='AB+BA' or questions[id]['placement']==placement]
                            for key,group in paired.items() if key[0]==dataset]
                    groups=[g for g in groups if g]
                    if not groups:
                        continue
                    for metric in likelihood_metrics[1:]:
                        for a,b in [('distractor','blocked'),('gold','blocked'),('gold','distractor')]:
                            values=[float(np.mean([rows[id,a][metric]-rows[id,b][metric] for id in group]))
                                    for group in groups]
                            contrasts.append({'dataset':dataset,'placement':placement,'metric':metric,
                                              'contrast':a+'-'+b,'primary':False,**estimate(values)})
            output['interpretation']='Exploratory pre-gold subsets. Candidate length-normalized scores are secondary; archived generation is not newly rerun.'
        output['contrasts']=contrasts
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'summary.json',output)
    if cells:
        with (folder/'cells.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(cells[0]))
            writer.writeheader();writer.writerows(cells)
    return output
