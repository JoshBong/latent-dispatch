"""Exploratory same-source comparison of all three completed control experiments."""
import importlib.util,json,pathlib,sys,tarfile
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
s=importlib.util.spec_from_file_location('tables',ROOT/'scripts/paper_tables.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
qs=t.questions();all_diffs={};rows=[];common=None
for model in ('Qwen3-8B','Llama 3.1 8B','Qwen3-4B'):
    main,sources,_=t.main_run(model,qs)
    if common is None:common=sources
    assert sources==common and len(sources)==200
    if model=='Qwen3-4B':
        with tarfile.open(HERE/'results-and-logs.tar.gz') as tar:
            raw=t.jsonl(tar.extractfile('wmac-4b-together/4b/together/rows.jsonl').read())
        ctrl=t.layout_scores(raw,{r['id']:qs[r['base_id']] for r in raw})
    else:ctrl=t.together_run(model,qs)
    assert t.sources_of(ctrl)==sources and len(ctrl)==800
    split=np.asarray(t.difference(main,'full','blocked',sources,1))
    together=np.asarray(t.difference(ctrl,'full','blocked',sources,1,('AA','BB')))
    interaction=split-together;all_diffs[model]=dict(split=split,together=together,interaction=interaction)
    row={'model':model}
    for name,values in all_diffs[model].items():
        mean,lo,hi=t.stats.bootstrap_ci(values,n_boot=20000,seed=20261116)
        row[name]=dict(mean=mean,ci95=[lo,hi])
    rows.append(row)
pooled={}
for name in ('split','together','interaction'):
    values=np.mean([d[name] for d in all_diffs.values()],axis=0)
    mean,lo,hi=t.stats.bootstrap_ci(values,n_boot=20000,seed=20261116)
    pooled[name]=dict(mean=mean,ci95=[lo,hi])
output=dict(sources=200,models=rows,equal_model_average=pooled,
    statistics='20,000 paired source bootstrap resamples, seed 20261116, 95% unadjusted percentile intervals. Models averaged within source, not treated as 600 independent sources.',
    scope='Secondary cross-model summary; 8B models use BF16/A100, 4B uses FP16/L4. Does not isolate model size or prove universality.')
(HERE/'cross-model.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
