"""Check the longest AA and BB control layouts before the 800-answer run."""
import importlib.util,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
s=importlib.util.spec_from_file_location('together',ROOT/'reports/wmac-4b-together/wmac_together.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=m.load_bundle('/home/elkalt/prepared.json')
model=m.load_model(b,'4b',device='cuda',dtype='float16')
assert model.dtype_name=='float16' and 'L4' in m.runtime(model)['device_name']
assert m.digest(model.tok.backend_tokenizer.to_str())==b['tokenizer_sha256']['4b']
kept,dropped=m.together_layouts(model,b['questions'],m.base_ids(b),b['config']['max_context_tokens'])
assert len(kept)==400 and not dropped
folder=pathlib.Path('/home/elkalt/wmac-4b-together-smoke');rows=m.Rows(folder/'rows.jsonl')
for placement in ('AA','BB'):
    q=max((q for q in kept.values() if q['placement']==placement),key=lambda q:len(m.paper_experiments.context_tokens(model,q)[0]))
    for condition in ('blocked','full'):
        key=q['id']+'|'+condition
        if key in rows.rows:continue
        out=m.wmac_experiments.generate(model,q,condition,'off',b['config']['think_budget'],b['config']['answer_tokens'])
        assert out['text'].strip() and not out['cut'] and '\\boxed{' in out['text'],(key,out['text'])
        rows.add(key,dict(id=q['id'],source_id=q['source_id'],placement=placement,condition=condition+'_off',question_sha256=m.digest(q),**out))
        print('SMOKE',key,out['gen_tokens'],'tokens',flush=True)
assert len(rows.rows)==4
m.write_json(folder/'status.json',dict(complete=True,rows=4,layouts=400,dropped=0,runtime=m.runtime(model)))
print('CONTROL SMOKE PASSED',flush=True)
