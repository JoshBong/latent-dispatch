"""Verify full-context decoder-only prefill against the original LM on the L4."""
import gc,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from latent_dispatch import arms,kv
from latent_dispatch.model import Model
from latent_dispatch.wmac_plan import load
bundle,questions=load(ROOT/'results/wmac_gpu_v3_l4/prepared.json')
assert 'L4' in torch.cuda.get_device_name(0)
torch.manual_seed(bundle['config']['seed'])
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=bundle['config']['models']['4b']
model=Model.load(cfg['name'],revision=cfg['revision'],fmt=cfg['format'],device='cuda',dtype='float16')
record=dict(model=cfg,device=torch.cuda.get_device_name(0),torch=torch.__version__,prefill='full_context_no_chunking')
output=ROOT/'results/wmac_gpu_v3_l4/gpu-t4/prefill-diagnostic.json'
output.parent.mkdir(parents=True,exist_ok=True)
short=model.ids('Compare the cache and final hidden state using identical tokens and positions. '*4)
for label,ids in [('short',short),('medium',(short*100)[:1152]),('cached_medium',(short*100)[:1152])]:
    prefix=model.forward(short).past_key_values if label=='cached_medium' else None
    reference=model.forward(ids,kv.copy(prefix) if prefix is not None else None,hidden=True)
    cache,hidden=model.prefill(ids,kv.copy(prefix) if prefix is not None else None)
    torch.testing.assert_close(hidden,reference.hidden_states[-1][:,-1,:],atol=0,rtol=0)
    for actual,expected in zip(kv.layers(cache),kv.layers(reference.past_key_values),strict=True):
        for a,e in zip(actual,expected,strict=True):torch.testing.assert_close(a,e,atol=0,rtol=0)
    record[label]=dict(tokens=len(ids),hidden_and_kv='bitwise_equal')
    output.write_text(json.dumps(record,indent=2)+'\n')
    del reference,cache,hidden,prefix,actual,expected,a,e
    gc.collect();torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
id=max(bundle['musique_ids'],key=lambda id:bundle['preflight']['4b'][id]['context_tokens'])
cache,pos,_=arms.b1_context(model,questions[id])
assert kv.length(cache)==pos
record.update(longest_id=id,prefill_tokens=pos,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,status='passed')
output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
