"""Real-model parity and longest frozen MuSiQue prefill memory check before retry."""
import gc
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from latent_dispatch import arms,kv
from latent_dispatch.model import Model
from latent_dispatch.wmac_plan import load

bundle,questions=load(ROOT/'results/wmac_gpu_v3/prepared.json')
cfg=bundle['config']['models']['4b']
model=Model.load(cfg['name'],revision=cfg['revision'],fmt=cfg['format'],device='cuda',dtype='float16')
record=dict(model=cfg,device=torch.cuda.get_device_name(0),torch=torch.__version__)
output=ROOT/'results/wmac_gpu_v3/gpu-t4/prefill-diagnostic.json'
output.parent.mkdir(parents=True,exist_ok=True)
def save(): output.write_text(json.dumps(record,indent=2)+'\n')
def relative(a,b): return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))

short=model.ids('Compare the cache and final hidden state using the same tokens and positions. ' * 4)
for label,ids in [('short',short),('chunked',(short*100)[:1152]),('cached_chunked',(short*100)[:1152])]:
    prefix=model.forward(short).past_key_values if label=='cached_chunked' else None
    reference=model.forward(ids,kv.copy(prefix) if prefix is not None else None,hidden=True)
    cache,hidden=model.prefill(ids,kv.copy(prefix) if prefix is not None else None)
    pairs=[(hidden,reference.hidden_states[-1][:,-1,:])]
    pairs.extend((a,e) for actual,expected in zip(kv.layers(cache),kv.layers(reference.past_key_values),strict=True)
                 for a,e in zip(actual,expected,strict=True))
    error=max(relative(a,e) for a,e in pairs)
    if label=='short':
        for a,e in pairs: torch.testing.assert_close(a,e,atol=0,rtol=0)
        del a,e
    assert error<=1e-3,(label,error)
    actual=model.forward(short[:1],cache).logits
    expected=model.forward(short[:1],reference.past_key_values).logits
    logit_error=relative(actual,expected)
    assert logit_error<=1e-3,logit_error
    assert actual[:,-1,:].argmax().item()==expected[:,-1,:].argmax().item()
    record[label]=dict(tokens=len(ids),max_relative_hidden_kv_error=error,
                       relative_next_logits_error=logit_error,next_token_equal=True)
    save()
    del reference,cache,hidden,pairs,actual,expected,prefix
    gc.collect();torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
id=max(bundle['musique_ids'],key=lambda id:bundle['preflight']['4b'][id]['context_tokens'])
cache,pos,_=arms.b1_context(model,questions[id])
assert kv.length(cache)==pos
record.update(longest_id=id,prefill_tokens=pos,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
              status='passed')
save();print(json.dumps(record,indent=2))
