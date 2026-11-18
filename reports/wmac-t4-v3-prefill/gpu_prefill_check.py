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
ids=model.ids('Compare the cache and final hidden state using the same tokens and positions. ' * 4)
reference=model.forward(ids,hidden=True)
cache,hidden=model.prefill(ids)
torch.testing.assert_close(hidden,reference.hidden_states[-1][:,-1,:],atol=0,rtol=0)
for actual,expected in zip(kv.layers(cache),kv.layers(reference.past_key_values),strict=True):
    for a,e in zip(actual,expected,strict=True):
        torch.testing.assert_close(a,e,atol=0,rtol=0)
del reference,cache,hidden
gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
id=max(bundle['musique_ids'],key=lambda id:bundle['preflight']['4b'][id]['context_tokens'])
cache,pos,_=arms.b1_context(model,questions[id])
assert kv.length(cache)==pos
record=dict(model=cfg,parity='bitwise_equal_short_context_hidden_and_all_kv',
            longest_id=id,prefill_tokens=pos,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            device=torch.cuda.get_device_name(0),torch=torch.__version__)
print(json.dumps(record,indent=2))
