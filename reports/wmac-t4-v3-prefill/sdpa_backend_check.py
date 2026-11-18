import json,torch
import torch.nn.functional as F
from torch.profiler import profile,ProfilerActivity
from torch.nn.attention import sdpa_kernel,SDPBackend
q=torch.randn(1,32,128,128,device='cuda',dtype=torch.float16)
k=torch.randn(1,8,128,128,device='cuda',dtype=torch.float16);v=torch.randn_like(k)
records={}
for label,gqa in [('native_gqa',True),('expanded_kv',False)]:
 kk=k if gqa else k.repeat_interleave(4,dim=1);vv=v if gqa else v.repeat_interleave(4,dim=1)
 with profile(activities=[ProfilerActivity.CPU]) as p:
  out=F.scaled_dot_product_attention(q,kk,vv,is_causal=True,enable_gqa=gqa)
  torch.cuda.synchronize()
 records[label]=[x.key for x in p.key_averages() if 'attention' in x.key]
print(json.dumps(records,indent=2))
