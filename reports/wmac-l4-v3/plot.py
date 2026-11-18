"""Plot the audited, source-paired primary results."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
s=json.loads((HERE/'recomputed/summary.json').read_text())
absolute={r['condition']:r for r in s['paired_absolute_scores'] if r['metric']=='f1'}
order=['blocked_off','matched_irrelevant_off','k0_off','full_off','b1_off','gold_off']
labels=['Blocked','Equal-token irrelevant','Separate caches (K0)','Full access','Joint reading (B1)','Supporting paragraph']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
y=np.arange(6);m=np.array([absolute[c]['mean']*100 for c in order]);lo=np.array([absolute[c]['ci95'][0]*100 for c in order]);hi=np.array([absolute[c]['ci95'][1]*100 for c in order])
ax.barh(y,m,color=['#94a3b8','#94a3b8','#60a5fa','#60a5fa','#2563eb','#d97706'],height=.58)
ax.errorbar(m,y,xerr=[m-lo,hi-m],fmt='none',ecolor='#1f2937',capsize=3,lw=1)
for yi,v,h in zip(y,m,hi):ax.text(h+1,yi,f'{v:.1f}',va='center',fontsize=10)
ax.set(yticks=y,yticklabels=labels,xlim=(0,60),xlabel='Token F1 score (0–100)',title='Absolute performance')
ax.invert_yaxis();ax.xaxis.grid(True,alpha=.16);ax.set_axisbelow(True)
names=['gold-blocked','gold-matched_irrelevant','b1-k0'];rows=[next(r for r in s['contrasts'] if r['contrast']==n and r['metric']=='f1' and r.get('cohort')=='all' and r.get('placement')=='AB+BA') for n in names]
for i,r in enumerate(rows):
 v=r['mean']*100;l,h=[z*100 for z in r['ci95']]
 bx.errorbar(v,i,xerr=[[v-l],[h-v]],fmt='o',color='#0f766e',capsize=4,ms=7)
 bx.text(v,i-.17,f'+{v:.1f}',ha='center',va='bottom',color='#0f766e',fontweight='bold')
bx.set(yticks=range(3),yticklabels=['Gold − blocked','Gold − irrelevant','Joint − separate caches'],xlim=(-1,17),ylim=(2.55,-.55),xlabel='Paired F1 difference (percentage points)',title='Primary contrasts')
bx.axvline(0,color='#475569',lw=1);bx.xaxis.grid(True,alpha=.16)
fig.suptitle('Qwen3-4B masking experiment · thinking off · FP16 on L4\n200 source questions, AB/BA support layouts',fontsize=14)
fig.supxlabel('95% unadjusted source-paired bootstrap intervals · 20,000 resamples',fontsize=9,color='#475569')
folder=HERE/'figures';folder.mkdir(exist_ok=True)
for ext in ('png','svg'):fig.savefig(folder/f'results.{ext}',dpi=180,bbox_inches='tight')

svg=folder/"results.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
