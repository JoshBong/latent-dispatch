"""Tests supplied for Josh; adding this file does not execute them."""
import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latent_dispatch import arms,masking,wmac_experiments,wmac_plan,wmac_summary
from latent_dispatch.paper_plan import RowStore
from scripts import run_wmac_a100 as runner
from .conftest import make_question


def matched_question():
    return make_question([
        ['Archives: Blank paper and unused envelopes fill the empty storage cabinet. '
         'The folders contain no entries. The ledger remains empty and unused.',
         "People: Ines Farrow's favorite city is Velmora."],
        ['Cities: The mayor of Velmora is Bram Holt.','Records: The inventory contains spare pencils.']],
        gold=((0,0,1),(1,1,0)))


def test_matched_control_only_changes_the_intended_attention_edges(tiny):
    q=matched_question()
    ids,mask,sizes,meta=wmac_experiments.matched_layout(tiny,q)
    reference,original,_,_=masking.layout(tiny,q,'gold')
    assert ids==reference
    prefix=len(ids)-sum(sizes); b=prefix+sizes[0]
    visible=(mask[0,0,b,prefix:b]==0).nonzero().flatten().tolist()
    assert visible==meta['a_visible_indices']
    assert len(visible)==meta['a_gold_tokens']
    assert max(visible)<min(meta['a_gold_indices'])
    outside=torch.ones_like(mask,dtype=torch.bool);outside[:,:,b:,prefix:b]=False
    assert torch.equal(mask[outside],original[outside])
    future=torch.triu(torch.ones(len(ids),len(ids),dtype=torch.bool),1)
    assert torch.all(mask[0,0][future]<0)


def test_insufficient_control_tokens_fail_instead_of_changing_budget(tiny):
    q=matched_question();q['workers'][0][0]='X.'
    with pytest.raises(ValueError,match='insufficient'):
        wmac_experiments.matched_indices(tiny,q)


@pytest.mark.parametrize('condition',['b1','k0'])
def test_fresh_baselines_use_actual_original_context_paths(tiny_family,two_workers,monkeypatch,condition):
    calls=[]
    name='b1_context' if condition=='b1' else 'k0_context'
    original=getattr(arms,name)
    def capture(*a,**kw):
        calls.append(kw)
        return original(*a,**kw)
    monkeypatch.setattr(arms,name,capture)
    row=wmac_experiments.generate(tiny_family,two_workers,condition,'off',8,2)
    assert calls==([{}] if condition=='b1' else [{'positions':'local','merge':'seq'}])
    assert row['mask']['key_shift']==(condition=='k0')
    assert row['mask']['encoding_positions']==('local' if condition=='k0' else 'true')
    assert row['answer_tokens']<=2


def test_prompt_template_and_shift_guards(tiny_family,two_workers,monkeypatch):
    assert wmac_experiments.check_prompt_template(tiny_family,two_workers)['native_template_matches']
    assert wmac_experiments.check_rope_shift(tiny_family)['relative_l2']<1e-4
    monkeypatch.setattr(tiny_family.tok,'apply_chat_template',lambda *a,**kw:'wrong native template')
    with pytest.raises(ValueError,match='native chat template'):
        wmac_experiments.check_prompt_template(tiny_family,two_workers)


def test_factorial_keeps_people_and_gold_positions_fixed(tiny):
    qs=wmac_plan.factorial_toy(3,20261116,tiny.tok)
    assert len(qs)==24
    qdict={q['id']:q for q in qs}
    checked=wmac_experiments.preflight(tiny,qdict,list(qdict),8192)
    for q in qs:
        assert [len(w) for w in q['workers']]==[3,3]
        assert {g['worker'] for g in q['gold']}=={0,1}
        assert [g['index'] for g in q['gold']]==[0,1]
    for source in {q['source_id'] for q in qs}:
        for order in ('AB','BA'):
            group=[q for q in qs if q['source_id']==source and q['placement']==order]
            assert len({tuple(w[0] for w in q['workers']) for q in group})==1
            assert len({tuple(checked[q['id']]['worker_tokens']) for q in group})==1


def test_numerical_comparison_feeds_identical_continuations(tiny,two_workers,tmp_path):
    continuation=tiny.ids('A fixed answer prefix.')
    r=wmac_experiments.compare_same_prefix(tiny,two_workers,'b1','masked_full',continuation,tmp_path)
    assert r['continuation_token_ids']==continuation
    assert len(r['steps'])==len(continuation)+1
    assert max(s['max_abs_delta'] for s in r['steps'])<1e-4
    with np.load(tmp_path/r['logits_file'],allow_pickle=False) as saved:
        assert saved['reference'].shape==saved['candidate'].shape


def test_history_scan_skips_appledouble_metadata(tmp_path):
    with tarfile.open(tmp_path/'old.tar.gz','w:gz') as t:
        for name,payload in [('x/._rows.jsonl',b'\x00\x05metadata'),('x/rows.jsonl',b'{"id":"2hop__old:AB:s0"}\n')]:
            m=tarfile.TarInfo(name);m.size=len(payload);t.addfile(m,io.BytesIO(payload))
    ids,hashes=wmac_plan.historical_sources(tmp_path)
    assert ids=={'2hop__old'} and len(hashes)==1


def pair(source,question='Who leads the favorite city?',city='Velmora',answer='Bram Holt'):
    out=[]
    for order in ('AB','BA'):
        workers=[[f'People: Ines prefers {city}.','Other: Empty paper.'],
                 [f'Cities: {city} is led by {answer}.','More: Spare folders.']]
        if order=='BA':workers=workers[::-1]
        pw,cw=(0,1) if order=='AB' else (1,0)
        q=make_question(workers,gold=((0,pw,0),(1,cw,0)),id=source+':'+order)
        q.update(source_id=source,question=question,answers=[answer],bridges=[city],dataset='musique',qtype='2hop',placement=order)
        out.append(q)
    return out


def test_selection_deduplicates_supports_and_honors_prospective_reviews():
    old=pair('2hop__old');reused=pair('2hop__reuse',question='Which leader is wanted?')
    fresh=pair('2hop__fresh',question='Who leads the new city?',city='Kastrel',answer='Celia Vance')
    selected,_=wmac_plan.select_sources(old+reused+fresh,{'2hop__old'},1,20261116)
    assert {q['source_id'] for q in selected}=={'2hop__fresh'}
    assert all(g['index']==1 for q in selected for g in q['gold'])
    with pytest.raises(ValueError,match='eligible'):
        wmac_plan.select_sources(fresh,set(),1,20261116,reviews={
            '2hop__fresh':{'reviewer':'reviewer','reason':'ambiguous support','exclude':True}})
    with pytest.raises(ValueError,match='eligible'):
        wmac_plan.select_sources(fresh,set(),1,20261116,eligible=lambda q:False)


def output(text='\\boxed{Wrong}',cut=False):
    return dict(text=text,gen_tokens=5,think_tokens=0,answer_tokens=5,cut=cut,think_cut=False,think_eos=False,
                worker_tokens=[100,100],mask={'a_gold_tokens':50})


def config():
    return json.loads(wmac_plan.CONFIG.read_text())


def test_v2_has_three_matched_models_and_preserves_full_toy_controls():
    cfg=config()
    for job in ('musique-8b','musique-4b','musique-llama'):
        specs=wmac_plan.specifications(job,cfg)
        assert set(specs)=={c+'_off' for c in wmac_plan.CONDITIONS}
        assert cfg['expected_main_rows'][job]==200*2*len(specs)==2400
        assert all(s['thinking']=='off' for s in specs.values())
    assert wmac_plan.model_key('musique-llama')==wmac_plan.model_key('validation-llama')=='llama'
    assert 'b1_mask' in wmac_plan.specifications('validation-llama',cfg)
    assert cfg['toy_sources']==25 and cfg['expected_main_rows']['toy-8b']==800
    assert len(wmac_plan.specifications('toy-8b',cfg))==4
    assert sum(n for job,n in cfg['expected_main_rows'].items() if not wmac_plan.is_validation(job))==8000
    assert sum(cfg['expected_probe_rows'].values())==1800


def test_reuse_preserves_questions_and_aliases_without_replacement():
    qs={q['id']:q for q in pair('s1')+pair('s2',question='Another question?')}
    qs['s1:AB']['answers'].append('Accepted alias')
    old=dict(questions=qs,musique_ids=list(qs))
    new=wmac_plan.reused_sources(old,2,lambda q:True)
    assert {q['id']:q for q in new}==qs
    new[0]['answers'].append('Local mutation')
    assert 'Local mutation' not in qs[new[0]['id']]['answers']
    with pytest.raises(ValueError,match='no question was replaced'):
        wmac_plan.reused_sources(old,2,lambda q:q['id']!='s1:AB')


def test_summary_counts_sources_and_does_not_filter_primary_by_probe_success(tmp_path):
    qs={q['id']:q for q in pair('s1')+pair('s2',question='Another question?')}
    specs={c+'_off':{} for c in wmac_plan.CONDITIONS}
    store=RowStore(tmp_path/'rows.jsonl','fp',runner.schedule(list(qs),specs))
    for id in qs:
        for c in specs:
            if id=='s2:BA' and c=='full_off':continue
            store.append(dict(id=id,condition=c,seconds=1,**output('\\boxed{Bram Holt}' if c=='gold_off' else '\\boxed{Wrong}')))
    probes=RowStore(tmp_path/'probes.jsonl','fp',[(id,p) for id in qs for p in ('closed_book','keep_hop0','keep_hop1')])
    for p in ('closed_book','keep_hop0','keep_hop1'):
        probes.append(dict(id='s1:AB',condition=p,seconds=1,**output('\\boxed{Bram Holt}')))
    cfg=config();cfg['bootstrap_resamples']=100
    r=wmac_summary.summarize('musique-8b',list(qs),specs,qs,store,tmp_path/'summary',cfg,probes)
    assert r['fully_paired_sources']==1 and not r['complete']
    primary=next(c for c in r['contrasts'] if c['primary'] and c['contrast']=='gold-blocked')
    assert primary['sources']==1 and primary['mean']==1
    assert r['probes']['strict_paired_sources']==0


def test_smoke_does_not_gate_on_correctness_or_natural_truncation():
    row=output('Cannot answer.',cut=True)
    runner.check_smoke(SimpleNamespace(rows={('q','off'):row}),{'off':{'thinking':'off'}})
    with pytest.raises(ValueError,match='did not close'):
        runner.check_smoke(SimpleNamespace(rows={('q','forced'):row}),{'forced':{'thinking':'forced'}})


@pytest.mark.parametrize('argv',[
    ['toy-8b','--dtype','float32'],['musique-4b','--attention','eager'],
    ['toy-8b','--validation-limit','2'],['validation','--max-hours','nan'],['prepare','--plan-only']])
def test_cli_refuses_incompatible_settings(argv):
    with pytest.raises(SystemExit):runner.arguments(argv)


def test_interruption_resume_and_self_contained_archive(monkeypatch,tmp_path):
    from latent_dispatch.model import Model
    qs={q['id']:q for q in pair('s1')+pair('s2',question='Another question?')}
    cfg=config();cfg['expected_main_rows']['musique-4b']=24;cfg['bootstrap_resamples']=100
    bundle=dict(config=cfg,musique_ids=list(qs))
    prepared=tmp_path/'prepared.json';prepared.write_text(json.dumps(bundle))
    monkeypatch.setattr(wmac_plan,'load',lambda path:(bundle,qs))
    monkeypatch.setattr(Model,'load',lambda *a,**kw:object())
    monkeypatch.setattr(runner,'runtime_metadata',lambda model:{'model_revision':'fixed'})
    monkeypatch.setattr(wmac_experiments,'preflight',lambda *a:{})
    monkeypatch.setattr(wmac_experiments,'check_prompt_template',lambda *a:{})
    monkeypatch.setattr(wmac_experiments,'check_rope_shift',lambda *a:{})
    monkeypatch.setattr(wmac_experiments,'generate',lambda *a:output())
    calls=[0]
    def probe(*args):
        calls[0]+=1
        if calls[0]==2:raise RuntimeError('simulated interruption')
        return output()
    monkeypatch.setattr(wmac_experiments,'probe',probe)
    argv=['musique-4b','--device','cpu','--prepared',str(prepared),'--output-root',str(tmp_path)]
    with pytest.raises(RuntimeError,match='simulated'):runner.main(argv)
    folder=tmp_path/'wmac_a100_v2/musique-4b'
    assert len((folder/'probes/rows.jsonl').read_text().splitlines())==1
    assert runner.main(argv)==0
    assert len((folder/'probes/rows.jsonl').read_text().splitlines())==6
    assert len((folder/'main/rows.jsonl').read_text().splitlines())==24
    before=calls[0]
    assert runner.main(argv)==0 and calls[0]==before
    with tarfile.open(str(folder)+'.tar.gz') as t:
        names=t.getnames()
        assert any(n.endswith('/prepared.json') for n in names)
        assert any(n.endswith('/probes/rows.jsonl') for n in names)
        assert any(n.endswith('/main/summary.json') for n in names)
