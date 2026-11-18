"""Protocol, resume, scoring, and numerical diagnostics; no pretrained model weights."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from latent_dispatch import arms, kv, masking, prompts
from latent_dispatch.paper_experiments import boxed_parts, compare_endpoints, endpoint, likelihood, thinking
from latent_dispatch.paper_plan import PLAN, RowStore, balanced_sample, digest, frozen_json, specifications, toy_candidates
from latent_dispatch.paper_summary import complete_sources, summarize
from .conftest import make_question

spec=importlib.util.spec_from_file_location('paper_runner',Path(__file__).resolve().parents[1]/'scripts/run_paper_ablations.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_frozen_sample_matches_agreed_counts():
    plan=json.loads(PLAN.read_text())
    assert [plan[j]['expected_rows'] for j in ['validation','thinking','likelihood']]==[96,600,255]
    for job in ['validation','thinking','likelihood']:
        ids=plan[job]['ids']
        assert len(ids)==len(set(ids))
        assert len(ids)*len(specifications(job,plan))==plan[job]['expected_rows']
    counts={c:sum(v['archived_group']==c for v in plan['validation']['selection'].values())
            for c in ['mismatch','agreement']}
    assert counts=={'mismatch':12,'agreement':12}
    assert sum(i.endswith(':AB') for i in plan['likelihood']['ids'])==42
    assert sum(i.endswith(':BA') for i in plan['likelihood']['ids'])==43
    for value in plan['likelihood']['candidates'].values():
        assert len(value['answers'])==len(set(value['answers']))==3
    thinking_ids=plan['thinking']['ids']
    assert sum(i.startswith('toy-') for i in thinking_ids)==50
    assert sum(i.startswith('2hop__') for i in thinking_ids)==50
    assert all(thinking_ids[i].replace(':AB',':BA')==thinking_ids[i+1] for i in range(0,100,2))


def test_balanced_selection_deterministic_and_exhaustion():
    questions={str(i):{'dataset':'a' if i<10 else 'b','placement':'AB' if i%2 else 'BA'} for i in range(20)}
    selected=balanced_sample(list(questions),questions,8,'test')
    assert selected==balanced_sample(list(reversed(questions)),questions,8,'test')
    assert sum(questions[i]['dataset']=='a' for i in selected)==4
    with pytest.raises(ValueError,match='eligible'):
        balanced_sample(questions,questions,21,'test')


def test_manifest_refuses_sample_or_budget_change(tmp_path):
    path=tmp_path/'manifest.json'
    frozen_json(path,{'ids':['a'],'budget':1024})
    frozen_json(path,{'ids':['a'],'budget':1024})
    with pytest.raises(ValueError,match='protocol changed'):
        frozen_json(path,{'ids':['a'],'budget':256})


def test_resume_keeps_complete_rows_and_recovers_torn_tail(tmp_path):
    path=tmp_path/'rows.jsonl';allowed=[('a','off'),('b','off')]
    store=RowStore(path,'fp',allowed)
    store.append({'id':'a','condition':'off','value':1})
    with path.open('ab') as f:
        f.write(b'{"id":"b",')
    resumed=RowStore(path,'fp',allowed)
    assert list(resumed.rows)==[('a','off')]
    assert len(list(tmp_path.glob('rows.jsonl.torn-*')))==1
    resumed.append({'id':'b','condition':'off','value':2})
    assert len(RowStore(path,'fp',allowed).rows)==2
    with pytest.raises(ValueError,match='fingerprint'):
        RowStore(path,'new-fp',allowed)
    with pytest.raises(ValueError,match='duplicate'):
        resumed.append({'id':'a','condition':'off'})


def test_resume_does_not_hide_interior_corruption(tmp_path):
    path=tmp_path/'rows.jsonl'
    path.write_text('{broken}\n'+json.dumps({'id':'b','condition':'off','fingerprint':'fp'})+'\n')
    with pytest.raises(ValueError,match='corrupt interior'):
        RowStore(path,'fp',[('b','off')])


def test_valid_final_row_without_newline_is_preserved(tmp_path):
    path=tmp_path/'rows.jsonl'
    path.write_text(json.dumps({'id':'a','condition':'off','fingerprint':'fp'}))
    store=RowStore(path,'fp',[('a','off'),('b','off')])
    store.append({'id':'b','condition':'off'})
    assert len(RowStore(path,'fp',[('a','off'),('b','off')]).rows)==2


def test_condition_schedule_finishes_layout_and_resume_skips_it(tmp_path):
    qs={i:{'id':i} for i in ['source:AB','source:BA']}
    specs={'off':{},'forced':{}};calls=[]
    store=RowStore(tmp_path/'rows.jsonl','fp',runner.schedule(qs,specs))
    def evaluate(q,name,options):
        calls.append((q['id'],name));return {'text':'done'}
    assert runner.execute_phase('test',list(qs),specs,qs,store,evaluate,lambda:False,float('inf'),'test')
    assert len(calls)==4 and calls[0][0]==calls[1][0]=='source:AB'
    assert runner.execute_phase('test',list(qs),specs,qs,store,evaluate,lambda:False,float('inf'),'test')
    assert len(calls)==4


def test_stop_between_rows_preserves_pairable_progress(tmp_path):
    specs={'a':{},'b':{}};qs={'q':{'id':'q'}}
    store=RowStore(tmp_path/'rows.jsonl','fp',runner.schedule(qs,specs))
    assert not runner.execute_phase('test',list(qs),specs,qs,store,lambda *a:{'text':'x'},
                                    lambda:len(store.rows)==1,float('inf'),'test')
    assert len(store.rows)==1


def test_thinking_smoke_requires_final_answer_after_close():
    assert runner.complete_box('reason </think> \\boxed{X}',True)
    assert runner.complete_box('\\boxed{\\text{X}}')
    assert not runner.complete_box('\\boxed{X}',True)
    assert not runner.complete_box('\\boxed{X} </think> missing answer',True)
    assert not runner.complete_box('\\boxed{unfinished')


def test_validation_caps_are_diagnostics_but_thinking_caps_fail_smoke(tmp_path):
    store=RowStore(tmp_path/'rows.jsonl','fp',[('q','a')])
    store.append({'id':'q','condition':'a','text':'unfinished answer','cut':True})
    specs={'a':{'thinking':'off'}}
    runner.check_smoke(store,specs,'validation',1)
    with pytest.raises(ValueError,match='smoke failed'):
        runner.check_smoke(store,specs,'thinking',1)


def test_resume_checks_recorded_options_and_question_hash(tmp_path):
    q={'id':'q'};specs={'a':{'budget':1024}}
    store=RowStore(tmp_path/'rows.jsonl','fp',[('q','a')])
    store.append({'id':'q','condition':'a','options':{'budget':256},'question_sha256':digest(q)})
    with pytest.raises(ValueError,match='saved row settings'):
        runner.verify_saved_rows(store,specs,{'q':q})


def test_partial_source_is_excluded_from_paired_analysis():
    ids=['AB','BA'];qs={i:{'dataset':'toy','source_id':'s'} for i in ids}
    rows={('AB','a'):{},('AB','b'):{},('BA','a'):{}}
    assert complete_sources(ids,qs,rows,['a','b'])=={}
    rows['BA','b']={}
    assert complete_sources(ids,qs,rows,['a','b'])=={('toy','s'):ids}


@pytest.fixture
def candidate_question():
    return make_question([
        ['Aria Venn was born in Dunmere.', "People: Ines Farrow's favorite city is Velmora."],
        ['Cities: The mayor of Velmora is Bram Holt. The mayor of Kastrel is Celia Vance. The mayor of Orinth is Dario Okafor.']],
        gold=((0,0,1),(1,1,0)))


def test_candidates_are_fixed_from_evidence(candidate_question):
    assert toy_candidates(candidate_question)==['Bram Holt','Celia Vance','Dario Okafor']


def test_boxed_partition_preserves_original_target(tiny):
    for answer in ['Bram Holt','Celia Vance','Dario Okafor']:
        target,prefix_n,answer_n=boxed_parts(tiny,answer)
        assert target==tiny.ids('\\boxed{'+answer+'}')
        assert tiny.decode(target[prefix_n:prefix_n+answer_n])==answer
        assert target[:prefix_n]==tiny.ids('\\boxed{')


def test_likelihood_decomposition_and_independent_cache_copies(tiny,candidate_question):
    q=candidate_question
    candidates={'answers':toy_candidates(q),'a_distractor_index':0,
                'archived':{'distractor':{'gold_logprob':-1,'em':False,'cut':False}}}
    result=likelihood(tiny,q,'distractor',candidates)
    assert result['mask']['a_paragraphs']==[0]
    for row in result['candidates']:
        assert row['boxed_logprob']==pytest.approx(row['prefix_logprob']+row['answer_logprob']+row['closing_logprob'])
        assert row['answer_mean_logprob']==pytest.approx(row['answer_logprob']/row['answer_token_count'])
        # Fresh context for each candidate must agree with the copies used by the shared-context scorer.
        cache,pos,_,_=masking.context(tiny,q,'distractor')
        tail=tiny.ids(prompts.tail(tiny.fmt,'off'))
        target=row['target_token_ids']
        out=tiny.forward(tail+target,cache,pos)
        logits=out.logits[0,len(tail)-1:len(tail)+len(target)-1].float()
        manual=torch.log_softmax(logits,-1).gather(1,torch.tensor(target).unsqueeze(1)).sum().item()
        assert row['boxed_logprob']==pytest.approx(manual,abs=1e-4)
    wrong=max(r['answer_logprob'] for r in result['candidates'][1:])
    assert result['gold_minus_best_wrong']==pytest.approx(result['gold_answer_logprob']-wrong)


def test_likelihood_refuses_changed_distractor(tiny,candidate_question):
    with pytest.raises(ValueError,match='frozen pre-gold'):
        likelihood(tiny,candidate_question,'distractor',{'a_distractor_index':1})


def test_endpoint_diagnostics_save_true_generation_and_logits(tiny,two_workers,tmp_path):
    a=endpoint(tiny,two_workers,'masked_full',2,tmp_path)
    b=endpoint(tiny,two_workers,'b1',2,tmp_path)
    diff=compare_endpoints(a,b,tmp_path)
    assert diff['first_logits_max_abs_delta']<1e-4
    assert diff['same_generated_tokens']
    assert len(a['token_ids'])==a['gen_tokens']==2
    assert a['trace'][0]['argmax']==a['token_ids'][0]
    broken={**b,'context_tokens_sha256':'wrong'}
    with pytest.raises(ValueError,match='structural mismatch'):
        compare_endpoints(a,broken,tmp_path)


def test_trace_does_not_change_generation(tiny,two_workers):
    cache,pos,_=arms.b1_context(tiny,two_workers)
    tail=tiny.ids(prompts.tail(tiny.fmt,'off'))
    untraced=tiny.greedy(tail,kv.copy(cache),pos,max_new=3)
    traced=tiny.greedy(tail,kv.copy(cache),pos,max_new=3,trace_topk=5)
    assert untraced.text==traced.text and untraced.token_ids==traced.token_ids
    assert not untraced.trace and traced.trace


def test_masked_thinking_uses_generation_without_old_likelihood_guard(tiny,two_workers,monkeypatch):
    m=copy.deepcopy(tiny);calls=[]
    original=m.greedy
    def capture(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(m,'greedy',capture)
    output=thinking(m,two_workers,'gold','forced',2,2)
    assert calls[0]['think_budget']==2 and calls[0]['max_new']==2
    assert '</think>' in output['text'] and output['think_cut']
    assert output['answer_tokens']==2


def test_primary_interactions_require_all_six_conditions(tmp_path,monkeypatch):
    # The sign and source aggregation are tested without expensive resampling.
    from latent_dispatch import paper_summary
    monkeypatch.setattr(paper_summary,'estimate',lambda v:{'sources':len(v),'mean':float(np.mean(v)),
                                                        'ci95':[0,1],'p_unadjusted':0.1})
    ids=['AB','BA'];qs={i:{'dataset':'toy_d2','source_id':'s','placement':i,'answers':['X']} for i in ids}
    specs={c+'_'+m:{} for c in ['blocked','gold','full'] for m in ['off','forced']}
    store=RowStore(tmp_path/'rows.jsonl','fp',runner.schedule(ids,specs))
    for id in ids:
        for name in specs:
            right=name!='blocked_off'
            store.append({'id':id,'condition':name,'text':'\\boxed{'+('X' if right else 'Y')+'}',
                          'seconds':1,'gen_tokens':1,'answer_tokens':1,'think_tokens':0,
                          'cut':False,'think_cut':False,'think_eos':False})
    summary=summarize('thinking',ids,specs,qs,store,tmp_path)
    interactions=[r for r in summary['contrasts'] if r['primary']]
    assert len(interactions)==2
    assert all(r['mean']==-1 and r['sources']==1 for r in interactions)
    assert all('p_holm_primary_interactions' in r for r in interactions)
