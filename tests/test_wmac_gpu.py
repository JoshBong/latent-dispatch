"""GPU scheduling checks supplied for operators; no model or tokenizer required."""
import json
import tarfile

import pytest

from latent_dispatch import wmac_plan
from scripts import run_wmac_gpu as gpu


def config():
    return json.loads(wmac_plan.CONFIG.read_text())


def test_gpu_assignments_are_disjoint_and_complete():
    cfg=config()
    t4=cfg['gpu_jobs']['t4']['jobs'];a100=cfg['gpu_jobs']['a100']['jobs']
    assert t4==['validation-4b','musique-4b']
    assert not set(t4)&set(a100)
    assert set(t4+a100)==set(wmac_plan.JOBS)
    assert all(wmac_plan.model_key(job)!='4b' for job in a100)
    assert sum(cfg['expected_main_rows'][j] for j in t4)==2496
    assert sum(cfg['expected_main_rows'][j] for j in a100)==5792


@pytest.mark.parametrize('target',['t4','a100'])
def test_commands_preserve_shared_inputs_and_only_select_assigned_components(tmp_path,target):
    prepared=tmp_path/'shared.json'
    args=gpu.arguments([target,'--prepared',str(prepared),'--output-root',str(tmp_path),'--smoke-only'])
    commands=list(gpu.commands(args,config()))
    assert [job for job,_ in commands]==config()['gpu_jobs'][target]['jobs']
    for job,cmd in commands:
        assert cmd[2]==job
        assert cmd[cmd.index('--prepared')+1]==str(prepared)
        assert '--smoke-only' in cmd
        assert '--dtype' not in cmd   # The child uses the frozen GPU profile.
        assert 'prepare' not in cmd


def test_failed_validation_stops_before_main_and_archives_only_its_gpu(monkeypatch,tmp_path):
    cfg=config();bundle={'config':cfg}
    prepared=tmp_path/'prepared.json';prepared.write_text(json.dumps(bundle))
    monkeypatch.setattr(wmac_plan,'load',lambda path:(bundle,{}))
    calls=[]
    class Process:
        def __init__(self,cmd,**kw): calls.append(cmd)
        def wait(self): return 1
    monkeypatch.setattr(gpu.subprocess,'Popen',Process)
    assert gpu.main(['t4','--prepared',str(prepared),'--output-root',str(tmp_path)])==1
    assert [cmd[2] for cmd in calls]==['validation-4b']
    status=json.loads((tmp_path/'wmac_gpu_v3/gpu-t4/status.json').read_text())
    assert status['status']=='failed'
    with tarfile.open(tmp_path/'wmac_gpu_v3-t4.tar.gz') as tar:
        assert 'wmac_gpu_v3/prepared.json' in tar.getnames()
        assert not any('gpu-a100' in name for name in tar.getnames())


def test_all_assigned_components_run_sequentially(monkeypatch,tmp_path):
    bundle={'config':config()}
    prepared=tmp_path/'prepared.json';prepared.write_text(json.dumps(bundle))
    monkeypatch.setattr(wmac_plan,'load',lambda path:(bundle,{}))
    events=[]
    class Process:
        def __init__(self,cmd,**kw): self.job=cmd[2];events.append(('start',self.job))
        def wait(self): events.append(('end',self.job));return 0
    monkeypatch.setattr(gpu.subprocess,'Popen',Process)
    assert gpu.main(['a100','--prepared',str(prepared),'--output-root',str(tmp_path)])==0
    assert events==[(phase,job) for job in config()['gpu_jobs']['a100']['jobs'] for phase in ('start','end')]
    assert (tmp_path/'wmac_gpu_v3-a100.tar.gz').is_file()


@pytest.mark.parametrize('argv',[
    ['t4','--max-hours','nan'],['a100','--max-hours','0'],['t4','--run-name','../outside'],
    ['t4','--smoke-only','--plan-only']])
def test_invalid_launch_options_are_rejected(argv):
    with pytest.raises(SystemExit): gpu.arguments(argv)
