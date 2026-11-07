"""New cohort identity, selection semantics and actual epoch-resume behavior."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import retraining_confirmation as c
from test_training import toy_data
from test_preparation import temporary_directory


def inputs():
    return json.loads((c.ROOT/'configs/formal.json').read_text()), json.loads(c.CONFIG.read_text())


def test_only_seed_roster_changes_and_old_data_validates():
    base,spec=inputs(); original=copy.deepcopy(base)
    cfg,meta,_=c.load()
    for k in ('paired_seeds','confirmation_seeds','paired_seed_count'):
        cfg['training'][k]=base['training'][k]
    cfg['evaluation']['primary_summary']=base['evaluation']['primary_summary']
    assert cfg==base==original
    jobs=c.tasks(spec)
    assert len(jobs)==len(set(jobs))==20
    assert not set(s for _,s in jobs)&set(c.OLD_SEEDS)
    assert len(meta['splits']['test'])==107 and not meta['test_q_in_cache']


@pytest.mark.parametrize('change',['old','duplicate','model','count'])
def test_invalid_new_cohort_rejected(change):
    base,spec=inputs()
    if change=='old':spec['new_seeds'][0]=11
    if change=='duplicate':spec['new_seeds'][1]=spec['new_seeds'][0]
    if change=='model':spec['models'].append('A1')
    if change=='count':spec['new_run_count']=21
    with pytest.raises(ValueError):c.amend(base,spec)


def test_old_seed_cannot_launch(monkeypatch):
    base,spec=inputs()
    monkeypatch.setattr(c,'load',lambda:(base,{},spec))
    monkeypatch.setattr(c,'check_versions',lambda:pytest.fail('old seed reached runtime'))
    with pytest.raises(ValueError):c.run('B0',11)


def test_real_fit_interruption_resume_completion_and_tamper(monkeypatch):
    from training import setup_device,WindowBatches
    with temporary_directory() as directory:
        root=Path(directory)
        cfg,meta,tr,va=toy_data(root)
        cfg=json.loads(json.dumps(cfg,default=str));cfg['training']['max_epochs']=2
        _,spec=inputs();device=setup_device('cpu')
        monkeypatch.setattr(c,'OUT',root/'new_cohort')
        monkeypatch.setattr(c,'load',lambda:(cfg,meta,spec))
        monkeypatch.setattr(c,'check_versions',lambda:None)
        monkeypatch.setattr(c,'setup_device',lambda _:device)
        monkeypatch.setattr(c,'provenance',lambda:{'frozen':'fixture'})
        monkeypatch.setattr(c.formal,'datasets',lambda *_:(tr,va,WindowBatches(tr,device),WindowBatches(va,device)))
        monkeypatch.setattr(c,'attach_graphs',lambda *a:None)
        original=c.fit
        def interrupted(*a,**kw):
            original(*a,**kw,pause_after_epoch=1)
            raise InterruptedError('simulated after saved first epoch')
        monkeypatch.setattr(c,'fit',interrupted)
        with pytest.raises(InterruptedError):c.run('B0',1009)
        path=c.OUT/'B0/seed_1009'
        assert (path/'last.pt').exists() and not (path/'result.json').exists()
        monkeypatch.setattr(c,'fit',original)
        c.run('B0',1009,resume=True)
        assert json.loads((path/'result.json').read_text())['completed_epochs']==2
        before={p.name:p.read_bytes() for p in path.iterdir()}
        monkeypatch.setattr(c,'fit',lambda *a,**kw:pytest.fail('completed job repeated'))
        c.run('B0',1009,resume=True)
        assert before=={p.name:p.read_bytes() for p in path.iterdir()}
        with (path/'predictions_val.npz').open('ab') as f:f.write(b'changed')
        with pytest.raises(RuntimeError,match='artifact changed'):c.run('B0',1009,resume=True)


def test_queue_stops_on_first_failed_child(monkeypatch):
    path=c.ROOT/'cloud/run_retraining_confirmation.py'
    module_spec=importlib.util.spec_from_file_location('confirmation_queue_test',path)
    queue=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(queue)
    launched=[]
    class Child:
        def wait(self):return 1
    def popen(*a,**kw):launched.append(a);return Child()
    with temporary_directory() as directory:
        monkeypatch.setattr(queue,'LOGS',Path(directory))
        monkeypatch.setattr(queue,'require_idle_gpu',lambda:None)
        monkeypatch.setattr(queue,'check_ready',lambda:None)
        monkeypatch.setattr(queue.subprocess,'Popen',popen)
        base,spec=inputs()
        with pytest.raises(RuntimeError,match='queue stopped'):queue.run_queue(base,spec)
        assert len(launched)==1


def test_known_signal_and_fixed_baseline_definitions():
    sys.path.insert(0,str(c.ROOT/'analysis'))
    from selection_transfer import summarize,details,weighted
    q=np.full((10,20),.03)
    r=summarize(q,q,np.repeat([0,1,2,3],5),100)
    assert r['other_vs_B0']['estimate']==pytest.approx(.03)
    assert r['other_vs_M1']['estimate']==pytest.approx(0)
    assert r['D_same_minus_other']['estimate']==pytest.approx(0)
    p=np.zeros_like(q)
    r=summarize(p,q,np.repeat([0,1,2,3],5),100)
    assert r['same_vs_B0']['estimate']==0  # ties choose B0
    assert r['same_vs_M1']['estimate']==pytest.approx(-.03)
    with pytest.raises(ValueError):weighted(q,q,np.r_[10,np.zeros(9)],np.ones(20))
