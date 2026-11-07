import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import ea_evaluate as e
import ea_extension as ea
from training import setup_device
from test_preparation import temporary_directory

SEEDS=[1009,1021,1033,1051,1063,1087,1097,1109,1123,1151]


@pytest.mark.parametrize('seed',SEEDS)
def test_actual_ea_checkpoints_reproduce_validation_subset(seed):
    cfg,meta,spec=ea.load()
    device=setup_device('cpu')
    cache=e.read_npz(ROOT/'data/formal/formal.npz')
    full=e.base.EvaluationWindows(cache,meta,meta['splits']['val'],cfg['periods']['development_labels'],cache['q_val'])
    ds=copy.copy(full)
    # Spans a full batch plus tail, and the first/middle/final validation basins.
    indices=np.r_[np.arange(64),len(full)//2,np.arange(len(full)-64,len(full))]
    for key in ('basins','ends','rows','observed'):
        setattr(ds,key,getattr(ds,key)[indices])
    net,source,checksum=e.load_network(cfg,meta,seed,device)
    prediction=e.base.predict(net,ds,cfg,meta,device,'EA CPU witness')
    saved=e.read_npz(ea.run_path(seed)/'predictions_val.npz')
    for key in ('basin_ids','dates','q_observed_mm_day'):
        np.testing.assert_array_equal(ds.records(prediction)[key],saved[key][indices])
    np.testing.assert_allclose(prediction,saved['q_predicted_mm_day'][indices],rtol=e.base.PRED_RTOL,atol=e.base.PRED_ATOL)
    assert source['test_q_read'] is False and net.training is False


def test_validation_gate_rejects_duplicate_seed_or_wrong_identity():
    checks=[dict(seed=s,status='passed',identity='fixed') for s in SEEDS]
    e.require_checks(checks,SEEDS,'fixed')
    with pytest.raises(RuntimeError): e.require_checks(checks[:-1]+[checks[0]],SEEDS,'fixed')
    with pytest.raises(RuntimeError): e.require_checks(checks,SEEDS,'changed')


def test_wrong_checkpoint_identity_rejected(monkeypatch):
    cfg,meta,_=ea.load()
    first,second=ea.run_path(1009),ea.run_path(1021)
    with temporary_directory() as p:
        (p/'best.pt').write_bytes((first/'best.pt').read_bytes())
        (p/'result.json').write_bytes((second/'result.json').read_bytes())
        monkeypatch.setattr(ea,'run_path',lambda seed:p)
        with pytest.raises(RuntimeError,match='checkpoint identity'):
            e.load_network(cfg,meta,1021,torch.device('cpu'))
