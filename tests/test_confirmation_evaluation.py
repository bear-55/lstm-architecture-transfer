import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import confirmation_evaluate as e
import retraining_confirmation as cohort
from training import setup_device


def fixture_records():
    dates = np.arange('2000-01-01', '2000-01-11', dtype='datetime64[D]')
    cache = dict(basin_ids=np.array(['a','b']), dates=dates, x_dynamic=np.ones((2,10,1), dtype='float32'),
                 x_static=np.ones((2,1), dtype='float32'), dynamic_valid=np.ones((2,10), dtype=bool))
    meta = dict(splits={'test':['a','b']}, sequence_length=3, land_indices=[0])
    records = dict(basin_ids=np.array(['a']*3+['b']*3), dates=np.tile(dates[[2,4,7]],2),
                   q_observed_mm_day=np.arange(6,dtype='float32'))
    return cache, meta, ['2000-01-03','2000-01-10'], records


def test_common_mask_and_labels_never_in_inputs():
    cache, meta, period, records = fixture_records()
    ds = e.dataset_from_records(cache, meta, period, records)
    assert len(ds) == 6
    assert set(ds.batch(0,2,torch.device('cpu'))) == {'x_dynamic','x_static','x_land'}
    p = ds.records(np.ones(6,dtype='float32'))
    assert np.array_equal(e.check_prediction(ds,p),np.ones(6))
    p['dates'][0] = np.datetime64('1999-01-01')
    with pytest.raises(RuntimeError,match='mask differs'):
        e.check_prediction(ds,p)


@pytest.mark.parametrize('error', ['duplicate','unknown_basin','outside_period','nan','reordered'])
def test_reject_changed_mask(error):
    cache, meta, period, records = fixture_records()
    if error == 'duplicate': records['dates'][1] = records['dates'][0]
    if error == 'unknown_basin': records['basin_ids'][0] = 'c'
    if error == 'outside_period': period[0] = '2000-01-05'
    if error == 'nan': records['q_observed_mm_day'][0] = np.nan
    if error == 'reordered': records = {k:v[::-1].copy() for k,v in records.items()}
    with pytest.raises(RuntimeError): e.dataset_from_records(cache,meta,period,records)


@pytest.mark.parametrize('name',['B0','M1'])
def test_new_checkpoint_cpu_validation_witness(name):
    cfg, meta, spec = cohort.load()
    device = setup_device('cpu')
    with np.load(ROOT/'data/formal/formal.npz',allow_pickle=False) as z:
        cache = {k:z[k] for k in z.files}
    full = e.base.EvaluationWindows(cache,meta,meta['splits']['val'],cfg['periods']['development_labels'],cache['q_val'])
    small = copy.copy(full)
    for k in ('basins','ends','rows','observed'):
        setattr(small,k,getattr(full,k)[:65])
    network,source,checksum = e.load_network(cfg,meta,name,spec['new_seeds'][0],device)
    pred = e.base.predict(network,small,cfg,meta,device,'CPU witness')
    with np.load(cohort.OUT/name/'seed_1009'/'predictions_val.npz',allow_pickle=False) as z:
        for k in ('basin_ids','dates','q_observed_mm_day'):
            np.testing.assert_array_equal(small.records(pred)[k],z[k][:65])
        np.testing.assert_allclose(pred,z['q_predicted_mm_day'][:65],rtol=e.base.PRED_RTOL,atol=e.base.PRED_ATOL)
    assert network.training is False
