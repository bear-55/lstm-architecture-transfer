import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ea_model import EALSTM,make_ea
import ea_extension as ea
from training import make_model,setup_device,WindowBatches,optimizer_for
from train import fit
from graph_checks import exact
from test_training import toy_data
from test_preparation import temporary_directory


@pytest.fixture
def ea_tmp():
    with temporary_directory() as path:
        yield path


def test_equations_and_gradients_against_stepwise_reference():
    torch.manual_seed(77)
    net=EALSTM(2,3,4,0).double()
    x=torch.randn(3,5,2,dtype=torch.double,requires_grad=True)
    s=torch.randn(3,3,dtype=torch.double,requires_grad=True)
    expected_i=(s@net.input_gate.weight.T+net.input_gate.bias).sigmoid()
    h=c=torch.zeros(3,4,dtype=torch.double)
    for t in range(5):
        terms=x[:,t]@net.dynamic_gates.weight.T+h@net.recurrent_gates.weight.T
        terms=terms+net.dynamic_gates.bias+net.recurrent_gates.bias
        f,g,o=terms.split(4,dim=1)
        c=f.sigmoid()*c+expected_i*g.tanh()
        h=o.sigmoid()*c.tanh()
    expected=F.softplus(h@net.head.weight.T+net.head.bias)
    actual,diag=net(x,s,return_diagnostics=True)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(diag['input_gate'],expected_i)
    inputs=(x,s,*net.parameters())
    ag=torch.autograd.grad(actual.sum(),inputs)
    eg=torch.autograd.grad(expected.sum(),inputs)
    for a,e in zip(ag,eg):
        torch.testing.assert_close(a,e,rtol=1e-10,atol=1e-12)


def test_static_only_input_gate_and_window_state_reset():
    net=EALSTM(2,3,4,0).eval()
    x,s=torch.randn(2,4,2),torch.randn(2,3)
    pred,diag=net(x,s,return_diagnostics=True)
    _,other=net(x*4,s,return_diagnostics=True)
    torch.testing.assert_close(diag['input_gate'],other['input_gate'],rtol=0,atol=0)
    net(torch.randn_like(x),torch.randn_like(s))
    torch.testing.assert_close(pred,net(x,s,torch.randn(2,17)),rtol=0,atol=0)
    with torch.no_grad():
        net.input_gate.weight.zero_()
    torch.testing.assert_close(net(x,s),net(x,s+20),rtol=0,atol=0)


def test_initialization_information_and_budget_match():
    cfg,meta,spec=ea.load()
    import retraining_confirmation as cohort
    basecfg,_,_=cohort.load()
    assert cfg['training']==basecfg['training']
    assert cfg['features']==basecfg['features']
    assert spec['development_seed'] not in spec['formal_seeds']
    assert meta['test_q_in_cache'] is False
    net=make_ea(cfg,meta,1009,torch.device('cpu'))
    b0=make_model(basecfg,meta,'B0',1009,torch.device('cpu'))
    h=cfg['model']['hidden_size']; d=len(meta['dynamic_names'])
    exact(net.input_gate.weight,b0.core.weight_ih_l0[:h,d:])
    exact(net.dynamic_gates.weight,b0.core.weight_ih_l0[h:,:d])
    exact(net.recurrent_gates.weight,b0.core.weight_hh_l0[h:])
    exact(net.head.state_dict(),b0.head.state_dict())
    assert sum(p.numel() for p in net.parameters())==15105


def test_epoch_resume_matches_full_training(ea_tmp):
    tmp_path=ea_tmp
    cfg,meta,tr,va=toy_data(tmp_path)
    device=setup_device('cpu')
    tb,vb=WindowBatches(tr,device),WindowBatches(va,device)
    for name,pause in [('full',False),('resume',True)]:
        dest=tmp_path/name; dest.mkdir()
        net=make_ea(cfg,meta,2027,device)
        fit(cfg,meta,net,optimizer_for(net,cfg),tr,va,tb,vb,dest,'ea-toy',2027,device,
            pause_after_epoch=1 if pause else None)
        if pause:
            net=make_ea(cfg,meta,2028,device)
            fit(cfg,meta,net,optimizer_for(net,cfg),tr,va,tb,vb,dest,'ea-toy',2027,device,resume=True)
    a,b=[torch.load(tmp_path/n/'last.pt',weights_only=True) for n in ('full','resume')]
    for key in ('model','best_model','optimizer','rng','best_epoch','best_score','bad_epochs'):
        exact(a[key],b[key])
    with np.load(tmp_path/'full'/'predictions_val.npz',allow_pickle=False) as a, np.load(tmp_path/'resume'/'predictions_val.npz',allow_pickle=False) as b:
        for key in a.files:
            np.testing.assert_array_equal(a[key],b[key])


@pytest.mark.parametrize('shape',[(0,4,2),(2,0,2),(2,4,3)])
def test_reject_wrong_dynamic_shape(shape):
    with pytest.raises(ValueError):
        EALSTM(2,3,4)(torch.empty(shape),torch.ones(2,3))
