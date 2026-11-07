"""EA actual-shape graph/eager and exact optimizer/RNG replay witness."""
import argparse
import copy
import math
import time
from pathlib import Path
import numpy as np
import torch
import formal
import ea_extension as ea
import retraining_confirmation as cohort
from ea_model import make_ea
from graph_backend import attach_graphs
from graph_checks import exact
from project import save_json
from training import setup_device, rng_state, restore_rng, optimizer_for, train_step, cpu_weights


def witness(output):
    cohort.check_versions()
    cfg, meta, spec = ea.load()
    device = setup_device('cuda')
    if torch.cuda.device_count()!=1:
        raise RuntimeError('Expose exactly one GPU')
    torch.manual_seed(0)
    x = torch.randn(32,32,device=device)
    if not torch.isfinite(x@x).all().item():
        raise RuntimeError('CUDA kernel witness failed')
    print('EA_CUDA_KERNEL_WITNESS_OK',flush=True)
    tr, va, tb, vb = formal.datasets(meta,device)
    size = cfg['training']['batch_size']
    def new(seed):
        net = make_ea(cfg,meta,seed,device)
        state = rng_state()
        attach_graphs(net,tb,vb,len(tr),len(va),size)
        exact(state,rng_state())
        return net
    net = new(spec['development_seed'])
    eager = make_ea(cfg,meta,spec['development_seed'],device)
    opt, eo = optimizer_for(net,cfg), optimizer_for(eager,cfg)
    errors = []
    for n in sorted({size,len(tr)%size}-{0}):
        values = tb(np.arange(n))
        state = rng_state()
        a = train_step(net,opt,values,cfg,meta['q_ref_mm_day'])
        restore_rng(state)
        b = train_step(eager,eo,values,cfg,meta['q_ref_mm_day'])
        np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
        for (k,p), (k2,ep) in zip(net.named_parameters(),eager.named_parameters()):
            assert k==k2 and p.grad is not None and ep.grad is not None
            torch.testing.assert_close(p.grad,ep.grad,rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(p,ep,rtol=1e-5,atol=1e-6)
            errors.append(float((p.detach()-ep.detach()).abs().max()))
    saved = dict(model=cpu_weights(net),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state())
    values = tb(np.arange(size))
    a = train_step(net,opt,values,cfg,meta['q_ref_mm_day'])
    expected = dict(model=cpu_weights(net),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state())
    resumed = new(spec['development_seed']+1)
    resumed.load_state_dict(saved['model'])
    ro = optimizer_for(resumed,cfg)
    ro.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'])
    exact(a,train_step(resumed,ro,values,cfg,meta['q_ref_mm_day']))
    exact(expected,dict(model=cpu_weights(resumed),optimizer=ro.state_dict(),rng=rng_state()))
    resumed.eval()
    with torch.no_grad():
        for n in sorted({size,len(va)%size}-{0}):
            values = vb(np.arange(n))
            args = tuple(values[k] for k in ('x_dynamic','x_static','x_land'))
            torch.testing.assert_close(resumed(*args),resumed._graph_backend.raw(*args),rtol=1e-5,atol=1e-6)
    times = {}
    for role,batches in [('train',tb),('validation',vb)]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(20):
            values = batches(np.arange(step*size,(step+1)*size))
            if role=='train':
                train_step(resumed,ro,values,cfg,meta['q_ref_mm_day'])
            else:
                resumed.eval()
                with torch.no_grad():
                    resumed(values['x_dynamic'],values['x_static'],values['x_land']).cpu()
        torch.cuda.synchronize()
        times[role+'_step_seconds']=(time.perf_counter()-started)/20
    epoch_seconds=times['train_step_seconds']*math.ceil(len(tr)/size)+times['validation_step_seconds']*math.ceil(len(va)/size)
    record=dict(status='passed',provenance=ea.provenance(),environment=cohort.runtime(device),
        train_windows=len(tr),validation_windows=len(va),parameters=sum(p.numel() for p in net.parameters()),
        full_and_tail_gradients=True,dropout_rng_preserved=True,optimizer_rng_replay_exact=True,
        max_parameter_error=max(errors),test_scored=False,experiment_checkpoints_created=False,**times,
        estimated_epoch_minutes=epoch_seconds/60,estimated_30epoch_hours=epoch_seconds*30/3600)
    save_json(output,record)
    print('EA_PREFLIGHT_PASSED: full/tail gradients, validation, dropout and exact optimizer/RNG replay',flush=True)
    print(record,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    witness(p.parse_args().output)
