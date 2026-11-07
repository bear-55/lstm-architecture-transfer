"""Isolated EA training using the existing frozen sampler, loss and fit loop."""
import argparse
import copy
import json
import numpy as np
import torch
import formal
import retraining_confirmation as cohort
from ea_model import make_ea
from graph_backend import attach_graphs
from project import ROOT, digest, save_json
from train import fit, identity_for
from training import setup_device, optimizer_for

CONFIG = ROOT/'configs/ea_extension_20261031.json'
OUT = ROOT/'runs/ea_extension_20261031'
NEW_FILES = ('src/ea_model.py','src/ea_extension.py','src/ea_checks.py',
    'cloud/run_ea_extension.py','configs/ea_extension_20261031.json','protocol/ea_extension_20261031.json',
    'analysis/multiseed_selection.py','protocol/multiseed_selection_20261031.json')


def load():
    cfg, meta, old = cohort.load()
    spec = json.loads(CONFIG.read_text(encoding='utf-8'))
    if (spec['model']!='EA' or spec['formal_seeds'] != old['new_seeds'] or spec['formal_runs']!=10
        or type(spec['development_seed']) is not int
        or not 0 <= spec['development_seed'] < 2**31-100000
        or spec['development_seed'] in set(old['new_seeds'])|set(cohort.OLD_SEEDS)):
        raise RuntimeError('Invalid EA seed roster')
    cfg = copy.deepcopy(cfg)
    cfg['model'].update(models={'EA':'ea_static_input_gate'}, copy_common_initial_backbone=False,
        initialization=spec['initialization'], architecture=spec['architecture'])
    cfg['evaluation']['primary_summary'] = 'EA extension; retain all ten validation-selected checkpoints'
    cfg['protocol'] = spec['study']
    return cfg, meta, spec


def provenance():
    return dict(cohort.provenance(), **{n:digest(ROOT/n) for n in NEW_FILES})


def run_path(seed, development=False):
    return OUT/('development' if development else 'EA')/f'seed_{seed}'


def identity(seed):
    return identity_for('EA', seed, provenance(), False, 'cuda_graph')


def verify(seed, development=False, env=None):
    cfg, _, spec = load()
    if (development and seed!=spec['development_seed']) or (not development and seed not in spec['formal_seeds']):
        raise ValueError('Unexpected seed/stage')
    p = run_path(seed, development)
    return cohort.check_existing(p,'EA',seed,cfg,provenance(),identity(seed),env)


def run(seed, development=False):
    cfg, meta, spec = load()
    if (development and seed!=spec['development_seed']) or (not development and seed not in spec['formal_seeds']):
        raise ValueError('Unexpected seed/stage')
    cohort.check_versions()
    device = setup_device('cuda')
    env, hashes, key = cohort.runtime(device), provenance(), identity(seed)
    p = run_path(seed,development)
    if p.exists():
        if verify(seed,development,env):
            print(f'EA {seed} already completed and verified', flush=True)
            return
    else:
        p.mkdir(parents=True)
        save_json(p/'run.json',dict(identity=key,provenance=hashes,model='EA',seed=seed,
            backend='cuda_graph',protocol=cfg,environment=env,cohort=spec,development=development))
    tr, va, tb, vb = formal.datasets(meta,device)
    model = make_ea(cfg,meta,seed,device)
    attach_graphs(model,tb,vb,len(tr),len(va),cfg['training']['batch_size'])
    print(f'TRAIN EA seed={seed}; development={development}; parameters={sum(p.numel() for p in model.parameters())}',flush=True)
    fit(cfg,meta,model,optimizer_for(model,cfg),tr,va,tb,vb,p,key,seed,device,resume=(p/'last.pt').exists())
    cohort.validate_completed(p,key)
    cohort.seal_completed(p)
    print(f'EA_COMPLETED seed={seed}; no Test evaluation',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--development',action='store_true')
    a = parser.parse_args()
    run(a.seed,a.development)
