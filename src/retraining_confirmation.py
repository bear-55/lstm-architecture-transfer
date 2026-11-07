"""Independent B0/M1 cohort; unchanged formal model, sampler and fit loop."""
import argparse
import copy
import json
import platform
from pathlib import Path

import numpy as np
import torch
import formal
from graph_backend import attach_graphs
from project import ROOT, digest, save_json
from train import fit, identity_for
from training import environment, make_model, optimizer_for, setup_device

CONFIG = ROOT / 'configs/retraining_confirmation.json'
OUT = ROOT / 'runs/retraining_confirmation_20261029'
MODELS = ('B0', 'M1')
OLD_SEEDS = (11, 29, 47, 65, 83, 103, 119, 137, 155, 173)
REQUIRED = ('run.json', 'result.json', 'best.pt', 'last.pt', 'history.csv',
            'validation.json', 'metrics_per_basin.csv', 'predictions_val.npz')


def amend(base, spec):
    seeds = spec['new_seeds']
    if (spec['models'] != list(MODELS) or len(seeds) != 10 or len(set(seeds)) != 10
            or any(type(s) is not int or not 0 <= s < 2**31 - 100000 for s in seeds)
            or set(seeds) & set(OLD_SEEDS) or spec['new_run_count'] != 20):
        raise ValueError('Expected B0/M1 with ten distinct NEW seeds, twenty runs')
    cfg = copy.deepcopy(base)
    cfg['training'].update(paired_seeds=seeds, confirmation_seeds=seeds, paired_seed_count=10)
    cfg['evaluation']['primary_summary'] = 'New cohort only; all ten validation-selected checkpoints retained'
    return cfg


def load():
    base, meta = formal.load_formal()
    spec = json.loads(CONFIG.read_text(encoding='utf-8'))
    for name, expected in spec['frozen_training_sources'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise RuntimeError('Frozen training source changed: ' + name)
    return amend(base, spec), meta, spec


def provenance():
    spec = json.loads(CONFIG.read_text(encoding='utf-8'))
    names = set(formal.provenance()) | set(spec['frozen_training_sources']) | {
        'configs/retraining_confirmation.json', 'src/retraining_confirmation.py',
        'cloud/run_retraining_confirmation.py', 'analysis/selection_transfer.py',
        'protocol/confirmation.json', 'protocol/env_spec.json'}
    return {name: digest(ROOT / name) for name in sorted(names)}


def runtime(device):
    out = environment(device)
    out.update(cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        out.update(compute_capability=[p.major, p.minor], total_memory=p.total_memory)
    return out


def check_versions():
    if str(torch.__version__) != '2.12.1+cu130' or np.__version__ != '2.3.2':
        raise RuntimeError('Expected torch 2.12.1+cu130 and numpy 2.3.2; do not auto-upgrade')
    if platform.python_version_tuple()[:2] != ('3', '12'):
        raise RuntimeError('Expected the restored Python 3.12 environment')


def tasks(spec):
    return [(m, s) for s in spec['new_seeds'] for m in MODELS]


def backend(model):
    return 'native_fused' if model == 'B0' else 'cuda_graph'


def check_existing(path, model, seed, cfg, hashes, identity, env=None):
    record = json.loads((path / 'run.json').read_text(encoding='utf-8'))
    if (record['identity'] != identity or record['provenance'] != hashes
            or record['model'] != model or record['seed'] != seed or record['protocol'] != cfg):
        raise RuntimeError('Existing run code/data/config/model/seed changed')
    if env is not None and record['environment'] != env:
        raise RuntimeError('Environment changed; refusing to mix training within a cohort')
    if (path / 'result.json').exists():
        result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
        if (result['status'] != 'completed' or result['identity'] != identity or result['test_q_read']
                or not all((path / f).is_file() for f in REQUIRED)):
            raise RuntimeError('Invalid or incomplete completed run')
        seal = path / 'completed_artifacts.json'
        if seal.exists():
            files = json.loads(seal.read_text())['files']
            if set(files) != set(REQUIRED):
                raise RuntimeError('Incomplete completed-artifact seal')
            for name, h in files.items():
                if name not in REQUIRED or digest(path / name) != h:
                    raise RuntimeError('Completed artifact changed: ' + name)
        else:
            # Completion may have preceded an interrupted final seal. Validate before recovery.
            validate_completed(path, identity)
            seal_completed(path)
        return True
    return False


def validate_completed(path, identity):
    import csv
    result = json.loads((path / 'result.json').read_text())
    best = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
    last = torch.load(path / 'last.pt', map_location='cpu', weights_only=True)
    if (best['identity'] != identity or last['identity'] != identity
            or best['epoch'] != result['best_epoch']
            or last['completed_epochs'] != result['completed_epochs']):
        raise RuntimeError('Checkpoint/result mismatch')
    with (path / 'history.csv').open(newline='') as f:
        history = list(csv.DictReader(f))
    if len(history) != result['completed_epochs'] or not np.isfinite(result['validation']['median_nse']):
        raise RuntimeError('Invalid training history or validation selection')


def seal_completed(path):
    save_json(path / 'completed_artifacts.json', {'files': {n: digest(path / n) for n in REQUIRED}})


def run(model_name, seed, resume=False):
    cfg, meta, spec = load()
    if model_name not in MODELS or seed not in spec['new_seeds']:
        raise ValueError('This entry accepts only the fixed NEW B0/M1 cohort')
    check_versions()
    device = setup_device('cuda')
    env, hashes = runtime(device), provenance()
    identity = identity_for(model_name, seed, hashes, False, backend(model_name))
    path = OUT / model_name / f'seed_{seed}'
    if path.exists():
        if not resume:
            raise RuntimeError('Existing run requires --resume')
        if check_existing(path, model_name, seed, cfg, hashes, identity, env):
            print(f'Already completed and verified: {model_name} seed={seed}', flush=True)
            return
    else:
        path.mkdir(parents=True)
        save_json(path / 'run.json', dict(identity=identity, provenance=hashes, model=model_name,
                  seed=seed, backend=backend(model_name), protocol=cfg, environment=env, cohort=spec))
    tr, va, tb, vb = formal.datasets(meta, device)
    model = make_model(cfg, meta, model_name, seed, device,
                       'native_fused' if model_name == 'B0' else 'reference_manual_loop')
    attach_graphs(model, tb, vb, len(tr), len(va), cfg['training']['batch_size'])
    print(f'TRAIN {model_name} seed={seed}; output={path}', flush=True)
    fit(cfg, meta, model, optimizer_for(model, cfg), tr, va, tb, vb, path, identity,
        seed, device, resume=(path / 'last.pt').exists())
    validate_completed(path, identity)
    seal_completed(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.model, args.seed, args.resume)
