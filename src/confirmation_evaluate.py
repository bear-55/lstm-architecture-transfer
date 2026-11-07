"""Separate, resumable inference of the fixed twenty completed B0/M1 runs."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

import final_evaluate as base
import retraining_confirmation as cohort
from evaluate import summarize
from project import ROOT, digest
from train import identity_for
from training import make_model, setup_device

OUT = ROOT / 'runs/confirmation_evaluation_20261030'
BUNDLE = ROOT / 'confirmation_evaluation_bundle.json'
DATA = ROOT / 'data/confirmation_evaluation_20261030'
PERIODS = ('earlier', 'later')


def verify_bundle():
    manifest = base.read_json(BUNDLE)
    for name, expected in {**manifest['files'], **manifest['shared_sources']}.items():
        path = (ROOT / name).resolve()
        base.require(path.is_relative_to(ROOT) and digest(path) == expected, 'Evaluation input changed: ' + name)
    return manifest


def prepare(device=None):
    manifest = verify_bundle()
    cfg, meta, spec = cohort.load()
    seal = base.read_json(cohort.OUT / 'cohort_completed.json')
    provenance = cohort.provenance()
    base.require(seal['status'] == 'completed' and seal['runs'] == 20 and not seal['test_scored']
                 and seal['new_seeds'] == spec['new_seeds'] and seal['paired_sampling_checked']
                 and seal['provenance'] == provenance, 'Complete sealed training batch required')
    expected = {f'{m}/seed_{s}/{n}' for m, s in cohort.tasks(spec)
                for n in cohort.REQUIRED + ('completed_artifacts.json',)}
    base.require(set(seal['files']) == expected, 'Unexpected training artifacts')
    base.verify_artifacts(cohort.OUT, seal['files'])
    for model, seed in cohort.tasks(spec):
        path = cohort.OUT / model / f'seed_{seed}'
        identity = identity_for(model, seed, provenance, False, cohort.backend(model))
        base.require(cohort.check_existing(path, model, seed, cfg, provenance, identity), 'Missing training result')
        cohort.validate_completed(path, identity)
    protocol = base.read_json(ROOT / 'protocol/confirmation.json')
    base.require(manifest['periods']['earlier'] == protocol['selection_period'] and
                 manifest['periods']['later'] == protocol['evaluation_period'], 'Period differs from frozen protocol')
    if device is None:
        return cfg, meta, spec, manifest
    cohort.check_versions()
    context = dict(bundle_sha256=digest(BUNDLE), cohort_sha256=digest(cohort.OUT / 'cohort_completed.json'),
                   environment=cohort.runtime(device), protocol_sha256=digest(ROOT / 'protocol/confirmation.json'))
    identity = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    path = OUT / 'evaluation_context.json'
    if path.exists():
        base.require(base.read_json(path) == dict(identity=identity, context=context),
                     'Inference context changed; preserve existing outputs')
    else:
        base.save_json(path, dict(identity=identity, context=context))
    with np.load(ROOT / 'data/formal/formal.npz', allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    return cfg, meta, spec, manifest, cache, identity


def load_network(cfg, meta, name, seed, device):
    path = cohort.OUT / name / f'seed_{seed}'
    result = base.read_json(path / 'result.json')
    best = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
    base.require(best['identity'] == result['identity'] and best['epoch'] == result['best_epoch'],
                 'Best checkpoint identity/epoch mismatch')
    network = make_model(cfg, meta, name, seed, device,
                         'native_fused' if name == 'B0' else 'reference_manual_loop')
    network.load_state_dict(best['model'], strict=True)
    base.require(all(bool(torch.isfinite(v).all()) for v in network.state_dict().values()), 'Nonfinite weights')
    network.eval()
    return network, result, digest(path / 'best.pt')


def release(network):
    if hasattr(network, '_graph_backend'):
        network.forward = network._graph_backend.raw
        del network._graph_backend


def check_validation(cfg, meta, spec, cache, device, identity):
    val = base.EvaluationWindows(cache, meta, meta['splits']['val'],
                                 cfg['periods']['development_labels'], cache['q_val'])
    checks = []
    for name, seed in cohort.tasks(spec):
        path = OUT / 'validation_checks' / name / f'seed_{seed}.json'
        checkpoint_hash = digest(cohort.OUT / name / f'seed_{seed}' / 'best.pt')
        if path.exists():
            record = base.read_json(path)
            base.require(record['identity'] == identity and record['status'] == 'passed'
                         and record['checkpoint_sha256'] == checkpoint_hash and record['model'] == name
                         and record['seed'] == seed and record['prediction_count'] == len(val), 'Validation resume mismatch')
        else:
            started = time.perf_counter()
            network, source, checksum = load_network(cfg, meta, name, seed, device)
            pred = base.predict(network, val, cfg, meta, device, f'Validation {name}/{seed}')
            with np.load(cohort.OUT / name / f'seed_{seed}' / 'predictions_val.npz', allow_pickle=False) as z:
                saved = {k: z[k] for k in z.files}
            record = base.compare_validation(val.records(pred), saved, base.metric_rows(val, pred, cfg), source['validation'])
            record.update(status='passed', identity=identity, checkpoint_sha256=checksum, model=name, seed=seed,
                          prediction_count=len(val), seconds=time.perf_counter()-started)
            base.save_json(path, record)
            release(network)
            del network
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        checks.append(record)
        print(f'Validation verified {name}/{seed}', flush=True)
    base.save_json(OUT / 'validation_recheck.json', dict(identity=identity, status='passed', runs=checks))
    return checks


def dataset_from_records(cache, meta, period, records):
    """Restore the exact historical common mask; labels never enter model.batch."""
    ids, dates = meta['splits']['test'], cache['dates']
    base.require(set(records) == {'basin_ids', 'dates', 'q_observed_mm_day'}, 'Only observation records allowed')
    base.require(all(a.ndim == 1 for a in records.values()) and
                 len({len(a) for a in records.values()}) == 1, 'Record lengths/shapes differ')
    base.require(set(records['basin_ids']) == set(ids), 'Test basin roster differs')
    base.require(np.isfinite(records['q_observed_mm_day']).all() and
                 (records['q_observed_mm_day'] >= 0).all(), 'Invalid observations')
    q = np.full((len(ids), len(dates)), np.nan, dtype=np.float32)
    for j, basin in enumerate(ids):
        mask = records['basin_ids'] == basin
        selected = records['dates'][mask]
        positions = np.searchsorted(dates, selected)
        base.require(bool((positions < len(dates)).all()), 'Label dates beyond cache')
        base.require(np.array_equal(dates[positions], selected) and len(np.unique(positions)) == len(positions),
                     'Missing/duplicate label dates')
        q[j, positions] = records['q_observed_mm_day'][mask]
    dataset = base.EvaluationWindows(cache, meta, ids, period, q)
    expected = dataset.records(np.zeros(len(dataset), dtype=np.float32))
    for key in records:
        base.require(np.array_equal(expected[key], records[key]), 'Observation mask/order differs: ' + key)
    return dataset


def check_prediction(dataset, records):
    p = records['q_predicted_mm_day']
    base.require(p.shape == (len(dataset),) and bool(np.isfinite(p).all() and (p >= 0).all()), 'Invalid predictions')
    expected = dataset.records(p)
    base.require(set(expected) == set(records), 'Unexpected prediction columns')
    for key in expected:
        base.require(np.array_equal(expected[key], records[key]), 'Prediction mask differs: ' + key)
    return p


def evaluate(cfg, meta, spec, manifest, cache, device, identity, checks):
    base.require(len(checks) == 20 and all(c['status'] == 'passed' for c in checks), 'Validate all 20 before Test')
    completed = OUT / 'completed.json'
    if completed.exists():
        saved = base.read_json(completed)
        base.require(saved['identity'] == identity and saved['evaluations'] == 40, 'Completion identity differs')
        base.verify_artifacts(OUT, saved['artifacts'])
        print('ALL_40_PERIOD_PREDICTIONS_COMPLETED_AND_VERIFIED', flush=True)
        return
    results = []
    for period in PERIODS:
        label_path = DATA / f'{period}_observations.npz'
        with np.load(label_path, allow_pickle=False) as z:
            records = {k: z[k] for k in z.files}
        dataset = dataset_from_records(cache, meta, manifest['periods'][period], records)
        base.require(len(dataset) == manifest['observations'][period]['count'], 'Observation count differs')
        for name, seed in cohort.tasks(spec):
            directory = OUT / period / name / f'seed_{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / 'result.json'
            checksum = digest(cohort.OUT / name / f'seed_{seed}' / 'best.pt')
            if path.exists():
                result = base.read_json(path)
                base.require(result['identity'] == identity and result['status'] == 'completed' and
                             result['checkpoint_sha256'] == checksum and result['model'] == name and
                             result['seed'] == seed and result['period'] == period, 'Inference resume mismatch')
                base.verify_artifacts(directory, result['artifacts'])
                with np.load(directory / 'predictions_test.npz', allow_pickle=False) as z:
                    prediction = check_prediction(dataset, {k: z[k] for k in z.files})
                base.require(summarize(base.metric_rows(dataset, prediction, cfg)) == result['summary'],
                             'Resumed metrics differ')
            else:
                started = time.perf_counter()
                network, source, current_hash = load_network(cfg, meta, name, seed, device)
                base.require(current_hash == checksum, 'Checkpoint changed')
                prediction = base.predict(network, dataset, cfg, meta, device, f'{period} {name}/{seed}')
                rows = base.metric_rows(dataset, prediction, cfg)
                base.save_predictions(directory / 'predictions_test.npz', dataset.records(prediction))
                base.save_csv(directory / 'metrics_per_basin.csv', rows)
                result = dict(status='completed', identity=identity, model=name, seed=seed, period=period,
                    checkpoint_sha256=checksum, best_epoch=source['best_epoch'], prediction_count=len(dataset),
                    seconds=time.perf_counter()-started, summary=summarize(rows),
                    artifacts={n: digest(directory / n) for n in ('predictions_test.npz', 'metrics_per_basin.csv')})
                base.save_json(path, result)
                release(network)
                del network
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            results.append(dict(period=period, model=name, seed=seed, best_epoch=result['best_epoch'], **result['summary']))
            print(f'Prediction complete {period} {name}/{seed} ({len(results)}/40)', flush=True)
    base.save_csv(OUT / 'run_summary.csv', results)
    artifacts = {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob('*')
                 if p.is_file() and p.suffix in ('.csv', '.json', '.npz') and p != completed}
    base.save_json(completed, dict(identity=identity, status='completed', evaluations=40, basins=107, artifacts=artifacts))
    print('ALL_40_PERIOD_PREDICTIONS_COMPLETED_AND_VERIFIED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'check', 'run'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.action == 'plan':
        cfg, meta, spec, manifest = prepare()
        print(json.dumps(dict(training_runs=20, validation_rechecks=20, period_evaluations=40,
            test_basins=len(meta['splits']['test']), periods=manifest['periods'], training=False,
            test_scored=False, output=str(OUT)), indent=2))
        return
    if args.device == 'cuda':
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        base.require(not active.strip(), 'Another GPU compute process is active; do not overlap jobs')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'evaluation.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.tell() == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = setup_device(args.device)
        cfg, meta, spec, manifest, cache, identity = prepare(device)
        checks = check_validation(cfg, meta, spec, cache, device, identity)
        if args.action == 'run':
            evaluate(cfg, meta, spec, manifest, cache, device, identity, checks)
        else:
            print('ALL_20_VALIDATION_RECHECKS_PASSED; no Test inference', flush=True)


if __name__ == '__main__':
    main()
