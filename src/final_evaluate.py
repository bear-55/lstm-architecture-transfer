"""Final PUB evaluation: check all validation predictions before reading Test labels.

Usage: python -u -B src/final_evaluate.py check
       python -u -B src/final_evaluate.py run
No training, model selection, refitting, or changes to the development label reader.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import tarfile
import time
from datetime import date, datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from dataset import valid_window_ends
from evaluate import basin_metrics, summarize
from formal import load_formal
from graph_backend import GraphBackend
from project import ROOT, digest
from training import environment, make_model, predict_scaled, setup_device

RESULT_ARCHIVE = ROOT / 'runs/server backups/formal_pub_527_first3_results.tar.gz'
LABEL_ARCHIVE = ROOT / 'dist/final_test_labels_107.zip'
OUT = ROOT / 'runs/final_pub_527'
MODELS = ('B0', 'A1', 'M1')
PRED_RTOL, PRED_ATOL, NSE_ATOL = 1e-4, 1e-5, 1e-5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_csv(path, rows):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def save_predictions(path, records):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **records)
    temporary.replace(path)


def append_access(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time_utc': datetime.now(timezone.utc).isoformat(), **record}, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def parse_streamflow(text, basin, period, area_m2):
    """Parse only the frozen period; same units/flag rules as development labels."""
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError('Finite positive forcing-header area required')
    start, end = map(date.fromisoformat, period)
    if end < start:
        raise ValueError('Invalid evaluation period')
    dates = np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, 'D'))
    q = np.full(len(dates), np.nan, dtype=np.float32)
    previous = None
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6 or fields[0] != basin:
            raise ValueError('Unexpected streamflow schema/basin')
        day = date(*map(int, fields[1:4]))
        if previous is not None and day <= previous:
            raise ValueError('Duplicate or unordered streamflow dates')
        previous = day
        if day < start:
            continue
        if day > end:
            break
        flag = fields[5]
        if not (flag == 'M' or flag.startswith('A') or flag.startswith('P')):
            raise ValueError('Unreviewed streamflow flag: ' + flag)
        observed = float(fields[4])
        if flag == 'M' or not np.isfinite(observed) or observed < 0:
            continue
        q[(day - start).days] = observed * 0.028316846592 * 86400 * 1000 / area_m2
    return dates, q


class EvaluationWindows:
    """One immutable common mask. Only meteorology/static covariates enter batches."""
    def __init__(self, cache, meta, ids, period, q):
        self.ids = list(ids)
        self.dates = cache['dates']
        self.dynamic, self.static = cache['x_dynamic'], cache['x_static']
        self.land_indices = meta['land_indices']
        self.length = meta['sequence_length']
        require(q.shape == (len(ids), len(self.dates)), 'Label shape mismatch')
        all_ids = cache['basin_ids'].tolist()
        rows = [all_ids.index(b) for b in ids]
        ends = [valid_window_ends(cache['dynamic_valid'][row], self.dates, *period, self.length, q[j])
                for j, row in enumerate(rows)]
        require(all(len(e) > 0 for e in ends), 'A basin has no common valid dates')
        self.basins = np.repeat(np.arange(len(ids)), [len(e) for e in ends])
        self.ends = np.concatenate(ends)
        self.rows = np.asarray(rows)[self.basins]
        self.observed = q[self.basins, self.ends]
        self.lags = np.arange(1 - self.length, 1)

    def __len__(self):
        return len(self.ends)

    def batch(self, start, stop, device):
        rows, ends = self.rows[start:stop], self.ends[start:stop]
        dynamic = self.dynamic[rows[:, None], ends[:, None] + self.lags]
        static = self.static[rows]
        arrays = {'x_dynamic': dynamic, 'x_static': static, 'x_land': static[:, self.land_indices]}
        require(all(np.isfinite(x).all() for x in arrays.values()), 'Nonfinite evaluation inputs')
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.items()}

    def records(self, prediction):
        return {'basin_ids': np.asarray(self.ids)[self.basins], 'dates': self.dates[self.ends],
                'q_observed_mm_day': self.observed, 'q_predicted_mm_day': prediction}


def metric_rows(dataset, prediction, cfg):
    rows = []
    for j, basin in enumerate(dataset.ids):
        mask = dataset.basins == j
        rows.append({'basin_id': basin, **basin_metrics(dataset.observed[mask], prediction[mask],
                    min_days=cfg['evaluation']['min_valid_days'],
                    quantile=cfg['evaluation']['high_flow_quantile'])})
    return rows


def load_run(archive, cfg, meta, name, seed, device):
    prefix = f'{name}/seed_{seed}/'
    spec = json.load(archive.extractfile(prefix + 'run.json'))
    result = json.load(archive.extractfile(prefix + 'result.json'))
    require(spec['model'] == name and spec['seed'] == seed, 'Wrong model or seed')
    require(spec['protocol'] == cfg, 'Result metadata differs from current protocol')
    require(result['status'] == 'completed' and not result['test_q_read'], 'Incomplete/development result required')
    for path in ('src/model.py', 'src/training.py', 'src/evaluate.py', 'src/dataset.py',
                 'data/formal/formal.npz', 'data/formal/scaler.json', 'data/formal/metadata.json'):
        # Large cache hashes were checked once by build_context, below.
        if path.startswith('src/'):
            require(digest(ROOT / path) == spec['provenance'][path], 'Executed model/metric source changed: ' + path)
        else:
            require(spec['provenance'][path] == meta['_verified_hashes'][path], 'Executed cache changed: ' + path)
    checkpoint_data = archive.extractfile(prefix + 'best.pt').read()
    checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location='cpu', weights_only=True)
    require(checkpoint['identity'] == spec['identity'] == result['identity'], 'Checkpoint identity mismatch')
    require(checkpoint['epoch'] == result['best_epoch'], 'Not the validation-selected best checkpoint')
    model = make_model(cfg, meta, name, seed, device, 'native_fused' if name == 'B0' else 'reference_manual_loop')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    for value in model.state_dict().values():
        require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor')
    return model, result, hashlib.sha256(checkpoint_data).hexdigest()


@torch.no_grad()
def predict(model, dataset, cfg, meta, device, label):
    size = cfg['training']['batch_size']
    if model.adapter_mode != 'none' and device.type == 'cuda':
        samples = []
        for n in sorted({min(size, len(dataset)), len(dataset) % size} - {0}):
            batch = dataset.batch(0, n, device)
            samples.append(tuple(batch[k] for k in ('x_dynamic', 'x_static', 'x_land')))
        runner = GraphBackend(model, [], samples)
        model._graph_backend = runner
        model.forward = runner.forward
        for sample in samples:
            torch.testing.assert_close(model(*sample), runner.raw(*sample), rtol=1e-5, atol=1e-6)
    result = np.empty(len(dataset), dtype=np.float32)
    for start in range(0, len(dataset), size):
        stop = min(start + size, len(dataset))
        output = (predict_scaled(model, dataset.batch(start, stop, device)) * meta['q_ref_mm_day']).cpu().numpy()[:, 0]
        require(bool(np.isfinite(output).all() and (output >= 0).all()), 'Invalid prediction; never shrink the mask')
        result[start:stop] = output
        if start and (start // size) % 1000 == 0:
            print(f'{label}: {stop:,}/{len(dataset):,} windows', flush=True)
    return result


def compare_validation(records, saved, rows, expected_summary):
    for key in ('basin_ids', 'dates', 'q_observed_mm_day'):
        require(np.array_equal(records[key], saved[key]), 'Validation alignment mismatch: ' + key)
    a, b = records['q_predicted_mm_day'], saved['q_predicted_mm_day']
    require(bool(np.allclose(a, b, rtol=PRED_RTOL, atol=PRED_ATOL)),
            'Validation prediction mismatch: stop before opening Test labels')
    local = summarize(rows)
    for key in ('basins', 'valid_nse_basins', 'invalid_nse_basins', 'negative_nse_basins'):
        require(local[key] == expected_summary[key], 'Validation summary mismatch: ' + key)
    for key in ('median_nse', 'mean_nse'):
        require(abs(local[key] - expected_summary[key]) <= NSE_ATOL, 'Validation NSE mismatch: ' + key)
    return {'max_absolute_prediction_difference_mm_day': float(np.max(np.abs(a - b))),
            'local_validation': local, 'cloud_validation': expected_summary}


def build_context(device):
    cfg, meta = load_formal()
    require(cfg['training']['paired_seeds'] == cfg['training']['confirmation_seeds'] == [11, 29, 47], 'Expected three fixed seeds')
    require(cfg['training']['paired_seed_count'] == 3 and set(cfg['model']['models']) == set(MODELS), 'Unexpected experiment roster')
    require(cfg['evaluation']['final_protocol_frozen'], 'Final protocol must be frozen')
    require(not cfg['evaluation']['read_final_test_q'] and not cfg['evaluation']['formal_training_reads_test_q'],
            'Keep development label protection enabled; this is a separate evaluation entry')
    require(cfg['periods']['final_labels'] == ['2003-10-01', '2008-09-30'], 'Unexpected test period')
    require(len(meta['splits']['test']) == 107, 'Expected 107 frozen Test basins')
    paths = ['configs/formal.json', 'configs/formal_seed_revision.json', 'data/formal/formal.npz',
             'data/formal/metadata.json', 'data/formal/scaler.json', 'splits/formal_pub_527/test.txt',
             'data_contract/formal_531_audit/basin_audit.csv', 'data_contract/formal_531_audit/source_hashes.json',
             'src/final_evaluate.py', 'src/model.py', 'src/dataset.py', 'src/training.py',
             'src/evaluate.py', 'src/graph_backend.py', 'src/formal.py', 'src/project.py']
    hashes = {p: digest(ROOT / p) for p in paths}
    hashes['results_archive'] = digest(RESULT_ARCHIVE)
    hashes['labels_archive'] = digest(LABEL_ARCHIVE)
    require(hashes['labels_archive'] == LABEL_ARCHIVE.with_suffix('.zip.sha256').read_text().split()[0], 'Label archive checksum mismatch')
    meta['_verified_hashes'] = hashes
    with ZipFile(LABEL_ARCHIVE) as labels:
        sources = json.loads(labels.read('final_test_labels_107/source_manifest.json'))
        require(len(labels.namelist()) == len(set(labels.namelist())), 'Duplicate label archive members')
        require(len(sources) == 107 and {r['basin_id'] for r in sources} == set(meta['splits']['test']), 'Test source roster mismatch')
        require(all(r['role'] == 'test' for r in sources), 'Non-test labels in final source roster')
    context = {'protocol': cfg, 'hashes': hashes, 'environment': environment(device),
               'prediction_parity_tolerance': {'rtol': PRED_RTOL, 'atol_mm_day': PRED_ATOL, 'nse_atol': NSE_ATOL},
               'primary_summary': cfg['evaluation']['primary_summary'],
               'paired_summary': 'For each basin, average M1-minus-baseline NSE across the same three seeds; then summarize basins.',
               'selection': 'All nine validation-selected best.pt checkpoints; no Test-based selection',
               'common_mask': 'All 365 input days valid; observed target finite and nonnegative; fixed for every model and seed',
               'label_usage': 'Evaluation metrics only; no fit, scaling, initialization, or model inputs'}
    identity = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    with np.load(ROOT / 'data/formal/formal.npz', allow_pickle=False) as loaded:
        cache = {k: loaded[k] for k in ('basin_ids', 'dates', 'x_dynamic', 'x_static', 'dynamic_valid', 'q_val')}
    return cfg, meta, cache, context, identity, sources


def validation_check(cfg, meta, cache, device, context, identity):
    path = OUT / 'validation_recheck.json'
    if path.exists():
        saved = read_json(path)
        if saved['identity'] == identity and saved['status'] == 'passed' and len(saved['runs']) == 9:
            print('Matching full nine-run validation check already passed; reusing it.', flush=True)
            return saved
    # Refuse to mix a changed protocol into an existing final evaluation directory.
    if (OUT / 'test_protocol.json').exists():
        require(read_json(OUT / 'test_protocol.json')['identity'] == identity, 'Final evaluation identity changed; preserve existing outputs')
    dataset = EvaluationWindows(cache, meta, meta['splits']['val'], cfg['periods']['development_labels'], cache['q_val'])
    checks = []
    with tarfile.open(RESULT_ARCHIVE, 'r:gz') as archive:
        for name in MODELS:
            for seed in cfg['training']['paired_seeds']:
                label = f'Validation {name} seed={seed}'
                print(label + ': start', flush=True)
                model, result, checkpoint_hash = load_run(archive, cfg, meta, name, seed, device)
                started = time.perf_counter()
                prediction = predict(model, dataset, cfg, meta, device, label)
                rows = metric_rows(dataset, prediction, cfg)
                with np.load(io.BytesIO(archive.extractfile(f'{name}/seed_{seed}/predictions_val.npz').read()), allow_pickle=False) as p:
                    saved = {key: p[key] for key in p.files}
                check = compare_validation(dataset.records(prediction), saved, rows, result['validation'])
                checks.append({'model': name, 'seed': seed, 'best_epoch': result['best_epoch'],
                               'checkpoint_sha256': checkpoint_hash, 'prediction_count': len(dataset),
                               'seconds': time.perf_counter() - started, **check})
                print(label + f": passed, max difference={check['max_absolute_prediction_difference_mm_day']:.3g} mm/day", flush=True)
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    report = {'status': 'passed', 'identity': identity, 'context': context, 'runs': checks, 'test_labels_parsed': False}
    save_json(path, report)
    return report


def test_labels(cfg, meta, cache, sources, identity):
    ids = meta['splits']['test']
    with (ROOT / 'data_contract/formal_531_audit/basin_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
        qc = {r['basin_id']: r for r in csv.DictReader(stream)}
    source_audit = {r['basin_id']: r for r in read_json(ROOT / 'data_contract/formal_531_audit/source_hashes.json')['basins']}
    source_by_basin = {r['basin_id']: r for r in sources}
    q = np.full((len(ids), len(cache['dates'])), np.nan, dtype=np.float32)
    access_path = ROOT / cfg['evaluation']['final_test_access_log']
    with ZipFile(LABEL_ARCHIVE) as labels:
        for j, basin in enumerate(ids):
            source = source_by_basin[basin]
            path = source['path']
            require(path == f'raw/streamflow_test/{basin}_streamflow_qc.txt', 'Unexpected label archive path')
            append_access(access_path, {'event': 'read_started', 'evaluation_identity': identity, 'basin_id': basin,
                                       'role': 'test', 'period': cfg['periods']['final_labels'], 'purpose': 'frozen_final_evaluation'})
            data = labels.read('final_test_labels_107/' + path)
            checksum = hashlib.sha256(data).hexdigest()
            require(checksum == source['sha256'] == source_audit[basin]['q_sha256'], 'Test label source hash mismatch')
            # This audited field was computed directly from the Daymet header, not a GIS area.
            area = float(qc[basin]['forcing_area_km2']) * 1e6
            dates, values = parse_streamflow(data.decode('utf-8-sig'), basin, cfg['periods']['final_labels'], area)
            positions = np.searchsorted(cache['dates'], dates)
            require(bool((positions < len(cache['dates'])).all()) and np.array_equal(cache['dates'][positions], dates), 'Missing test dates in input cache')
            q[j, positions] = values
            append_access(access_path, {'event': 'read_completed', 'evaluation_identity': identity, 'basin_id': basin,
                                       'area_m2': area, 'valid_q_days': int(np.isfinite(values).sum()), 'source_sha256': checksum})
    dataset = EvaluationWindows(cache, meta, ids, cfg['periods']['final_labels'], q)
    counts = np.bincount(dataset.basins, minlength=len(ids))
    for j, basin in enumerate(ids):
        require(int(counts[j]) == int(qc[basin]['test_joint_valid_days']), 'Common date mask differs from prior audit: ' + basin)
    save_csv(OUT / 'common_mask_summary.csv', [{'basin_id': b, 'n_days': int(counts[j]),
             'first_valid_date': str(dataset.dates[dataset.ends[dataset.basins == j][0]]),
             'last_valid_date': str(dataset.dates[dataset.ends[dataset.basins == j][-1]])} for j, b in enumerate(ids)])
    return dataset


def aggregate(runs, rows_by_run, seeds):
    model_rows = []
    for name in MODELS:
        selected = [r for r in runs if r['model'] == name]
        require(len(selected) == len(seeds) and {r['seed'] for r in selected} == set(seeds), 'Incomplete seed roster')
        medians = [r['median_nse'] for r in selected]
        model_rows.append({'model': name, 'seed_count': len(seeds),
                           'mean_seed_median_nse': float(np.mean(medians)),
                           'sample_sd_seed_median_nse': float(np.std(medians, ddof=1)),
                           'mean_seed_mean_nse': float(np.mean([r['mean_nse'] for r in selected])),
                           'mean_seed_negative_nse_fraction': float(np.mean([r['negative_nse_fraction'] for r in selected]))})
    ids = [r['basin_id'] for r in rows_by_run['M1', seeds[0]]]
    require(all([r['basin_id'] for r in rows] == ids for rows in rows_by_run.values()), 'Unpaired basin order')
    paired_rows, paired_summary = [], []
    for i, basin in enumerate(ids):
        row = {'basin_id': basin}
        for baseline in ('B0', 'A1'):
            deltas = []
            for seed in seeds:
                a, b = rows_by_run['M1', seed][i]['nse'], rows_by_run[baseline, seed][i]['nse']
                value = a - b if a is not None and b is not None else None
                row[f'M1_minus_{baseline}_seed_{seed}'] = value
                deltas.append(value)
            row[f'M1_minus_{baseline}_mean_across_seeds'] = float(np.mean(deltas)) if all(v is not None for v in deltas) else None
        paired_rows.append(row)
    for baseline in ('B0', 'A1'):
        values = [r[f'M1_minus_{baseline}_mean_across_seeds'] for r in paired_rows]
        valid = [x for x in values if x is not None]
        require(bool(valid), 'No valid paired NSE basins')
        paired_summary.append({'comparison': f'M1-{baseline}', 'seed_count': len(seeds),
                               'valid_paired_basins': len(valid), 'invalid_paired_basins': len(values) - len(valid),
                               'median_basin_mean_delta_nse': float(np.median(valid)),
                               'mean_basin_mean_delta_nse': float(np.mean(valid)),
                               'improved_basins': sum(x > 0 for x in valid), 'worse_basins': sum(x < 0 for x in valid),
                               'tied_basins': sum(x == 0 for x in valid),
                               'improved_fraction': float(np.mean(np.asarray(valid) > 0))})
    return model_rows, paired_rows, paired_summary


def verify_artifacts(directory, hashes):
    for name, expected in hashes.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and path.is_file(), 'Missing evaluation artifact: ' + name)
        require(digest(path) == expected, 'Evaluation artifact changed: ' + name)


def run_test(cfg, meta, cache, device, context, identity, sources):
    frozen = OUT / 'test_protocol.json'
    if frozen.exists():
        require(read_json(frozen)['identity'] == identity, 'Existing final protocol differs; do not overwrite')
    else:
        save_json(frozen, {'identity': identity, 'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'context': context})
    completed = OUT / 'completed.json'
    if completed.exists():
        previous = read_json(completed)
        require(previous['identity'] == identity, 'Completed evaluation identity differs')
        verify_artifacts(OUT, previous['artifacts'])
        print('All nine final evaluations already completed and checksums verified: ' + str(OUT), flush=True)
        return
    # This is the first point where final labels are parsed, after all nine validation checks.
    print('All validation checks passed. Reading the 107 frozen Test label files.', flush=True)
    dataset = test_labels(cfg, meta, cache, sources, identity)
    runs, rows_by_run = [], {}
    with tarfile.open(RESULT_ARCHIVE, 'r:gz') as archive:
        for name in MODELS:
            for seed in cfg['training']['paired_seeds']:
                directory = OUT / name / f'seed_{seed}'
                directory.mkdir(parents=True, exist_ok=True)
                result_path = directory / 'result.json'
                if result_path.exists():
                    saved = read_json(result_path)
                    require(saved['identity'] == identity and saved['status'] == 'completed', 'Run identity mismatch')
                    verify_artifacts(directory, saved['artifacts'])
                    with np.load(directory / 'predictions_test.npz', allow_pickle=False) as p:
                        for key in ('basin_ids', 'dates', 'q_observed_mm_day'):
                            require(np.array_equal(p[key], dataset.records(p['q_predicted_mm_day'])[key]), 'Resumed evaluation mask mismatch')
                        prediction = p['q_predicted_mm_day'].copy()
                    rows = metric_rows(dataset, prediction, cfg)
                    require(summarize(rows) == saved['test'], 'Saved test summary mismatch')
                    print(f'Test {name} seed={seed}: completed output verified; skipped.', flush=True)
                else:
                    label = f'Test {name} seed={seed}'
                    print(label + ': start', flush=True)
                    model, source_result, checkpoint_hash = load_run(archive, cfg, meta, name, seed, device)
                    started = time.perf_counter()
                    prediction = predict(model, dataset, cfg, meta, device, label)
                    rows = metric_rows(dataset, prediction, cfg)
                    save_predictions(directory / 'predictions_test.npz', dataset.records(prediction))
                    save_csv(directory / 'metrics_per_basin.csv', rows)
                    saved = {'status': 'completed', 'identity': identity, 'model': name, 'seed': seed,
                             'best_epoch': source_result['best_epoch'], 'source_run_identity': source_result['identity'],
                             'checkpoint_sha256': checkpoint_hash, 'test': summarize(rows), 'test_q_read': True,
                             'test_period': cfg['periods']['final_labels'], 'prediction_count': len(dataset),
                             'evaluation_seconds': time.perf_counter() - started,
                             'artifacts': {f: digest(directory / f) for f in ('predictions_test.npz', 'metrics_per_basin.csv')}}
                    save_json(result_path, saved)
                    print(label + f": finished; median NSE={saved['test']['median_nse']:.6f}", flush=True)
                    del model
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                runs.append({'model': name, 'seed': seed, 'best_epoch': saved['best_epoch'], **saved['test']})
                rows_by_run[name, seed] = rows
    model_rows, paired_rows, paired_summary = aggregate(runs, rows_by_run, cfg['training']['paired_seeds'])
    for name, rows in (('run_summary.csv', runs), ('model_summary.csv', model_rows),
                       ('paired_basin_nse.csv', paired_rows), ('paired_summary.csv', paired_summary)):
        save_csv(OUT / name, rows)
    artifacts = {p.relative_to(OUT).as_posix(): digest(p) for p in OUT.rglob('*')
                 if p.is_file() and p.suffix in ('.json', '.csv', '.npz') and p.name != 'completed.json'}
    save_json(completed, {'status': 'completed', 'identity': identity, 'runs': len(runs),
                         'test_basins': len(dataset.ids), 'test_windows_per_run': len(dataset),
                         'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'artifacts': artifacts})
    print('Final evaluation completed. Summary: ' + str(OUT / 'model_summary.csv'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'run'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    device = setup_device(args.device)
    require(str(torch.__version__).split('+')[0] == '2.12.1', 'Use the installed PyTorch 2.12.1 environment')
    OUT.mkdir(parents=True, exist_ok=True)
    # OS-held lock releases automatically on exit, including interrupted evaluations.
    with (OUT / 'evaluation.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg, meta, cache, context, identity, sources = build_context(device)
        validation_check(cfg, meta, cache, device, context, identity)
        if args.action == 'run':
            run_test(cfg, meta, cache, device, context, identity, sources)
        else:
            print('CHECK PASSED: all nine full validation predictions reproduced. No Test labels parsed.', flush=True)


if __name__ == '__main__':
    main()
