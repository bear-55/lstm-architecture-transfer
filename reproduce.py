"""Portable entry points for the publication's archived experiment library."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
FIGURES = tuple(str(i) for i in range(1, 10)) + ('S1', 'S2', 'S3', 'S4')


def figure_id(value):
    return str(int(value)) if value.isdecimal() else value.upper()


def figure_folder(number):
    return Path('figures') / ('Fig' + number.zfill(2))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_assets(folder):
    catalog = json.loads((ROOT / 'release-assets.json').read_text(encoding='utf-8'))
    for asset in catalog['assets']:
        path = folder / asset['name']
        if not path.is_file():
            raise FileNotFoundError(f'Download {asset["name"]} to {folder}')
        assert path.stat().st_size == asset['bytes'], path.name
        assert digest(path) == asset['sha256'], path.name
        with zipfile.ZipFile(path) as archive:
            assert set(archive.namelist()) == {x['path'] for x in asset['files']}
            for item in asset['files']:
                payload = archive.read(item['path'])
                assert len(payload) == item['bytes']
                assert hashlib.sha256(payload).hexdigest() == item['sha256'], item['path']
        print(f'PASS asset: {path.name} ({len(asset["files"])} files)', flush=True)
    return catalog


def verify(assets=None):
    import numpy as np
    import pandas as pd

    manifest = json.loads((ROOT / 'provenance/files.sha256.json').read_text(encoding='utf-8'))
    for name, expected in manifest.items():
        assert digest(ROOT / name) == expected, f'File changed: {name}'
    cfg = json.loads((ROOT / 'configs/retraining_confirmation.json').read_text(encoding='utf-8'))
    for name, expected in cfg['frozen_training_sources'].items():
        assert digest(ROOT / name) == expected, f'Frozen training file changed: {name}'
    splits = {role: (ROOT / f'splits/formal_pub_527/{role}.txt').read_text().split()
              for role in ['train', 'val', 'test']}
    assert {role: len(ids) for role, ids in splits.items()} == {'train': 370, 'val': 50, 'test': 107}
    assert len(set(sum(splits.values(), []))) == 527
    array_path = ROOT / 'outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz'
    with np.load(array_path, allow_pickle=False) as data:
        models, seeds, basins = data['models'], data['seeds'], data['basins']
        assert models.tolist() == ['B0', 'M1', 'EA']
        assert seeds.tolist() == cfg['new_seeds']
        assert set(basins) == set(splits['test'])
        early, late = data['earlier'][..., 0], data['later'][..., 0]
    table = pd.read_csv(ROOT / 'outputs/jh_predraft_audit_20261102/main_NSE_endpoints.csv')
    checks = []
    for mi, model in enumerate(models[1:], 1):
        choices = early[mi] > early[0]
        difference = late[mi] - late[0]
        same = np.mean(choices * difference)
        independent = np.mean([choices[s] * difference[t]
                               for s in range(10) for t in range(10) if s != t])
        selected_global = (early[mi] - early[0]).mean(axis=1) > 0
        global_gain = np.mean([selected_global[s] * difference[t]
                              for s in range(10) for t in range(10) if s != t])
        for endpoint, value in {'same': same, 'other': independent, 'D': same-independent,
                                'global_other': global_gain, 'incremental_other': independent-global_gain}.items():
            reference = table[(table.comparison == model+'-B0') & (table.endpoint == endpoint)]
            assert len(reference) == 1
            np.testing.assert_allclose(value, reference.estimate.iloc[0], atol=1e-12, rtol=0)
        checks.append({'comparison':model+'-B0', 'G_same':same, 'G_new':independent, 'D':same-independent})
    for number in FIGURES:
        folder = ROOT / figure_folder(number)
        assert (folder / f'plot_figure{number}.py').is_file()
        assert (folder / 'caption_manuscript.txt').is_file()
        if number in ('1', 'S3'):
            continue
        report = json.loads((folder/'data_verification.json').read_text(encoding='utf-8'))
        assert report['status'].upper() == 'PASS'
        hashes = report.get('data_hashes', {'figure2_strata_skill.csv':report.get('data_file_sha256')})
        for name, expected in hashes.items():
            assert digest(folder/'data'/name) == expected, f'Figure {number}: {name}'
    figure8 = ROOT / figure_folder('8')
    forest = json.loads((figure8/'enhanced_data_verification.json').read_text(encoding='utf-8'))
    for name, expected in forest['input_sha256'].items():
        assert digest(ROOT/name) == expected, name
    with np.load(figure8/'data/forest_distributions.npz', allow_pickle=False) as draws:
        for row in forest['upper_panel']:
            key = row['key']
            values = draws[row['model'] + '__' + key]
            expected_count = {'k5': 20000, 'prespec': 5000, 'windows': 7}.get(key, 2000)
            assert len(values) == row['n'] == expected_count
            limits = [values.min(), values.max()] if key == 'windows' else np.quantile(values, [.025, .975])
            np.testing.assert_allclose(limits, [row['low'], row['high']], atol=1e-12, rtol=0)
    table_ids = ('1','2','3','4','S12','S2a','S2b','S3','S4a','S4b','S5a','S5b','S6','S7','S8','S9a','S9b','S10')
    assert {p.stem for p in (ROOT/'tables').glob('*.csv')} == {'Table_'+s for s in table_ids}
    if assets:
        verify_assets(assets)
    print(json.dumps({'status':'PASS', 'verified_files':len(manifest),
                      'frozen_training_files':len(cfg['frozen_training_sources']),
                      'main_point_estimates':checks, 'figures':FIGURES, 'paper_tables':len(table_ids)}, indent=2))


def stage(destination, all_inputs=True):
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT / 'work'):
        raise ValueError('Reproduction outputs must be under work/')
    destination.mkdir(parents=True, exist_ok=True)
    folders = ['figures', 'src'] if not all_inputs else [
        'scripts', 'analysis', 'src', 'configs', 'protocol', 'splits',
        'data_contract', 'data/attributes', 'outputs', 'deliverables', 'figures']
    for name in folders:
        src = ROOT / name
        for path in src.rglob('*'):
            if not path.is_file() or any(x in path.parts for x in ['qa', '__pycache__']):
                continue
            if path.suffix in {'.png','.tif','.svg','.pdf','.log'}:
                continue
            rel = path.relative_to(ROOT)
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    return destination


def run_script(workspace, relative):
    env = dict(os.environ, PYTHONUTF8='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    path = workspace / relative
    logs = workspace / 'logs'
    logs.mkdir(exist_ok=True)
    log = logs / (path.stem + '.log')
    print(f'RUN {relative}', flush=True)
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-X', 'utf8', str(path)], cwd=workspace,
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text(encoding='utf-8')[-6000:])
        raise RuntimeError(f'{relative} failed. See {log.relative_to(ROOT)}')
    print(f'PASS {relative}', flush=True)


def compare_table(reference, actual):
    import numpy as np
    import pandas as pd
    expected = pd.read_csv(reference, float_precision='round_trip')
    obtained = pd.read_csv(actual, float_precision='round_trip')
    assert list(expected.columns) == list(obtained.columns), reference.name
    assert expected.shape == obtained.shape, reference.name
    max_error = 0.0
    for col in expected:
        if pd.api.types.is_numeric_dtype(expected[col]):
            np.testing.assert_allclose(obtained[col], expected[col], rtol=1e-10, atol=1e-11, equal_nan=True, err_msg=col)
            finite = np.isfinite(expected[col]) & np.isfinite(obtained[col])
            if finite.any():
                max_error = max(max_error, float(np.max(np.abs(obtained.loc[finite,col].astype(float)-expected.loc[finite,col].astype(float)))))
        else:
            assert obtained[col].fillna('').equals(expected[col].fillna('')), col
    return {'file': reference.relative_to(ROOT).as_posix(), 'rows':len(expected), 'max_absolute_difference':max_error}


def stage_daily_predictions(workspace, assets, names):
    if assets is None:
        for name in names:
            source = ROOT/name
            if not source.is_file():
                raise FileNotFoundError(f'Missing {name}. Supply --assets PATH or extract the model attachment.')
            target = workspace/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return
    catalog = json.loads((ROOT/'release-assets.json').read_text(encoding='utf-8'))
    item = next(x for x in catalog['assets'] if x['name'].startswith('model-artifacts'))
    archive_path = assets/item['name']
    if digest(archive_path) != item['sha256']:
        raise ValueError(f'Checksum mismatch: {item["name"]}')
    with zipfile.ZipFile(archive_path) as archive:
        for name in names:
            target = workspace/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def prediction_names(models, periods):
    seeds = json.loads((ROOT/'configs/retraining_confirmation.json').read_text(encoding='utf-8'))['new_seeds']
    for period in periods:
        for model in models:
            directory = ('domain_matched_20261102' if period == 'earlier' else
                         'ea_evaluation_20261101' if model == 'EA' else 'confirmation_evaluation_20261030')
            for seed in seeds:
                yield f'runs/{directory}/{period}/{model}/seed_{seed}/predictions_test.npz'


def results(assets=None):
    workspace = stage(ROOT / 'work/analysis')
    for script in ['matching_scale_followup.py', 'multimetric_transfer.py',
                   'jh_predraft_baseline_audit.py', 'multiseed_budget_curve_20261104.py', 'selection_sensitivity.py']:
        run_script(workspace, 'scripts/' + script)
    names = [
        'outputs/selection_transfer_P1_20261102/P1_endpoints.csv',
        'outputs/selection_transfer_P1_20261102/matching_pairs.csv',
        'outputs/multimetric_transfer_20261102/transfer_endpoints.csv',
        'outputs/multimetric_transfer_20261102/transfer_per_basin.csv',
        'outputs/jh_predraft_audit_20261102/main_NSE_endpoints.csv',
        'outputs/multiseed_budget_curve_20261104/budget_curve_endpoints.csv',
        'outputs/multiseed_budget_curve_20261104/source_subset_results.csv',
        'outputs/multiseed_budget_curve_20261104/paired_budget_changes.csv',
        'tables/Table_S3.csv']
    if assets is not None:
        stage_daily_predictions(workspace, assets, prediction_names(('B0','M1','EA'), ('later',)))
        run_script(workspace, 'scripts/flv_valid_subset.py')
        names.append('outputs/metric_evidence_plan_20261102/flv_valid103_endpoints.csv')
    run_script(workspace, 'figures/Fig09/prepare_figure9_data.py')
    names.extend([
        'outputs/external_eq7_check_20261111/eq7_summary.csv',
        'outputs/external_eq7_check_20261111/eq7_basins.csv',
        'figures/Fig09/data/catchment_components.csv',
    ])
    checks = [compare_table(ROOT/name, workspace/name) for name in names]
    import numpy as np
    sensitivity_file = Path('outputs/selection_sensitivity/estimates.json')
    expected = json.loads((ROOT/sensitivity_file).read_text(encoding='utf-8'))
    actual = json.loads((workspace/sensitivity_file).read_text(encoding='utf-8'))
    for model in ('M1','EA'):
        for endpoint, summary in expected['seed_only_reconstruction'][model].items():
            for stat, val in summary.items():
                np.testing.assert_allclose(actual['seed_only_reconstruction'][model][endpoint][stat],val,rtol=0,atol=1e-12)
    for seed, summary in expected['permutation_reconstruction'].items():
        for stat, val in summary.items():
            np.testing.assert_allclose(actual['permutation_reconstruction'][seed][stat], val, rtol=0, atol=1e-12)
    comparison = {'status':'PASS','checks':checks,'seed_and_permutation_checks':'PASS'}
    (workspace/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    print(json.dumps(comparison, indent=2))


def figures(numbers, prepare=False, assets=None):
    workspace = stage(ROOT / ('work/figure-data' if prepare else 'work/figures'), all_inputs=prepare)
    if prepare and '7' in numbers:
        stage_daily_predictions(workspace, assets, prediction_names(('B0','EA'), ('earlier','later')))
    if prepare:
        required = set(numbers)
        if '8' in required:
            required.update(('4','5','S1'))
        if 'S3' in required:
            required.add('5')
        for number in ('1','2','3','4','5','6','S1','7','8','9','S4','S2','S3'):
            if number not in required:
                continue
            folder = figure_folder(number)
            extra = {'3': ['prepare_variance.py'], 'S4': ['prepare_radar.py'], '7': ['prepare_independent.py']}
            if number == '8':
                builders = ['prepare_influence.py', 'prepare_forest_references.py', 'prepare_figure8_data.py']
            elif number == '1':
                builders = []
            else:
                builders = [f'prepare_figure{number}_data.py'] + extra.get(number, [])
            for name in builders:
                run_script(workspace, folder/name)
    for number in numbers:
        folder = figure_folder(number)
        run_script(workspace, folder/f'plot_figure{number}.py')
    print(f'PASS {len(numbers)} figures: {workspace.relative_to(ROOT)}/figures', flush=True)


def extract(folder):
    catalog = verify_assets(folder)
    # Refuse to overwrite a changed local file; never accept a path outside the repository.
    for item in catalog['assets']:
        for record in item['files']:
            destination = (ROOT / record['path']).resolve()
            assert destination.is_relative_to(ROOT)
            if destination.exists() and digest(destination) != record['sha256']:
                raise FileExistsError(f'Different local file: {record["path"]}')
    for item in catalog['assets']:
        with zipfile.ZipFile(folder/item['name']) as archive:
            for record in item['files']:
                destination = ROOT / record['path']
                destination.parent.mkdir(parents=True,exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(archive.read(record['path']))
    print('PASS release assets extracted at repository root')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'results', 'figures', 'prepare-figures', 'extract'])
    parser.add_argument('--figures', nargs='+', type=figure_id, choices=FIGURES, default=FIGURES)
    parser.add_argument('--assets', type=Path, help='Directory containing the three release ZIP assets')
    args = parser.parse_args()
    if args.action == 'verify': verify(args.assets)
    elif args.action == 'results': results(args.assets)
    elif args.action in ['figures','prepare-figures']: figures(args.figures, args.action=='prepare-figures', args.assets)
    elif args.action == 'extract':
        if args.assets is None: parser.error('extract requires --assets PATH')
        extract(args.assets)


if __name__ == '__main__':
    main()
