"""Reproduce Table S2b sensitivity checks on raw and bounded NSE.

Run from the repository root:
    python scripts/reproduce_s2b_sensitivity.py --out outputs/s2b_sensitivity
The subsets are frozen using early raw NSE for both evaluation scales.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import numpy as np


def run(root, out):
    sys.path.insert(0, str(root / 'scripts'))
    from weakness_stats import reml

    out.mkdir(parents=True, exist_ok=True)
    inputs = {
        'Held-out': 'outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz',
        'Training': 'outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz',
    }
    rows, details, subsets, hashes = [], [], [], {}
    for group, rel in inputs.items():
        path = root / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as a:
            models = a['models'].astype(str).tolist()
            early = a['earlier'][..., 0]
            late = a['later'][..., 0]
            basins = a['basins'].astype(str)
        base = models.index('B0')
        for candidate in ['M1', 'EA']:
            m = models.index(candidate)
            early_delta = early[m] - early[base]
            masks = {
                'nonnegative early NSE': np.flatnonzero((early[[base, m]] >= 0).all(axis=(0, 1))),
                'omit 5 largest early differences': np.argsort(np.abs(early_delta.mean(axis=0)))[:-5],
            }
            for name, idx in masks.items():
                subsets.append({'Group': group, 'Contrast': candidate + '/B0', 'subset': name,
                                'basin_ids': basins[idx].tolist()})
            for scale in ['NSE', 'NSE*']:
                transformed = late if scale == 'NSE' else late / (2.0 - late)
                delta = transformed[m] - transformed[base]
                for name, idx in masks.items():
                    result = reml(delta[:, idx])
                    row = {'Group': group, 'Contrast': candidate + '/B0', 'Scale': scale + ', ' + name,
                           'N': len(idx), 'Catchment': f"{100 * result['basin_fraction']:.2f}",
                           'Seed': f"{100 * result['seed_fraction']:.2f}",
                           'Interaction / residual': f"{100 * result['residual_fraction']:.2f}"}
                    rows.append(row)
                    details.append({**row, **result})
    with (out / 'sensitivity_rows.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (out / 'reml_full_precision.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8', newline='\n')
    (out / 'subsets.json').write_text(json.dumps(subsets, indent=2) + '\n', encoding='utf-8', newline='\n')
    (out / 'provenance.json').write_text(json.dumps({
        'inputs_sha256': hashes,
        'estimator': 'scripts/weakness_stats.py:reml (balanced crossed-factor REML)',
        'evaluation_period': 'WY2004-2008',
        'subset_definition_period': 'WY2000-2003',
        'bounded_transform': 'NSE* = NSE / (2 - NSE), applied before candidate-minus-B0 differences',
        'subset_rule': 'Identical early-raw-NSE subsets on both scales; ten paired seeds per contrast.',
    }, indent=2) + '\n', encoding='utf-8', newline='\n')
    table_path = root / 'tables/Table_S2b.csv'
    if table_path.exists():
        with table_path.open(encoding='utf-8-sig', newline='') as f:
            table = list(csv.DictReader(f))
        old_sensitive = [r for r in table if r['Scale'] not in ['NSE', 'NSE*', 'NSE ranks', 'KGE']]
        for old in old_sensitive:
            label = old['Scale']
            if not label.startswith('NSE'):
                label = 'NSE, ' + label[0].lower() + label[1:]
            new = next(r for r in rows if (r['Group'], r['Contrast'], r['Scale']) ==
                       (old['Group'], old['Contrast'], label))
            for col in ['N', 'Catchment', 'Seed', 'Interaction / residual']:
                assert str(old[col]) == str(new[col]), (col, old, new)
        combined = []
        for group in inputs:
            for contrast in ['M1/B0', 'EA/B0']:
                combined.extend(r for r in table if (r['Group'], r['Contrast']) == (group, contrast)
                                and r['Scale'] in ['NSE', 'NSE*', 'NSE ranks', 'KGE'])
                combined.extend(r for r in rows if (r['Group'], r['Contrast']) == (group, contrast))
        assert len(combined) == 32
        with (out / 'Table_S2b.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(combined)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path('outputs/s2b_sensitivity'))
    args = parser.parse_args()
    run(args.root.resolve(), args.out.resolve())
