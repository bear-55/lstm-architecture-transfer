"""Run the dated, prespecified finite-pool CPU application analysis."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.multiseed_selection import selection_curve, direct_curve


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(p, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    proto_path = ROOT/'protocol/multiseed_selection_20261031.json'
    proto = json.loads(proto_path.read_text(encoding='utf-8'))
    source = ROOT/proto['source']
    manifest = json.loads((source.parent/'output_manifest.json').read_text(encoding='utf-8'))
    for name, h in manifest['files'].items():
        if sha(source.parent/name) != h:
            raise RuntimeError('Source output seal changed: '+name)
    with np.load(source, allow_pickle=False) as a:
        p, q = a['early_delta_nse'], a['later_delta_nse']
        seeds, basins, regions = a['seeds'], a['basins'], a['regions']
    assert p.shape == q.shape == (10, 107)
    results = selection_curve(p, q, tuple(proto['selection_members']))
    old = json.loads((source.parent/'summary.json').read_text(encoding='utf-8'))['primary']
    for ref in ('B0', 'M1'):
        np.testing.assert_allclose(results[1]['vs_'+ref].mean(), old['other_vs_'+ref]['estimate'], rtol=0, atol=1e-14)
    out = ROOT/'outputs/multiseed_selection_20261031'
    out.mkdir(exist_ok=True, parents=True)
    summary, by_target, by_basin, by_region = [], [], [], []
    errors = []
    for m, r in results.items():
        a, b = direct_curve(p, q, m)
        for direct, key in ((a, 'vs_B0'), (b, 'vs_M1')):
            errors.append(float(np.max(np.abs(direct-r[key]))))
            np.testing.assert_allclose(direct, r[key], rtol=0, atol=1e-12)
        row = {'selection_seeds': m, 'subsets_per_target': r['subsets_per_target'][0],
               'M1_selection_fraction': float(r['selection_probability'].mean())}
        for ref in ('B0', 'M1'):
            x = r['vs_'+ref]
            row.update({f'mean_gain_vs_{ref}': float(x.mean()),
                        f'basin_median_gain_vs_{ref}': float(np.median(x.mean(axis=0))),
                        f'positive_target_seeds_vs_{ref}': int((x.mean(axis=1)>0).sum()),
                        f'positive_basins_vs_{ref}': int((x.mean(axis=0)>0).sum())})
        summary.append(row)
        for t, seed in enumerate(seeds):
            by_target.append(dict(m=m, target_seed=int(seed), **{key:float(r[key][t].mean()) for key in ('vs_B0','vs_M1','selection_probability')}))
        for j, basin in enumerate(basins):
            by_basin.append(dict(m=m, basin=str(basin), HUC2=str(regions[j]), **{key:float(r[key][:,j].mean()) for key in ('vs_B0','vs_M1','selection_probability')}))
        for region in np.unique(regions):
            mask = regions == region
            by_region.append(dict(m=m, HUC2=str(region), basins=int(mask.sum()), **{key:float(r[key][:,mask].mean()) for key in ('vs_B0','vs_M1','selection_probability')}))
    for name, rows in [('curve',summary), ('by_target_seed',by_target), ('by_basin',by_basin), ('by_region',by_region)]:
        write_csv(out/(name+'.csv'), rows)
    np.savez_compressed(out/'all_target_basin_arrays.npz', seeds=seeds, basins=basins, regions=regions,
                        **{f'm{m}_{k}':r[k] for m,r in results.items() for k in ('vs_B0','vs_M1','selection_probability')})
    record = dict(status='completed', primary_endpoint_unchanged=True, curve=summary,
        independent_direct_max_abs_error=max(errors), m1_reproduces_confirmed_other_gain=True,
        protocol_sha256=sha(proto_path), source_sha256=sha(source),
        code_sha256={n:sha(ROOT/n) for n in ('analysis/multiseed_selection.py','scripts/analyze_multiseed_selection.py')},
        scope=proto['uncertainty'])
    (out/'summary.json').write_text(json.dumps(record, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
    lines = ['# Multi-seed selection with existing predictions', '',
             'Ten seeds, 107 held-out catchments. Selection uses early mean NSE and excludes the evaluation seed. Each deployed prediction comes from one independently trained model.', '',
             '| Selection seeds | Mean gain over B0 | Mean gain over M1 | Median catchment gain over B0 | Evaluation seeds with positive gain over B0 |',
             '|---|---|---|---|---|']
    for r in summary:
        lines.append(f"| {r['selection_seeds']} | {r['mean_gain_vs_B0']:+.5f} | {r['mean_gain_vs_M1']:+.5f} | {r['basin_median_gain_vs_B0']:+.5f} | {r['positive_target_seeds_vs_B0']}/10 |")
    if all(r['mean_gain_vs_B0']<0 for r in summary):
        lines += ['', 'All four budgets yield lower mean NSE than fixed B0. The nine-seed estimate exceeds the one-seed estimate slightly, but the curve is non-monotonic.']
    lines += ['', 'All planned budgets are shown. Overlapping subsets give finite-library estimates. The one-seed result reproduces the original independent-seed gain.',
              '', 'Assess gains against both fixed architectures using the full curve.',
              '', 'Direct enumeration agrees with the vectorized results for every catchment and evaluation seed.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    files = {f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.name!='output_manifest.json'}
    (out/'output_manifest.json').write_text(json.dumps(dict(status='completed',files=files), indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
