"""Exploratory FLV transfer on observation-defined valid basins; CPU only."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from matching_scale_followup import weighted_median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/metric_evidence_plan_20261102'


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    validity_path = OUT / 'flv_observation_validity_per_basin.csv'
    heldout_path = ROOT / 'outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz'
    training_path = ROOT / 'outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz'
    endpoints_path = ROOT / 'outputs/multimetric_transfer_20261102/transfer_endpoints.csv'
    paths = [Path(__file__), validity_path, heldout_path, training_path, endpoints_path]
    a, training = load(heldout_path), load(training_path)
    validity = pd.read_csv(validity_path, dtype={'basin_id': str})
    validity = validity[validity.scenario == 'unseen107'].pivot(index='basin_id', columns='era', values='raw_log_denominator')
    valid = (validity.reindex(a['basins']) > 1e-6).all(axis=1).to_numpy()
    assert valid.sum() == 103
    rng = np.random.default_rng(20261102)
    sw = np.ones((2001, 10))
    sw[1:] = rng.multinomial(10, np.full(10, .1), size=2000)
    for i in np.flatnonzero(np.count_nonzero(sw, axis=1) < 2):
        while np.count_nonzero(sw[i]) < 2:
            sw[i] = rng.multinomial(10, np.full(10, .1))
    groups = np.unique(np.concatenate([a['regions'], training['regions']]))
    gw = np.ones((2001, len(groups)))
    gw[1:] = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=2000)
    full_w = gw[:, [list(groups).index(r) for r in a['regions']]]
    w = full_w[:, valid]
    assert (w.sum(axis=1) > 0).all()

    def outcomes(choice, q):
        same = sw @ (choice*q) / sw.sum(axis=1, keepdims=True)
        other = ((sw @ choice)*(sw @ q)-(sw**2)@(choice*q)) / ((sw.sum(axis=1))**2-(sw**2).sum(axis=1))[:, None]
        explicit = np.mean([choice[s]*q[t] for s in range(10) for t in range(10) if s != t], axis=0)
        np.testing.assert_allclose(other[0], explicit, atol=1e-12)
        return {'same': same, 'other': other, 'D': same-other}

    nse_i = list(a['metrics']).index('nse')
    flv_i = list(a['metrics']).index('flv')
    existing = pd.read_csv(endpoints_path)
    for mi, model in enumerate(a['models'][1:], 1):
        p = (a['earlier'][mi, :, :, nse_i] > a['earlier'][0, :, :, nse_i]).astype(float)
        q = a['later'][mi, :, :, nse_i] - a['later'][0, :, :, nse_i]
        d = outcomes(p, q)['D']
        agg = (d*full_w).sum(axis=1)/full_w.sum(axis=1)
        old = existing[(existing.scenario == 'unseen107') & (existing.comparison == model+'-B0') &
                       (existing.selection_rule == 'fixed_early_NSE') & (existing.metric == 'NSE') &
                       (existing.aggregation == 'mean') & (existing.endpoint == 'D')].iloc[0]
        np.testing.assert_allclose([agg[0], *np.quantile(agg[1:], [.025, .975])], old[['estimate','low95','high95']].astype(float), atol=1e-12)

    epsilons = [1e-7, 1e-6, 1e-5]
    values = np.empty((3, 3, 10, int(valid.sum())))
    basins = a['basins'][valid]
    for mi, model in enumerate(a['models']):
        for si, seed in enumerate(a['seeds']):
            directory = 'ea_evaluation_20261101' if model == 'EA' else 'confirmation_evaluation_20261030'
            pred = ROOT / f'runs/{directory}/later/{model}/seed_{seed}/predictions_test.npz'
            paths.append(pred)
            z = load(pred)
            for bi, basin in enumerate(basins):
                mask = z['basin_ids'] == basin
                obs = z['q_observed_mm_day'][mask].astype(float)
                sim = z['q_predicted_mm_day'][mask].astype(float)
                ok = np.isfinite(obs) & np.isfinite(sim) & (obs >= 0)
                obs, sim = obs[ok], sim[ok]
                nl = max(1, round(.3*len(obs)))
                o, s = np.sort(obs)[::-1][-nl:], np.sort(sim)[::-1][-nl:]
                for ei, eps in enumerate(epsilons):
                    ol, sl = np.log(np.where(o > 0, o, eps)), np.log(np.where(s > 0, s, eps))
                    denominator = np.sum(ol-ol.min())
                    assert denominator > 1e-6
                    values[ei, mi, si, bi] = -100*(np.sum(sl-sl.min())-denominator)/(denominator+1e-6)
        print('FLV_RECOMPUTED', model, flush=True)
    np.testing.assert_allclose(values[1], a['later'][:, :, :, flv_i][:, :, valid], rtol=1e-10, atol=1e-8)
    rows, basin_rows = [], []
    for ei, eps in enumerate(epsilons):
        for mi, model in enumerate(a['models'][1:], 1):
            choice = (a['earlier'][mi, :, :, nse_i] > a['earlier'][0, :, :, nse_i]).astype(float)[:, valid]
            q = -np.abs(values[ei, mi]) + np.abs(values[ei, 0])
            results = outcomes(choice, q)
            for bi, basin in enumerate(basins):
                basin_rows.append(dict(comparison=model+'-B0', replacement_eps=eps, basin_id=str(basin), **{k:v[0,bi] for k,v in results.items()}))
            for endpoint, v in results.items():
                for aggregation in ['mean', 'basin_median']:
                    agg = (v*w).sum(axis=1)/w.sum(axis=1) if aggregation == 'mean' else weighted_median(v, w)
                    lo, hi = np.quantile(agg[1:], [.025, .975])
                    rows.append(dict(comparison=model+'-B0', replacement_eps=eps, n_basins=103,
                                     selection_rule='fixed_early_NSE', metric='abs_FLV', unit='percentage points',
                                     aggregation=aggregation, endpoint=endpoint, estimate=agg[0], low95=lo, high95=hi))
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'flv_valid103_endpoints.csv', index=False)
    pd.DataFrame(basin_rows).to_csv(OUT/'flv_valid103_per_basin.csv', index=False)
    record = dict(training=False, new_predictions=False, exploratory=True,
                  n_basins=103, excluded_basins=a['basins'][~valid].tolist(),
                  validation='Baseline FLV reproduces all 3090 saved values on eligible basins; full-107 NSE point and intervals reproduce existing results.',
                  inputs={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'flv_supplement_provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(result[result.endpoint == 'D'].to_string(index=False))


if __name__ == '__main__':
    main()
