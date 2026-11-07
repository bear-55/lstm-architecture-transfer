"""Read-only multi-metric extension of the aligned selection/retraining experiment."""
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
OUT = ROOT / 'outputs/multimetric_transfer_20261102'
SOURCES = {
    'unseen107': ROOT / 'outputs/training_domain_followup_20261102/unseen_aligned',
    'trained370': ROOT / 'outputs/weakness_resolution_20261102/local_internal',
}
DEFINITIONS = [
    ('NSE', 'nse', 'identity', 'NSE'),
    ('NSE_star', 'nse', 'bounded', 'NSE*'),
    ('KGE', 'kge', 'identity', 'KGE'),
    ('abs_FHV', 'fhv', 'negative_absolute', 'percentage points'),
    ('MAE_low', 'mae_low', 'negative', 'mm/day'),
    ('MAE_high', 'mae_high', 'negative', 'mm/day'),
    ('nMAE_low', 'mae_low', 'negative_normalized', 'dimensionless'),
    ('nMAE_high', 'mae_high', 'negative_normalized', 'dimensionless'),
    ('abs_Q98_bias', 'peak_day_bias_pct', 'negative_absolute', 'percentage points'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def prediction_path(scenario, era):
    if scenario == 'trained370':
        return ROOT / f'runs/internal_temporal_extension_20261102/{era}/B0/seed_1009/predictions_test.npz'
    prefix = 'domain_matched_20261102' if era == 'earlier' else 'confirmation_evaluation_20261030'
    return ROOT / f'runs/{prefix}/{era}/B0/seed_1009/predictions_test.npz'


def utility(a, metric_names, definition, obs_mean):
    _, source, transform, _ = definition
    x = a[..., list(metric_names).index(source)].astype(float)
    if transform == 'bounded':
        return x / (2 - x)
    if transform == 'negative_absolute':
        return -np.abs(x)
    if transform == 'negative':
        return -x
    if transform == 'negative_normalized':
        return -x / obs_mean[None, None, :]
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [ROOT / 'protocol/multimetric_transfer_20261102.md', Path(__file__)]
    z = {}
    for key, path in SOURCES.items():
        z[key] = load(path / 'metric_arrays.npz')
        source_paths += [path / 'metric_arrays.npz', path / 'policy_hydrology_per_basin.csv', path / 'individual_metrics.csv']
    for k in ['models', 'seeds', 'metrics']:
        np.testing.assert_array_equal(z['unseen107'][k], z['trained370'][k])
    assert list(z['unseen107']['models']) == ['B0', 'M1', 'EA']
    rng = np.random.default_rng(20261102)
    sw = np.ones((2001, 10))
    sw[1:] = rng.multinomial(10, np.full(10, .1), size=2000)
    for i in np.flatnonzero(np.count_nonzero(sw, axis=1) < 2):
        while np.count_nonzero(sw[i]) < 2:
            sw[i] = rng.multinomial(10, np.full(10, .1))
    groups = np.unique(np.concatenate([a['regions'] for a in z.values()]))
    gw = np.ones((2001, len(groups)))
    gw[1:] = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=2000)
    weights = {key: gw[:, [list(groups).index(r) for r in a['regions']]] for key, a in z.items()}
    swsum = sw.sum(1, keepdims=True)
    off_denom = ((sw.sum(1))**2 - (sw**2).sum(1))[:, None]

    def outcomes(choose, q):
        same = sw @ (choose*q) / swsum
        other = ((sw @ choose) * (sw @ q) - (sw**2) @ (choose*q)) / off_denom
        # Independent explicit enumeration of the point estimate.
        explicit_same = np.mean(choose*q, axis=0)
        explicit_other = np.mean([choose[i]*q[j] for i in range(10) for j in range(10) if i != j], axis=0)
        np.testing.assert_allclose(same[0], explicit_same, atol=1e-12)
        np.testing.assert_allclose(other[0], explicit_other, atol=1e-12)
        return {'same': same, 'other': other, 'D': same-other}

    rows, basinrows, draws, obsrows = [], [], {}, []
    policy_rows, policy_summary, utilities = [], [], {}
    obs_means = {}
    for scenario, a in z.items():
        assert np.isfinite(a['earlier']).all() and np.isfinite(a['later']).all()
        obs_means[scenario] = {}
        individual = pd.read_csv(SOURCES[scenario] / 'individual_metrics.csv', dtype={'basin_id': str})
        for era in ['earlier', 'later']:
            path = ROOT / 'outputs/multimetric_transfer_20261102/observation_normalizers.csv'
            source_paths.append(path)
            norms = pd.read_csv(path, dtype={'basin_id': str}, float_precision='round_trip')
            norms = norms[(norms.scenario == scenario) & (norms.era == era)].set_index('basin_id').loc[a['basins']]
            assert len(norms) == len(a['basins']) and (norms.mean_q_mm_day > 0).all()
            for b, rec in norms.iterrows():
                old = individual[(individual.era == era) & (individual.model == 'B0') & (individual.seed == 1009) & (individual.basin_id == b)]
                assert len(old) == 1 and int(rec.n_days) == int(old.n_days.iloc[0])
                obsrows.append(dict(scenario=scenario, era=era, basin_id=b, n_days=int(rec.n_days), mean_q_mm_day=float(rec.mean_q_mm_day), min_date=rec.min_date, max_date=rec.max_date))
            obs_means[scenario][era] = norms.mean_q_mm_day.to_numpy(dtype=float)
        for definition in DEFINITIONS:
            name, _, _, unit = definition
            eu = utility(a['earlier'], a['metrics'], definition, obs_means[scenario]['earlier'])
            lu = utility(a['later'], a['metrics'], definition, obs_means[scenario]['later'])
            utilities[(scenario, name)] = (eu, lu)
            for rule in ['fixed_early_NSE', 'metric_specific']:
                selection = a['earlier'][..., 0] if rule == 'fixed_early_NSE' else eu
                for mi, model in enumerate(['M1', 'EA'], 1):
                    p = selection[mi] - selection[0]
                    choose = (p > 0).astype(float)
                    q = lu[mi] - lu[0]
                    loc = outcomes(choose, q)
                    glob = outcomes(np.broadcast_to((p.mean(1) > 0)[:, None], p.shape).astype(float), q)
                    all_out = dict(loc)
                    for endpoint in loc:
                        all_out['global_' + endpoint] = glob[endpoint]
                        all_out['incremental_' + endpoint] = loc[endpoint] - glob[endpoint]
                        np.testing.assert_allclose(all_out[endpoint], all_out['global_' + endpoint] + all_out['incremental_' + endpoint], atol=1e-12)
                    tags = dict(scenario=scenario, comparison=model+'-B0', selection_rule=rule, metric=name, unit=unit, n_basins=len(a['basins']), n_seeds=10)
                    for bi, b in enumerate(a['basins']):
                        basinrows.append(dict(**tags, basin_id=b, **{k: v[0, bi] for k, v in all_out.items()}))
                    for endpoint, values in all_out.items():
                        for aggregation in ['mean', 'basin_median']:
                            w = weights[scenario]
                            agg = (values*w).sum(1)/w.sum(1) if aggregation == 'mean' else weighted_median(values, w)
                            lo, hi = np.quantile(agg[1:], [.025, .975])
                            rows.append(dict(**tags, aggregation=aggregation, endpoint=endpoint, estimate=agg[0], low95=lo, high95=hi))
                            draws[f'{scenario}__{model}__{rule}__{name}__{aggregation}__{endpoint}'] = agg
            print('METRIC_COMPLETE', scenario, name, flush=True)
        policy = pd.read_csv(SOURCES[scenario] / 'policy_hydrology_per_basin.csv', dtype={'basin_id': str})
        meanmap = dict(zip(a['basins'], obs_means[scenario]['later']))
        policy['nse_star'] = policy.nse / (2-policy.nse)
        policy['abs_fhv'] = policy.fhv.abs()
        policy['abs_Q98_bias'] = policy.peak_day_bias_pct.abs()
        policy['abs_alpha_error'] = (policy.alpha-1).abs()
        policy['abs_beta_error'] = (policy.beta-1).abs()
        policy['nmae_low'] = policy.mae_low / policy.basin_id.map(meanmap)
        policy['nmae_high'] = policy.mae_high / policy.basin_id.map(meanmap)
        # Individual-seed references are averages of individual metrics, not
        # scores of an ensemble. Apply nonlinear transforms before seed means.
        indices = {b: i for i, b in enumerate(a['basins'])}
        metric_index = {m: i for i, m in enumerate(a['metrics'])}
        for model_index, model in enumerate(a['models']):
            mask = policy.policy == model+'_individual_seed_mean'
            for idx in policy.index[mask]:
                raw = a['later'][model_index, :, indices[policy.at[idx, 'basin_id']]]
                nse = raw[:, metric_index['nse']]
                policy.at[idx, 'nse_star'] = np.mean(nse/(2-nse))
                for derived, source, center in [('abs_fhv','fhv',0),('abs_Q98_bias','peak_day_bias_pct',0),('abs_alpha_error','alpha',1),('abs_beta_error','beta',1)]:
                    policy.at[idx, derived] = np.abs(raw[:, metric_index[source]]-center).mean()
                # Even this reference must not average unstable FLV values.
                policy.at[idx, 'flv'] = np.median(raw[:, metric_index['flv']])
        policy.insert(0, 'scenario', scenario)
        policy_rows.append(policy)
        for (comparison, strategy), g in policy.groupby(['comparison', 'policy'], sort=False):
            for col in [c for c in policy if c not in ['scenario', 'comparison', 'policy', 'basin_id']]:
                policy_summary.append(dict(scenario=scenario, comparison=comparison, policy=strategy, metric=col, mean=np.nan if col == 'flv' else g[col].mean(), median=g[col].median(), q25=g[col].quantile(.25), q75=g[col].quantile(.75), n=len(g)))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/'transfer_endpoints.csv', index=False)
    pd.DataFrame(basinrows).to_csv(OUT/'transfer_per_basin.csv', index=False)
    pd.DataFrame(obsrows).to_csv(OUT/'observation_normalizers.csv', index=False)
    pd.concat(policy_rows, ignore_index=True).to_csv(OUT/'policy_metrics_per_basin.csv', index=False)
    pd.DataFrame(policy_summary).to_csv(OUT/'policy_metric_summary.csv', index=False)
    np.savez_compressed(OUT/'bootstrap_draws.npz', **draws)

    # Exact reproduction of the existing P1 endpoint and interval calculations.
    p1path = ROOT/'outputs/selection_transfer_P1_20261102/P1_endpoints.csv'
    source_paths.append(p1path)
    p1 = pd.read_csv(p1path)
    checks = []
    for scenario in z:
        for model in ['M1', 'EA']:
            for metric in ['NSE', 'NSE_star']:
                for aggregation in ['mean', 'basin_median']:
                    for endpoint in ['same', 'other', 'D']:
                        old = p1[(p1.analysis == 'scale_influence') & (p1.scenario == scenario) & (p1.comparison == model+'-B0') & (p1.scale == metric) & (p1.variant == aggregation) & (p1.endpoint == endpoint)]
                        new = table[(table.scenario == scenario) & (table.comparison == model+'-B0') & (table.metric == metric) & (table.aggregation == aggregation) & (table.endpoint == endpoint) & (table.selection_rule == 'fixed_early_NSE')]
                        assert len(old) == len(new) == 1
                        np.testing.assert_allclose(old[['estimate','low95','high95']], new[['estimate','low95','high95']], atol=1e-12)
    checks.append('All 48 existing P1 NSE/NSE* mean/median endpoint estimates and intervals reproduced to 1e-12')
    checks.append('Every local/global point estimate independently enumerated over 10 same-seed and 90 distinct-seed pairs')
    checks.append('Mean D decomposes exactly into global plus local-incremental D; no median additivity assumed')
    checks.append('All four observation normalizer sets use the original per-basin effective-day counts; no missing/zero normalizers')
    checks.append('Signed bias converted to negative absolute bias before target-specific selection and transfer calculation')
    config = json.loads((ROOT/'configs/retraining_confirmation.json').read_text(encoding='utf-8'))
    for p, expected in config['frozen_training_sources'].items():
        assert sha(ROOT/p) == expected, p
    checks.append('All 12 frozen training source files unchanged')
    source_paths.append(ROOT/'configs/retraining_confirmation.json')
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    (OUT/'input_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed', checks=checks, new_training=False, n_boot=2000, interval='pointwise percentile seed/HUC2 bootstrap', n_metric_definitions=len(DEFINITIONS)), indent=2), encoding='utf-8')
    pd.DataFrame(DEFINITIONS, columns=['metric', 'source_metric', 'utility_transform', 'unit']).to_csv(OUT/'metric_definitions.csv', index=False)
    core = table[(table.scenario == 'unseen107') & (table.selection_rule == 'fixed_early_NSE') & (table.aggregation == 'mean') & (table.endpoint == 'D')]
    print(core[['comparison','metric','estimate','low95','high95']].to_string(index=False), flush=True)
    print('MULTIMETRIC_COMPLETE',len(table), 'rows', flush=True)


if __name__ == '__main__':
    main()
