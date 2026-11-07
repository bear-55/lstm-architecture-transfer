"""Reconcile fixed-policy and reselected-global intervals for the current outline."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/jh_predraft_audit_20261102'


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    inputs = [ROOT/'outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz',
              ROOT/'outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz',
              ROOT/'outputs/multimetric_transfer_20261102/transfer_endpoints.csv',
              Path(__file__)]
    a, train = read_npz(inputs[0]), read_npz(inputs[1])
    existing = pd.read_csv(inputs[2])
    rng = np.random.default_rng(20261102)
    sw = np.ones((2001, 10))
    sw[1:] = rng.multinomial(10, np.full(10, .1), size=2000)
    for i in np.flatnonzero(np.count_nonzero(sw, axis=1) < 2):
        while np.count_nonzero(sw[i]) < 2:
            sw[i] = rng.multinomial(10, np.full(10, .1))
    groups = np.unique(np.concatenate([a['regions'], train['regions']]))
    gw = np.ones((2001, len(groups)))
    gw[1:] = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=2000)
    bw = gw[:, [list(groups).index(r) for r in a['regions']]]
    bw = bw/bw.sum(axis=1, keepdims=True)
    swsum = sw.sum(axis=1)
    off_denom = swsum**2-(sw**2).sum(axis=1)
    rows, draws = [], {}
    for mi, model in enumerate(a['models'][1:], 1):
        j = list(a['metrics']).index('nse')
        p = a['earlier'][mi, :, :, j] - a['earlier'][0, :, :, j]
        q = a['later'][mi, :, :, j] - a['later'][0, :, :, j]
        choice = (p > 0).astype(float)
        same_b = sw@(choice*q)/swsum[:, None]
        other_b = ((sw@choice)*(sw@q)-(sw**2)@(choice*q))/off_denom[:, None]
        local_same = (same_b*bw).sum(axis=1)
        local_other = (other_b*bw).sum(axis=1)
        qbar = bw@q.T
        for variant in ['conditional_fixed_global', 'reselect_global_in_bootstrap']:
            if variant == 'conditional_fixed_global':
                glob = np.broadcast_to((p.mean(axis=1) > 0)[None, :], sw.shape).astype(float)
            else:
                glob = (bw@p.T > 0).astype(float)
            gs = (sw*glob*qbar).sum(axis=1)/swsum
            go = ((sw*glob).sum(axis=1)*(sw*qbar).sum(axis=1)-((sw**2)*glob*qbar).sum(axis=1))/off_denom
            result = {'same':local_same, 'other':local_other, 'D':local_same-local_other,
                      'global_same':gs, 'global_other':go, 'global_D':gs-go,
                      'incremental_same':local_same-gs, 'incremental_other':local_other-go,
                      'incremental_D':(local_same-local_other)-(gs-go)}
            np.testing.assert_allclose(result['D'],result['global_D']+result['incremental_D'],atol=1e-12)
            for endpoint, values in result.items():
                lo, hi = np.quantile(values[1:], [.025, .975])
                if variant == 'conditional_fixed_global':
                    old = existing[(existing.scenario=='unseen107') & (existing.comparison==model+'-B0') &
                                   (existing.selection_rule=='fixed_early_NSE') & (existing.metric=='NSE') &
                                   (existing.aggregation=='mean') & (existing.endpoint==endpoint)].iloc[0]
                    np.testing.assert_allclose([values[0],lo,hi],old[['estimate','low95','high95']].astype(float),atol=1e-12)
                rows.append(dict(comparison=model+'-B0', variant=variant, endpoint=endpoint,
                                 estimate=values[0], low95=lo, high95=hi, n_seeds=10, n_basins=107))
                draws[f'{model}__{variant}__{endpoint}'] = values
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'global_baseline_interval_reconciliation.csv',index=False)
    result[result.variant=='reselect_global_in_bootstrap'].to_csv(OUT/'main_NSE_endpoints.csv',index=False)
    np.savez_compressed(OUT/'baseline_bootstrap_draws.npz',**draws)
    rec=dict(training=False,new_predictions=False,new_primary_endpoint=False,
             reason='Match uncertainty to the early-global recommendation procedure; retain conditional-policy sensitivity.',
             old_endpoint_replication='All 18 conditional point estimates and 36 interval bounds reproduced within 1e-12.',
             sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'baseline_audit_provenance.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result[(result.variant=='reselect_global_in_bootstrap') & result.endpoint.isin(['D','incremental_same','incremental_other','incremental_D'])].to_string(index=False))


if __name__=='__main__':
    main()
