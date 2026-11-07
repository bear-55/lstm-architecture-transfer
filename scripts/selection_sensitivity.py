"""Read-only checks of manuscript claims; writes only separate audit outputs."""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/selection_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
sources = {}

def source(rel):
    p = ROOT / rel
    sources[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def readnpz(rel):
    with np.load(source(rel), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def summarize(v):
    return dict(estimate=float(v[0]), low95=float(np.quantile(v[1:], .025)), high95=float(np.quantile(v[1:], .975)), upper_one_sided_95=float(np.quantile(v[1:], .95)))

def gains(p, q, sw):
    choice = (p > 0).astype(float)
    total = sw.sum(axis=1)
    off_denom = total**2 - (sw**2).sum(axis=1)
    same_b = sw @ (choice*q) / total[:,None]
    other_b = ((sw@choice)*(sw@q) - (sw**2)@(choice*q)) / off_denom[:,None]
    g = (p.mean(axis=1) > 0).astype(float)
    qb = q.mean(axis=1)
    gs = (sw*(g*qb)).sum(axis=1)/total
    go = ((sw@g)*(sw@qb)-(sw**2)@(g*qb))/off_denom
    same, other = same_b.mean(axis=1), other_b.mean(axis=1)
    return dict(same=same, other=other, D=same-other, global_same=gs, global_other=go, incremental_same=same-gs, incremental_other=other-go)

result = {'purpose':'Independent deterministic numerical checks, no experiment files changed', 'same_seed_bootstrap_rng':20261102, 'source_files':sources}
arrays = {}
for scenario, rel in [('heldout','outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz'), ('training','outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz')]:
    a = readnpz(rel)
    arrays[scenario] = a
    j = list(a['metrics']).index('nse')
    n = a['earlier'].shape[1]
    result[scenario] = {}
    for mi, model in enumerate(a['models'][1:], 1):
        e0, e1 = a['earlier'][0,:,:,j], a['earlier'][mi,:,:,j]
        l0, l1 = a['later'][0,:,:,j], a['later'][mi,:,:,j]
        p, q = e1/(2-e1)-e0/(2-e0), l1/(2-l1)-l0/(2-l0)
        corr_same = [spearmanr(p[s],q[s]).statistic for s in range(n)]
        corr_other = [spearmanr(p[s],(q.sum(axis=0)-q[s])/(n-1)).statistic for s in range(n)]
        choice = (e1-e0 > 0).astype(float)
        sb = (choice*q).mean(axis=0)
        ob = (choice.sum(axis=0)*q.sum(axis=0)-(choice*q).sum(axis=0))/(n*(n-1))
        db = sb-ob
        model_result = {'spearman_same_mean':float(np.mean(corr_same)), 'spearman_other_mean':float(np.mean(corr_other)), 'positive_NSE_star_D_count':int((db>0).sum()), 'n_basins':len(db), 'positive_NSE_star_D_percent':float((db>0).mean()*100), 'positive_regions_count':int(sum(db[a['regions']==r].mean()>0 for r in np.unique(a['regions']))), 'n_regions':len(np.unique(a['regions']))}
        result[scenario][str(model)] = model_result

a = arrays['heldout']
j = list(a['metrics']).index('nse')
rng = np.random.default_rng(20261102)
sw = np.ones((2001,10))
sw[1:] = rng.multinomial(10, np.full(10,.1), size=2000)
for row in np.flatnonzero(np.count_nonzero(sw, axis=1)<2):
    while np.count_nonzero(sw[row])<2:
        sw[row] = rng.multinomial(10,np.full(10,.1))
result['seed_only_reconstruction'] = {}
for mi, model in enumerate(a['models'][1:],1):
    p=a['earlier'][mi,:,:,j]-a['earlier'][0,:,:,j]
    q=a['later'][mi,:,:,j]-a['later'][0,:,:,j]
    g=gains(p,q,sw)
    result['seed_only_reconstruction'][str(model)]={k:summarize(v) for k,v in g.items()}

draws=readnpz('outputs/jh_predraft_audit_20261102/baseline_bootstrap_draws.npz')
result['archived_bootstrap_checks']={}
for model in a['models'][1:]:
    prefix=f'{model}__reselect_global_in_bootstrap__'
    ratio=draws[prefix+'other']/draws[prefix+'same']
    result['archived_bootstrap_checks'][str(model)]={'transfer_ratio':summarize(ratio),'incremental_new':summarize(draws[prefix+'incremental_other']), 'nonpositive_denominators':int((draws[prefix+'same']<=0).sum())}

mi=list(a['models']).index('EA')
eC, eB=a['earlier'][mi,:,:,j], a['earlier'][0,:,:,j]
lC, lB=a['later'][mi,:,:,j], a['later'][0,:,:,j]
plain=np.ones((1,10))
observed=float(gains(eC-eB,lC-lB,plain)['D'][0])
result['permutation_reconstruction']={}
for seed in [20261102,20261103]:
    rng=np.random.default_rng(seed)
    ds=[]
    for _ in range(2000):
        perm=rng.permutation(10)
        ds.append(float(gains(eC-eB[perm],lC-lB[perm],plain)['D'][0]))
    ds=np.array(ds)
    result['permutation_reconstruction'][str(seed)]={'observed':observed,'low95':float(np.quantile(ds,.025)), 'high95':float(np.quantile(ds,.975)), 'minimum':float(ds.min()), 'observed_percentile':float((ds<=observed).mean()*100)}

t=pd.read_csv(source('outputs/multimetric_transfer_20261102/transfer_endpoints.csv'))
result['lowflow_targeted_medians']=t[(t.scenario=='unseen107')&(t.selection_rule=='metric_specific')&(t.metric=='nMAE_low')&(t.aggregation=='basin_median')&(t.endpoint=='D')].to_dict(orient='records')
result['source_files']=sources
(OUT/'estimates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
schemes = {
    'Seed + HUC2, reselect global': 'reselect_global_in_bootstrap',
    'Seed + HUC2, fixed global': 'conditional_fixed_global',
    'Seed only, reselect global': 'seed_only',
}
endpoints = {'same':'Retained gain', 'other':'Transferred gain', 'D':'D',
             'global_other':'Global transferred', 'incremental_same':'Local retained increment',
             'incremental_other':'Local transferred increment'}
rows = []
for model in ('M1', 'EA'):
    for scheme, mode in schemes.items():
        for key, label in endpoints.items():
            stats = (result['seed_only_reconstruction'][model][key] if mode == 'seed_only'
                     else summarize(draws[f'{model}__{mode}__{key}']))
            estimate = f"{stats['estimate']:.4f}".replace('-', '−')
            limits = f"[{stats['low95']:.4f}, {stats['high95']:.4f}]".replace('-', '−')
            rows.append([model+'/B0', scheme, label, estimate, limits])
tables = ROOT/'tables'
tables.mkdir(exist_ok=True)
pd.DataFrame(rows, columns=['Contrast','Resampling','Endpoint','Estimate','95% interval']).to_csv(
    tables/'Table_S3.csv', index=False, lineterminator='\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_files'},ensure_ascii=False,indent=2))
