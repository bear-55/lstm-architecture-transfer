"""Source budget 1--5, evaluated on disjoint five-instance target pools.

All evaluations average individual NSE scores, not daily discharge.  Averaging
over all disjoint five-member target subsets equals the complementary-pool mean.
"""
from pathlib import Path
from itertools import combinations
from math import comb
from hashlib import sha256
from datetime import datetime, timezone
import os, json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/multiseed_budget_curve_20261104'
ARRAYS=ROOT/'outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz'
NBOOT=20000
RNG=20261104
ENDPOINTS=['G_same','G_new','G_global_same','G_global_new','D','increment_new','increment_same','increment_D']

def digest(p): return sha256(Path(p).read_bytes()).hexdigest()
def write_json(name,value): (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(name,rows): pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.15g')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(ARRAYS,allow_pickle=False) as z:
        early=z['earlier'][:,:,:,0];late=z['later'][:,:,:,0]
        basins=z['basins'];regions=z['regions'];seeds=z['seeds']
        assert z['models'].tolist()==['B0','M1','EA']
    assert early.shape==late.shape==(3,10,107)
    groups,gidx=np.unique(regions,return_inverse=True)
    assert len(groups)==16
    gm=np.eye(len(groups))[gidx]
    counts=np.random.default_rng(RNG).multinomial(16,np.full(16,1/16),size=NBOOT)
    weights=counts/(counts@gm.sum(0))[:,None]
    spec=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='Exploratory extension',
              selection_period='WY2000-2003',evaluation_period='WY2004-2008',scenario='107 held-out gauged catchments',
              source_budgets=[1,2,3,4,5],target_budget=5,
              choice='Mean source individual early NSE, catchment-wise versus same-information global architecture',
              evaluation='Arithmetic mean of five target individual scores, not NSE of mean discharge',
              enumeration='Every C(10,m) source subset and every C(10-m,5) disjoint target subset; both orientations at m=5',
              compression='Mean over all five-member target subsets is the mean of all non-source instances; verified explicitly',
              bootstrap='HUC2 cluster percentile intervals conditional on the fitted library; all source/target subsets kept together; global rule reselected',
              bootstrap_draws=NBOOT,bootstrap_seed=RNG,
              upper_bound='One-sided 95% percentile upper bound = bootstrap 0.95 quantile, distinct from two-sided 97.5% endpoint',
              score_sensitivity='Evaluate NSE* with early raw-NSE choice fixed; transform individual scores before averaging',
              metric_arrays_sha256=digest(ARRAYS),
              script_sha256=digest(__file__))
    write_json('analysis_spec.json',spec)
    rows=[];source_rows=[];pool_rows=[];draws={};points={};witness_count=0;partition_checks=[]
    for m in range(1,6):
        sources=np.array(list(combinations(range(10),m)))
        for sid,s in enumerate(sources):
            remain=[i for i in range(10) if i not in s]
            pool_rows.append(dict(source_budget=m,source_id=sid,source_seeds=';'.join(str(int(seeds[i])) for i in s),
                                  available_target_seeds=';'.join(str(int(seeds[i])) for i in remain),target_budget=5,
                                  target_subsets_per_source=comb(10-m,5)))
        for c,model in [(1,'M1'),(2,'EA')]:
            ed=(early[c][sources]-early[0][sources]).mean(1)
            choice=ed>0
            gc=ed.mean(1)>0
            for metric in ['NSE','NSE_star']:
                q=late if metric=='NSE' else late/(2-late)
                source0,source1=q[0][sources].mean(1),q[c][sources].mean(1)
                target0=(q[0].sum(0)-m*source0)/(10-m)
                target1=(q[c].sum(0)-m*source1)/(10-m)
                ds=source1-source0;dn=target1-target0
                gs=choice*ds;gn=choice*dn
                ggs=gc[:,None]*ds;ggn=gc[:,None]*dn
                source_values=np.stack([gs.mean(1),gn.mean(1),ggs.mean(1),ggn.mean(1),
                                        (gs-gn).mean(1),(gn-ggn).mean(1),(gs-ggs).mean(1),
                                        ((gs-ggs)-(gn-ggn)).mean(1)],axis=1)
                estimates=source_values.mean(0)
                key=f'{model}__m{m}__{metric}'
                local_vectors=np.stack([gs.mean(0),gn.mean(0)])@gm
                aggs={k:v@gm for k,v in [('ed',ed),('ds',ds),('dn',dn)]}
                output=np.empty((NBOOT,len(ENDPOINTS)))
                for start in range(0,NBOOT,500):
                    w=weights[start:start+500]
                    bgs,bgn=(w@local_vectors.T).T
                    bgc=w@aggs['ed'].T>0
                    bggs=np.mean(bgc*(w@aggs['ds'].T),axis=1)
                    bggn=np.mean(bgc*(w@aggs['dn'].T),axis=1)
                    output[start:start+len(w)]=np.column_stack([bgs,bgn,bggs,bggn,bgs-bgn,bgn-bggn,bgs-bggs,
                                                              (bgs-bggs)-(bgn-bggn)])
                np.testing.assert_allclose(estimates[4],estimates[0]-estimates[1],atol=1e-13,rtol=0)
                np.testing.assert_allclose(estimates[5],estimates[1]-estimates[3],atol=1e-13,rtol=0)
                # Fully enumerate actual five-member targets, independently of the compression.
                explicit=[]
                for sid,s in enumerate(sources):
                    targets=np.array(list(combinations([i for i in range(10) if i not in s],5)))
                    assert np.isin(targets,s).sum()==0
                    difference=(q[c][targets]-q[0][targets]).mean(1)
                    np.testing.assert_allclose(difference.mean(0),dn[sid],atol=1e-12,rtol=0)
                    explicit.append(np.mean((choice[sid].astype(float)-gc[sid])*difference))
                np.testing.assert_allclose(np.mean(explicit),estimates[5],atol=1e-12,rtol=0)
                # Explicit repeated-basin witness of global re-selection in bootstrap draw 0.
                ix=np.repeat(np.arange(107),counts[0,gidx])
                gc0=ed[:,ix].mean(1)>0
                witness=np.mean((choice[:,ix].astype(float)-gc0[:,None])*dn[:,ix])
                np.testing.assert_allclose(witness,output[0,5],atol=1e-12,rtol=0)
                witness_count+=1
                if m==5:
                    idx={tuple(s):i for i,s in enumerate(sources)}
                    pairs=[(i,idx[tuple(t for t in range(10) if t not in s)]) for i,s in enumerate(sources) if i<idx[tuple(t for t in range(10) if t not in s)]]
                    assert len(pairs)==126
                    oriented=source_values[np.asarray(pairs)].mean(1)
                    np.testing.assert_allclose(oriented.mean(0),estimates,atol=1e-13,rtol=0)
                    partition_checks.append(key)
                for j,ep in enumerate(ENDPOINTS):
                    lo,hi=np.quantile(output[:,j],[.025,.975])
                    lower,upper=np.quantile(output[:,j],[.05,.95])
                    rows.append(dict(comparison=model+'-B0',metric=metric,source_budget=m,target_budget=5,
                                     endpoint=ep,estimate=estimates[j],low95=lo,high95=hi,one_sided_lower95=lower,
                                     one_sided_upper95=upper,source_subset_sd=float(source_values[:,j].std(ddof=0)),
                                     n_source_subsets=len(sources),n_oriented_source_target_pairs=comb(10,m)*comb(10-m,5),
                                     n_boot=NBOOT))
                for sid in range(len(sources)):
                    source_rows.append(dict(comparison=model+'-B0',metric=metric,source_budget=m,source_id=sid,
                                            target_subsets=comb(10-m,5),global_candidate=bool(gc[sid]),
                                            **{ep:source_values[sid,j] for j,ep in enumerate(ENDPOINTS)}))
                draws[key]=output;points[key]=estimates
                print(model,metric,'m=',m,'D=',estimates[4],'increment=',estimates[5],flush=True)
    write_csv('budget_curve_endpoints.csv',rows)
    write_csv('source_subset_results.csv',source_rows)
    write_csv('pool_definitions.csv',pool_rows)
    changes=[]
    for model in ['M1','EA']:
        for metric in ['NSE','NSE_star']:
            for first,last in [(1,2),(2,3),(3,4),(4,5),(1,5)]:
                a,b=f'{model}__m{first}__{metric}',f'{model}__m{last}__{metric}'
                for ep in ['G_new','D','increment_new']:
                    j=ENDPOINTS.index(ep)
                    lo,hi=np.quantile(draws[b][:,j]-draws[a][:,j],[.025,.975])
                    changes.append(dict(comparison=model+'-B0',metric=metric,source_budget_from=first,source_budget_to=last,
                                        endpoint=ep,estimate=points[b][j]-points[a][j],low95=lo,high95=hi))
    write_csv('paired_budget_changes.csv',changes)
    np.savez_compressed(OUT/'bootstrap_draws.npz',endpoint_names=np.array(ENDPOINTS),**draws)
    assert digest(ARRAYS)==spec['metric_arrays_sha256']
    check=dict(status='PASS',metric_arrays_unchanged=True,
               explicit_target_subset_enumeration_passed=True,source_target_identity_overlap=False,
               bootstrap_global_rule_reselected=True,explicit_bootstrap_witnesses=witness_count,

               unordered_126_partitions_match_oriented_252_partitions=partition_checks,
               script_sha256=digest(__file__))
    write_json('verification.json',check)
    print('VERIFIED',OUT,flush=True)

if __name__=='__main__':main()
