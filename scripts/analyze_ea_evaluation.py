"""Audit EA predictions and execute the fixed EA/B0 transfer comparison."""
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import sys
import tarfile
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis'),str(ROOT/'scripts')]
import ea_evaluate as ee
import ea_extension as ea
import confirmation_evaluate as ce
import retraining_confirmation as cohort
import selection_transfer as transfer
from multiseed_selection import selection_curve,direct_curve
from analyze_confirmation_evaluation import distribution,reproducibility
from final_evaluate import read_json,require,save_json
from project import digest

OUT=ROOT/'outputs/ea_selection_analysis_20261101'
ARCHIVE=ROOT/'runs/server backups/ea_evaluation_results_20261101.tar.gz'
ATTRIBUTES=ROOT/'outputs/ems_cpu_reanalysis_20261028/inputs/basin_attributes.csv'
FLOWS=('all','low','mid','high')
NAMES=tuple(k.replace('M1','EA') for k in transfer.NAMES)


def rename(values): return {k.replace('M1','EA'):v for k,v in values.items()}
def sha(raw): return hashlib.sha256(raw).hexdigest()
def save(rows,name):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.15g')


def audit():
    cfg,meta,spec,manifest=ee.prepare()
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        entries=archive.getmembers()
        require(len({m.name for m in entries})==len(entries),'Duplicate archive member')
        for m in entries:
            p=PurePosixPath(m.name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in m.name
                and (m.isfile() or m.isdir()),'Unsafe archive member')
        raw={m.name:archive.extractfile(m).read() for m in entries if m.isfile()}
    prefix='runs/ea_evaluation_20261101/'
    read=lambda n:json.loads(raw[prefix+n])
    done,ctx=read('completed.json'),read('evaluation_context.json')
    require(done['status']=='completed' and done['evaluations']==20 and done['basins']==107
        and done['B0_predictions_reused'],'Incomplete EA evaluation')
    identity=sha(json.dumps(ctx['context'],sort_keys=True).encode())
    require(ctx['identity']==done['identity']==identity,'Inference identity differs')
    require(raw[ee.BUNDLE.name]==ee.BUNDLE.read_bytes(),'Inference bundle changed')
    expected_context=dict(bundle_sha256=digest(ee.BUNDLE),environment=read_json(ea.run_path(1009)/'run.json')['environment'],
        cohort_sha256=digest(ea.OUT/'cohort_completed.json'),baseline_completion_sha256=digest(ce.OUT/'completed.json'))
    require(ctx['context']==expected_context,'Inference inputs/environment differ')
    expected={'evaluation_context.json','validation_recheck.json','run_summary.csv'}
    expected|={f'validation_checks/seed_{s}.json' for s in spec['formal_seeds']}
    expected|={f'{era}/EA/seed_{s}/{n}' for era in ee.PERIODS for s in spec['formal_seeds']
        for n in ('result.json','metrics_per_basin.csv','predictions_test.npz')}
    require(set(done['artifacts'])==expected,'Unexpected EA evaluation roster')
    for n,h in done['artifacts'].items(): require(sha(raw[prefix+n])==h,'Changed inference artifact: '+n)
    vr=read('validation_recheck.json')
    require(vr['status']=='passed' and vr['identity']==identity,'Validation reproduction incomplete')
    checks={r['seed']:r for r in vr['runs']}
    require(len(vr['runs'])==10 and set(checks)==set(spec['formal_seeds']),'Validation roster mismatch')
    for s,r in checks.items():
        require(r==read(f'validation_checks/seed_{s}.json') and r['identity']==identity and r['status']=='passed'
            and r['model']=='EA' and r['checkpoint_sha256']==digest(ea.run_path(s)/'best.pt')
            and r['prediction_count']==73050,'Validation/checkpoint mismatch')
        trained=read_json(ea.run_path(s)/'result.json')['validation']
        require(r['cloud_validation']==trained,'Validation reference changed')
        for k in ('median_nse','mean_nse'): require(abs(r['local_validation'][k]-trained[k])<=1e-5,'Validation score differs')
    seeds=spec['formal_seeds'];basins=meta['splits']['test']
    matrices={};metric_rows=[];runs=[];counts=[];max_error=0.
    for era in ee.PERIODS:
        obs=ee.read_npz(ce.DATA/f'{era}_observations.npz');o=obs['q_observed_mm_day'].astype(float)
        groups=[np.flatnonzero(obs['basin_ids']==b) for b in basins];masks=[]
        for b,ix in zip(basins,groups):
            q20,q95=np.quantile(o[ix],[.2,.95]);hi=o[ix]>=q95;lo=~hi&(o[ix]<=q20)
            fm=dict(all=np.ones(len(ix),bool),low=lo,mid=~hi&~lo,high=hi)
            require(all(v.any() for v in fm.values()),'Empty flow stratum')
            masks.append(fm)
            counts.append(dict(era=era,basin_id=b,q20=q20,q95=q95,**{f'n_{f}':int(v.sum()) for f,v in fm.items()}))
        for model in ('B0','EA'):
            nse=np.empty((10,107));mae={f:np.empty((10,107)) for f in FLOWS}
            for i,s in enumerate(seeds):
                rel=f'{era}/{model}/seed_{s}/'
                data=(lambda n:raw[prefix+rel+n]) if model=='EA' else (lambda n:(ce.OUT/rel/n).read_bytes())
                result=json.loads(data('result.json'))
                train_path=ea.run_path(s) if model=='EA' else cohort.OUT/'B0'/f'seed_{s}'
                train=read_json(train_path/'result.json')
                rid=identity if model=='EA' else read_json(ce.OUT/'completed.json')['identity']
                require(result['identity']==rid and result['status']=='completed' and result['model']==model
                    and result['seed']==s and result['period']==era and result['best_epoch']==train['best_epoch']
                    and result['checkpoint_sha256']==digest(train_path/'best.pt'),'Wrong evaluation checkpoint')
                require(set(result['artifacts'])=={'predictions_test.npz','metrics_per_basin.csv'},'Per-run roster differs')
                for n,h in result['artifacts'].items(): require(sha(data(n))==h,'Per-run hash differs')
                with np.load(io.BytesIO(data('predictions_test.npz')),allow_pickle=False) as z:records={k:z[k] for k in z.files}
                require(set(records)==set(obs)|{'q_predicted_mm_day'},'Unexpected prediction fields')
                for k in obs: require(np.array_equal(records[k],obs[k]),'Changed observation/date mask: '+k)
                pred=records['q_predicted_mm_day'].astype(float)
                require(pred.shape==o.shape and len(pred)==result['prediction_count']==manifest['observations'][era]['count']
                    and np.isfinite(pred).all() and (pred>=0).all(),'Invalid predictions')
                frame=pd.read_csv(io.BytesIO(data('metrics_per_basin.csv')),dtype={'basin_id':str},float_precision='round_trip')
                require(frame.basin_id.tolist()==basins,'Basin order differs')
                for j,(b,ix,fm) in enumerate(zip(basins,groups,masks)):
                    residual=pred[ix]-o[ix];sse=np.dot(residual,residual);centered=o[ix]-o[ix].mean()
                    nse[i,j]=1-sse/np.dot(centered,centered)
                    vals=[nse[i,j],np.abs(residual).mean(),np.sqrt(sse/len(ix))]
                    stored=frame.loc[j,['nse','mae_mm_day','rmse_mm_day']].to_numpy(float)
                    np.testing.assert_allclose(vals,stored,rtol=1e-12,atol=1e-10)
                    max_error=max(max_error,float(np.max(abs(np.array(vals)-stored))))
                    for f,mask in fm.items():
                        mae[f][i,j]=np.abs(residual[mask]).mean()
                        metric_rows.append(dict(era=era,model=model,seed=s,basin_id=b,flow=f,n_days=int(mask.sum()),
                            mae=mae[f][i,j],nse=nse[i,j] if f=='all' else None))
                np.testing.assert_allclose([nse[i].mean(),np.median(nse[i])],
                    [result['summary']['mean_nse'],result['summary']['median_nse']],rtol=1e-12,atol=1e-10)
                runs.append(dict(era=era,model=model,seed=s,mean_nse=float(nse[i].mean()),median_nse=float(np.median(nse[i])),
                    best_epoch=result['best_epoch'],inference_seconds=result['seconds']))
            matrices[era,model,'nse','all']=nse
            for f in FLOWS:matrices[era,model,'mae',f]=mae[f]
        print(f'Verified {era} EA/B0 masks and independently recomputed all metrics',flush=True)
    for n in sorted(expected|{'completed.json'}):
        dest=ee.OUT/n;content=raw[prefix+n]
        if dest.exists(): require(dest.read_bytes()==content,'Changed local output: '+n)
        else: dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    OUT.mkdir(parents=True,exist_ok=True)
    save(metric_rows,'metrics.csv');save(runs,'run_summary.csv');save(counts,'flow_counts.csv')
    audit=dict(status='passed',EA_evaluations=20,B0_evaluations_reused=20,EA_sealed_files=len(expected),
        archive_sha256=digest(ARCHIVE),evaluation_identity=identity,independent_metric_max_abs_error=max_error,
        validation_max_abs_error=max(r['max_absolute_prediction_difference_mm_day'] for r in checks.values()),
        inference_hours=sum(r['inference_seconds'] for r in runs if r['model']=='EA')/3600,
        validation_recheck_hours=sum(r['seconds'] for r in checks.values())/3600,
        masks_and_observations_identical=True,all_best_weights_verified=True,environment=ctx['context']['environment'])
    save_json(OUT/'audit.json',audit)
    return matrices,basins,seeds,audit


def main():
    matrices,basins,seeds,audit_report=audit()
    attrs=pd.read_csv(ATTRIBUTES,dtype={'basin_id':str,'huc2':str}).set_index('basin_id')
    require((attrs.loc[basins,'role']=='test').all(),'Wrong basin attributes')
    regions=attrs.loc[basins,'huc2'].to_numpy();require(not pd.isna(regions).any(),'Missing regions')
    p=matrices['earlier','EA','nse','all']-matrices['earlier','B0','nse','all']
    q=matrices['later','EA','nse','all']-matrices['later','B0','nse','all']
    primary=rename(transfer.summarize(p,q,regions,repeats=5000,random_seed=20261029))
    detail=rename(transfer.details(p,q))
    # Independent deployment-policy calculation against both fixed references.
    policies=np.where(p[:,None,:]>0,matrices['later','EA','nse','all'][None,:,:],matrices['later','B0','nse','all'][None,:,:])
    against={m:policies-matrices['later',m,'nse','all'][None,:,:] for m in ('B0','EA')}
    for m in ('B0','EA'):
        same=np.array([against[m][i,i] for i in range(10)])
        other=np.array([against[m][i,np.arange(10)!=i].mean(0) for i in range(10)])
        np.testing.assert_allclose(same,detail['same_vs_'+m],atol=1e-12)
        np.testing.assert_allclose(other,detail['other_vs_'+m],atol=1e-12)
    # Expand bootstrap multiplicities explicitly, excluding equal ORIGINAL IDs.
    rng=np.random.default_rng(3192026);names,index=np.unique(regions,return_inverse=True);max_boot=0.
    for _ in range(30):
        si=rng.integers(0,10,10);gi=rng.integers(0,len(names),len(names))
        if len(set(si))<2:continue
        bi=np.concatenate([np.flatnonzero(index==g) for g in gi]);vs=against['B0'];ve=against['EA']
        sb=np.mean([vs[i,i,bi].mean() for i in si]);ob=np.mean([vs[i,j,bi].mean() for i in si for j in si if i!=j])
        se=np.mean([ve[i,i,bi].mean() for i in si]);oe=np.mean([ve[i,j,bi].mean() for i in si for j in si if i!=j])
        actual=transfer.weighted(p,q,np.bincount(si,minlength=10),np.bincount(bi,minlength=107))
        target=np.array([sb,ob,sb-ob,se,oe]);np.testing.assert_allclose(actual,target,atol=1e-12,rtol=1e-12)
        max_boot=max(max_boot,float(abs(actual-target).max()))
    save([dict(metric=k,**v) for k,v in primary.items()],'primary_endpoint.csv')
    by_seed=[dict(seed=s,selected_EA_basins=int((p[i]>0).sum()),**{k:float(v[i].mean()) for k,v in detail.items()}) for i,s in enumerate(seeds)]
    by_basin=[dict(basin_id=b,HUC2=regions[j],**{k:float(v[:,j].mean()) for k,v in detail.items()}) for j,b in enumerate(basins)]
    save(by_seed,'selection_by_seed.csv');save(by_basin,'selection_by_basin.csv')
    distributions=[dict(metric=k,**distribution(v.mean(0))) for k,v in detail.items()]
    save(distributions,'basin_distributions.csv')
    loo=[]
    for i,s in enumerate(seeds):
        vals=transfer.weighted(np.delete(p,i,0),np.delete(q,i,0),np.ones(9),np.ones(107))
        loo.append(dict(kind='leave_one_seed_out',omitted=str(s),**dict(zip(NAMES,vals))))
    for region in sorted(set(regions)):
        keep=regions!=region;vals=transfer.weighted(p[:,keep],q[:,keep],np.ones(10),np.ones(keep.sum()))
        loo.append(dict(kind='leave_one_HUC2_out',omitted=region,**dict(zip(NAMES,vals))))
    save(loo,'leave_one_out.csv')
    pairs=[];flow_rows=[]
    for metric,flow in [('nse','all')]+[('mae',f) for f in FLOWS]:
        sign=1 if metric=='nse' else -1
        early=sign*(matrices['earlier','EA',metric,flow]-matrices['earlier','B0',metric,flow])
        late=sign*(matrices['later','EA',metric,flow]-matrices['later','B0',metric,flow])
        pairs.extend(reproducibility(early,late,metric,flow,seeds))
        if metric=='mae':
            vals=rename(transfer.summarize(early,late,regions,repeats=5000,random_seed=20261029))
            flow_rows.extend(dict(flow=flow,metric=k,exploratory=True,units='mm/day MAE reduction',**v) for k,v in vals.items())
    save(pairs,'reproducibility_pairs.csv');save(flow_rows,'secondary_flow_selection.csv')
    rep=pd.DataFrame(pairs).groupby(['metric','flow','kind'],sort=False).agg(n_pairs=('rho','size'),
        rho_mean=('rho','mean'),sign_agreement=('strict_sign_agreement','mean'),decision_agreement=('decision_agreement','mean'),
        marginal_independence=('marginal_independence','mean')).reset_index()
    rep.to_csv(OUT/'reproducibility_summary.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for era in ee.PERIODS:
        for model in ('B0','EA'):
            a=matrices[era,model,'nse','all'];medians=np.median(a,axis=1)
            comparisons.append(dict(era=era,model=model,mean_of_seed_basin_medians=float(medians.mean()),
                sd_of_seed_basin_medians=float(medians.std(ddof=1)),grand_mean_nse=float(a.mean()),minimum_nse=float(a.min())))
    save(comparisons,'model_comparison.csv')
    curves=selection_curve(p,q,(1,3,5,9));curve_rows=[];curve_seeds=[];curve_basins=[]
    for m,r in curves.items():
        da,db=direct_curve(p,q,m)
        np.testing.assert_allclose(da,r['vs_B0'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(db,r['vs_M1'],atol=1e-12,rtol=1e-12)
        if m==1:
            np.testing.assert_allclose([da.mean(),db.mean()],
                [primary['other_vs_B0']['estimate'],primary['other_vs_EA']['estimate']],atol=1e-12,rtol=1e-12)
        curve_rows.append(dict(m=m,mean_gain_vs_B0=float(da.mean()),mean_gain_vs_EA=float(db.mean()),
            basin_median_vs_B0=float(np.median(da.mean(0))),basin_median_vs_EA=float(np.median(db.mean(0))),
            positive_target_seeds_vs_B0=int((da.mean(1)>0).sum()),positive_target_seeds_vs_EA=int((db.mean(1)>0).sum()),
            EA_selection_fraction=float(r['selection_probability'].mean())))
        curve_seeds.extend(dict(m=m,target_seed=s,vs_B0=float(da[i].mean()),vs_EA=float(db[i].mean())) for i,s in enumerate(seeds))
        curve_basins.extend(dict(m=m,basin_id=b,vs_B0=float(da[:,j].mean()),vs_EA=float(db[:,j].mean())) for j,b in enumerate(basins))
    save(curve_rows,'multiseed_curve.csv');save(curve_seeds,'multiseed_by_target.csv');save(curve_basins,'multiseed_by_basin.csv')
    # Full-sample estimate remains primary; tail information is descriptive.
    dominant=sorted(by_basin,key=lambda r:abs(r['D_same_minus_other']),reverse=True)
    save(dominant,'basin_contributions_ranked.csv')
    total=sum(r['D_same_minus_other'] for r in dominant)
    tail=[dict(removed=k,basins=';'.join(r['basin_id'] for r in dominant[:k]),
        fraction_signed_total=sum(r['D_same_minus_other'] for r in dominant[:k])/total,
        remaining_mean_D=sum(r['D_same_minus_other'] for r in dominant[k:])/(107-k),posthoc_descriptive=True) for k in (1,3,5)]
    save(tail,'tail_influence_descriptive.csv')
    previous=read_json(ROOT/'outputs/retraining_confirmation_analysis_20261031/summary.json')['primary']
    side=[]
    for label,values in [('M1/B0',previous),('EA/B0',primary)]:
        for k,v in values.items():side.append(dict(comparison=label,endpoint=k.replace('M1','candidate').replace('EA','candidate'),**v))
    save(side,'comparison_with_M1.csv')
    protocol=read_json(ROOT/'protocol/ea_extension_20261031.json');crit=primary['D_same_minus_other']
    summary=dict(status='completed',primary=primary,seeds=seeds,basins=107,HUC2=len(set(regions)),
        supports_material_attenuation=crit['estimate']>=.01 and crit['low95']>0,
        lower_bound_above_reference=crit['low95']>.01,same_gain_positive_both_baselines=all(primary['same_vs_'+m]['low95']>0 for m in ('B0','EA')),
        other_gain_positive_both_baselines=all(primary['other_vs_'+m]['low95']>0 for m in ('B0','EA')),
        basin_distributions=distributions,positive_D_basins=sum(r['D_same_minus_other']>0 for r in by_basin),
        positive_D_seeds=sum(r['D_same_minus_other']>0 for r in by_seed),
        LOO_seed_D_range=[min(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_seed_out'),max(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_seed_out')],
        LOO_HUC2_D_range=[min(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_HUC2_out'),max(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_HUC2_out')],
        bootstrap_expansion_max_error=max_boot,multiseed_curve=curve_rows,model_comparison=comparisons,
        boundary=protocol['boundary'],uncertainty=protocol['uncertainty'])
    save_json(OUT/'summary.json',summary)
    np.savez_compressed(OUT/'primary_arrays.npz',early_delta_nse=p,later_delta_nse=q,basins=basins,seeds=seeds,regions=regions.astype(str))
    sources=[ARCHIVE,ee.BUNDLE,ATTRIBUTES,ROOT/'protocol/ea_extension_20261031.json',ROOT/'analysis/selection_transfer.py',
        ROOT/'analysis/multiseed_selection.py',Path(__file__),ROOT/'scripts/analyze_confirmation_evaluation.py']
    save_json(OUT/'source_hashes.json',{p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    locked=read_json(ROOT/'outputs/final_mechanism_audit_20261028/protocol_lock.json')['submission']
    require(all(digest(ROOT/p)==h for p,h in locked.items()),'Manuscript changed')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
