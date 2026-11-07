"""Audit all forty new predictions and execute the frozen confirmation estimand."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'analysis')]
import confirmation_evaluate as ce
import retraining_confirmation as cohort
from final_evaluate import read_json, require, save_json
from project import digest
from selection_transfer import NAMES, details, summarize, weighted

OUT = ROOT/'outputs/retraining_confirmation_analysis_20261031'
ARCHIVE = ROOT/'runs/server backups/confirmation_evaluation_results_20261030.tar.gz'
ATTRIBUTES = ROOT/'outputs/ems_cpu_reanalysis_20261028/inputs/basin_attributes.csv'
FLOWS = ('all','low','mid','high')


def save(rows, name):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.15g')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def distribution(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()),
        q05=float(np.quantile(a,.05)), q25=float(np.quantile(a,.25)), q75=float(np.quantile(a,.75)),
        q95=float(np.quantile(a,.95)), max=float(a.max()), fraction_negative=float(np.mean(a<0)),
        fraction_positive=float(np.mean(a>0)))


def audit():
    cfg, meta, spec, manifest = ce.prepare()
    with tarfile.open(ARCHIVE,'r:gz') as archive:
        members = archive.getmembers()
        require(len({m.name for m in members}) == len(members), 'Duplicate archive member')
        for m in members:
            name = PurePosixPath(m.name)
            require(not name.is_absolute() and '..' not in name.parts and '\\' not in m.name
                    and (m.isfile() or m.isdir()), 'Unsafe archive member')
        raw = {m.name: archive.extractfile(m).read() for m in members if m.isfile()}
    prefix = 'runs/confirmation_evaluation_20261030/'
    read = lambda n: json.loads(raw[prefix+n])
    complete, context = read('completed.json'), read('evaluation_context.json')
    require(complete['status']=='completed' and complete['evaluations']==40 and complete['basins']==107,
            'Incomplete evaluation')
    identity = sha_bytes(json.dumps(context['context'], sort_keys=True).encode())
    require(identity == complete['identity'] == context['identity'], 'Evaluation identity mismatch')
    require(raw['confirmation_evaluation_bundle.json'] == ce.BUNDLE.read_bytes(), 'Evaluation bundle changed')
    expected_context = dict(bundle_sha256=digest(ce.BUNDLE),cohort_sha256=digest(cohort.OUT/'cohort_completed.json'),
        environment=read_json(cohort.OUT/'B0/seed_1009/run.json')['environment'],
        protocol_sha256=digest(ROOT/'protocol/confirmation.json'))
    require(context['context']==expected_context,'Execution environment or frozen inputs changed')
    expected_files = {'evaluation_context.json','validation_recheck.json','run_summary.csv'}
    expected_files |= {f'validation_checks/{m}/seed_{s}.json' for m,s in cohort.tasks(spec)}
    expected_files |= {f'{era}/{m}/seed_{s}/{n}' for era in ce.PERIODS for m,s in cohort.tasks(spec)
                      for n in ('result.json','predictions_test.npz','metrics_per_basin.csv')}
    require(set(complete['artifacts'])==expected_files,'Unexpected inference roster')
    for name,h in complete['artifacts'].items():
        require(sha_bytes(raw[prefix+name])==h,'Evaluation artifact changed: '+name)
    validation = read('validation_recheck.json')
    require(validation['identity']==identity and validation['status']=='passed' and len(validation['runs'])==20,
            'Missing full validation recheck')
    lookup={(r['model'],r['seed']):r for r in validation['runs']}
    require(set(lookup)==set(cohort.tasks(spec)), 'Validation roster mismatch')
    for m,s in cohort.tasks(spec):
        check=read(f'validation_checks/{m}/seed_{s}.json')
        train=read_json(cohort.OUT/m/f'seed_{s}'/'result.json')
        require(check==lookup[m,s] and check['status']=='passed' and check['identity']==identity
                and check['checkpoint_sha256']==digest(cohort.OUT/m/f'seed_{s}'/'best.pt')
                and check['prediction_count']==73050 and check['cloud_validation']==train['validation'],
                'Validation check identity mismatch')
        for k in ('median_nse','mean_nse'):
            require(abs(check['local_validation'][k]-train['validation'][k])<=1e-5,'Validation score mismatch')
    basins=meta['splits']['test']; seeds=spec['new_seeds']
    matrices={}; metric_rows=[]; run_rows=[]; counts=[]; maximum_difference=0.
    for era in ce.PERIODS:
        with np.load(ce.DATA/f'{era}_observations.npz',allow_pickle=False) as z:
            obs={k:z[k] for k in z.files}
        groups=[np.flatnonzero(obs['basin_ids']==b) for b in basins]
        o=obs['q_observed_mm_day'].astype(float)
        flow_masks=[]
        for b,ix in zip(basins,groups):
            q20,q95=np.quantile(o[ix],[.2,.95])
            high=o[ix]>=q95;low=~high&(o[ix]<=q20)
            masks=dict(all=np.ones(len(ix),bool),low=low,mid=~low&~high,high=high)
            require(all(mask.any() for mask in masks.values()),'Empty flow stratum')
            flow_masks.append(masks)
            counts.append(dict(era=era,basin_id=b,q20=q20,q95=q95,**{f'n_{f}':int(v.sum()) for f,v in masks.items()}))
        for name in cohort.MODELS:
            nse_matrix=np.empty((10,107));mae_matrices={f:np.empty((10,107)) for f in FLOWS}
            for i,seed in enumerate(seeds):
                relative=f'{era}/{name}/seed_{seed}/'
                result=read(relative+'result.json')
                train=read_json(cohort.OUT/name/f'seed_{seed}'/'result.json')
                require(result['identity']==identity and result['status']=='completed' and result['model']==name
                    and result['seed']==seed and result['period']==era and result['best_epoch']==train['best_epoch']
                    and result['checkpoint_sha256']==digest(cohort.OUT/name/f'seed_{seed}'/'best.pt'),
                    'Evaluation did not use selected checkpoint')
                require(set(result['artifacts'])=={'predictions_test.npz','metrics_per_basin.csv'}, 'Per-run artifact roster')
                for n,h in result['artifacts'].items():
                    require(sha_bytes(raw[prefix+relative+n])==h,'Per-run artifact hash')
                with np.load(io.BytesIO(raw[prefix+relative+'predictions_test.npz']),allow_pickle=False) as z:
                    records={k:z[k] for k in z.files}
                require(set(records)==set(obs)|{'q_predicted_mm_day'},'Unexpected prediction fields')
                for k in obs:
                    require(np.array_equal(records[k],obs[k]),'Different common mask or observed labels: '+k)
                pred=records['q_predicted_mm_day'].astype(float)
                require(pred.shape==o.shape and len(pred)==result['prediction_count']==manifest['observations'][era]['count']
                    and np.isfinite(pred).all() and (pred>=0).all(),'Invalid prediction')
                frame=pd.read_csv(io.BytesIO(raw[prefix+relative+'metrics_per_basin.csv']),dtype={'basin_id':str},float_precision='round_trip')
                require(frame.basin_id.tolist()==basins,'Metric basin order differs')
                for j,(basin,ix,masks) in enumerate(zip(basins,groups,flow_masks)):
                    residual=pred[ix]-o[ix]
                    # Independent SSE/SST formula instead of evaluator mean/variance implementation.
                    nse=1-np.dot(residual,residual)/np.dot(o[ix]-o[ix].mean(),o[ix]-o[ix].mean())
                    all_mae=np.abs(residual).mean();rmse=np.sqrt(np.dot(residual,residual)/len(ix))
                    nse_matrix[i,j]=nse
                    actual=np.array([nse,all_mae,rmse]);saved=frame.loc[j,['nse','mae_mm_day','rmse_mm_day']].to_numpy(float)
                    np.testing.assert_allclose(actual,saved,atol=1e-10,rtol=1e-12)
                    maximum_difference=max(maximum_difference,float(np.max(np.abs(actual-saved))))
                    for f,mask in masks.items():
                        mae=float(np.abs(residual[mask]).mean());mae_matrices[f][i,j]=mae
                        metric_rows.append(dict(era=era,model=name,seed=seed,basin_id=basin,flow=f,
                            n_days=int(mask.sum()),mae=mae,nse=float(nse) if f=='all' else None))
                np.testing.assert_allclose([nse_matrix[i].mean(),np.median(nse_matrix[i])],
                    [result['summary']['mean_nse'],result['summary']['median_nse']],atol=1e-10,rtol=1e-12)
                run_rows.append(dict(era=era,model=name,seed=seed,mean_nse=float(nse_matrix[i].mean()),
                    median_nse=float(np.median(nse_matrix[i])),best_epoch=result['best_epoch']))
            matrices[era,name,'nse','all']=nse_matrix
            for f in FLOWS:matrices[era,name,'mae',f]=mae_matrices[f]
        print(f'Verified all 20 {era} predictions and independently recomputed basin scores',flush=True)
    # Install only checked artifacts into a new evaluation directory; no remote code is executed.
    for n in sorted(expected_files|{'completed.json'}):
        dest=ce.OUT/n;content=raw[prefix+n]
        if dest.exists():require(dest.read_bytes()==content,'Different existing local evaluation: '+n)
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
    OUT.mkdir(parents=True,exist_ok=True)
    save(metric_rows,'metrics.csv');save(run_rows,'run_summary.csv');save(counts,'flow_counts.csv')
    report=dict(status='passed',evaluations=40,verified_files=len(expected_files),basins=107,
        archive_sha256=digest(ARCHIVE),evaluation_identity=identity,environment=context['context']['environment'],
        independent_metric_max_absolute_difference=maximum_difference,
        validation_recheck_max_absolute_difference=max(r['max_absolute_prediction_difference_mm_day'] for r in lookup.values()),
        new_training=False,all_observations_and_masks_identical=True,all_weights_validation_selected=True)
    save_json(OUT/'audit.json',report)
    return matrices,basins,seeds,report


def reproducibility(early,late,metric,flow,seeds):
    rows=[]
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            pairs=[('across_period_same_seed' if i==j else 'across_period_other_seed',early[i],late[j])]
            if i<j:pairs.append(('within_later_other_seed',late[i],late[j]))
            for kind,x,y in pairs:
                rho=np.corrcoef(pd.Series(x).rank().to_numpy(),pd.Series(y).rank().to_numpy())[0,1]
                xp,yp=(x>0).mean(),(y>0).mean()
                rows.append(dict(metric=metric,flow=flow,kind=kind,source_seed=seeds[i],target_seed=seeds[j],
                    rho=float(rho),strict_sign_agreement=float(np.mean(x*y>0)),
                    decision_agreement=float(np.mean((x>0)==(y>0))),
                    marginal_independence=float(xp*yp+(1-xp)*(1-yp))))
    return rows


def main():
    started=time.perf_counter()
    matrices,basins,seeds,audit_report=audit()
    attrs=pd.read_csv(ATTRIBUTES,dtype={'basin_id':str,'huc2':str}).set_index('basin_id')
    regions=attrs.loc[basins,'huc2'].to_numpy()
    require(not pd.isna(regions).any(),'Missing HUC2')
    require((attrs.loc[basins,'role']=='test').all(),'Wrong basin attributes')
    protocol=read_json(ROOT/'protocol/confirmation.json')
    p=matrices['earlier','M1','nse','all']-matrices['earlier','B0','nse','all']
    q=matrices['later','M1','nse','all']-matrices['later','B0','nse','all']
    primary=summarize(p,q,regions,repeats=5000,random_seed=protocol['analysis_random_seed'])
    detail=details(p,q)
    # A second direct policy implementation verifies the estimand and reference models.
    selected=np.empty((10,10,107));vs_b=np.empty_like(selected);vs_m=np.empty_like(selected)
    for i in range(10):
        for j in range(10):
            selected[i,j]=np.where(p[i]>0,matrices['later','M1','nse','all'][j],matrices['later','B0','nse','all'][j])
            vs_b[i,j]=selected[i,j]-matrices['later','B0','nse','all'][j]
            vs_m[i,j]=selected[i,j]-matrices['later','M1','nse','all'][j]
    same_b=np.array([vs_b[i,i] for i in range(10)])
    other_b=np.array([np.mean(vs_b[i,np.arange(10)!=i],axis=0) for i in range(10)])
    same_m=np.array([vs_m[i,i] for i in range(10)])
    other_m=np.array([np.mean(vs_m[i,np.arange(10)!=i],axis=0) for i in range(10)])
    for name,arr in zip(NAMES,[same_b,other_b,same_b-other_b,same_m,other_m]):
        np.testing.assert_allclose(arr,detail[name],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(arr.mean(),primary[name]['estimate'],atol=1e-12,rtol=1e-12)
    # Independently expand random bootstrap multiplicities to verify weighted formula.
    rng=np.random.default_rng(3192026);max_boot_error=0.
    group_names,group_index=np.unique(regions,return_inverse=True)
    for _ in range(30):
        selected_seeds=rng.integers(0,10,10); selected_groups=rng.integers(0,len(group_names),len(group_names))
        if len(np.unique(selected_seeds))<2:continue
        selected_basins=np.concatenate([np.flatnonzero(group_index==g) for g in selected_groups])
        same_a=np.array([vs_b[i,i,selected_basins].mean() for i in selected_seeds]).mean()
        same_c=np.array([vs_m[i,i,selected_basins].mean() for i in selected_seeds]).mean()
        other_a=np.array([vs_b[i,j,selected_basins].mean() for i in selected_seeds for j in selected_seeds if i!=j]).mean()
        other_c=np.array([vs_m[i,j,selected_basins].mean() for i in selected_seeds for j in selected_seeds if i!=j]).mean()
        expected=np.array([same_a,other_a,same_a-other_a,same_c,other_c])
        actual=weighted(p,q,np.bincount(selected_seeds,minlength=10),np.bincount(selected_basins,minlength=107))
        np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
        max_boot_error=max(max_boot_error,float(np.max(abs(actual-expected))))
    save([dict(metric=k,**v) for k,v in primary.items()],'primary_endpoint.csv')
    by_seed=[dict(seed=s,selected_M1_basins=int((p[i]>0).sum()),
        later_B0_mean_nse=float(matrices['later','B0','nse','all'][i].mean()),
        later_M1_mean_nse=float(matrices['later','M1','nse','all'][i].mean()),
        **{k:float(v[i].mean()) for k,v in detail.items()}) for i,s in enumerate(seeds)]
    save(by_seed,'selection_by_seed.csv')
    by_basin=[dict(basin_id=b,huc2=regions[j],**{k:float(v[:,j].mean()) for k,v in detail.items()}) for j,b in enumerate(basins)]
    save(by_basin,'selection_by_basin.csv')
    distributions=[dict(metric=k,aggregation='seed_mean_per_basin',**distribution(v.mean(0))) for k,v in detail.items()]
    distributions += [dict(metric=k,aggregation='basin_mean_per_source_seed',**distribution(v.mean(1))) for k,v in detail.items()]
    save(distributions,'effect_distributions.csv')
    loo=[]
    for i,s in enumerate(seeds):
        values=weighted(np.delete(p,i,0),np.delete(q,i,0),np.ones(9),np.ones(107))
        loo.append(dict(kind='leave_one_seed_out',omitted=str(s),**dict(zip(NAMES,values))))
    for region in sorted(set(regions)):
        keep=regions!=region
        values=weighted(p[:,keep],q[:,keep],np.ones(10),np.ones(keep.sum()))
        loo.append(dict(kind='leave_one_HUC2_out',omitted=region,**dict(zip(NAMES,values))))
    save(loo,'leave_one_out.csv')
    pair_rows=[];flow_rows=[]
    for metric,flow in [('nse','all')]+[('mae',f) for f in FLOWS]:
        sign=1 if metric=='nse' else -1
        early=sign*(matrices['earlier','M1',metric,flow]-matrices['earlier','B0',metric,flow])
        late=sign*(matrices['later','M1',metric,flow]-matrices['later','B0',metric,flow])
        pair_rows.extend(reproducibility(early,late,metric,flow,seeds))
        if metric=='mae':
            scores=summarize(early,late,regions,repeats=5000,random_seed=protocol['analysis_random_seed'])
            flow_rows.extend(dict(flow=flow,endpoint=k,units='mm/day MAE reduction',exploratory=True,**v) for k,v in scores.items())
    save(pair_rows,'reproducibility_pairs.csv');save(flow_rows,'secondary_flow_selection.csv')
    frame=pd.DataFrame(pair_rows)
    reps=frame.groupby(['metric','flow','kind'],sort=False).agg(n_pairs=('rho','size'),
        rho_mean=('rho','mean'),rho_median=('rho','median'),strict_sign_agreement=('strict_sign_agreement','mean'),
        decision_agreement=('decision_agreement','mean'),marginal_independence=('marginal_independence','mean')).reset_index()
    reps.to_csv(OUT/'reproducibility_summary.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    for era in ce.PERIODS:
        for m in cohort.MODELS:
            a=matrices[era,m,'nse','all'];seed_medians=np.median(a,axis=1)
            comparisons.append(dict(era=era,model=m,mean_of_seed_basin_medians=float(seed_medians.mean()),
                sd_of_seed_basin_medians=float(seed_medians.std(ddof=1)),grand_mean_nse=float(a.mean()),min_nse=float(a.min())))
    save(comparisons,'model_comparison.csv')
    dominant=sorted(by_basin,key=lambda r:abs(r['D_same_minus_other']),reverse=True)
    save(dominant,'basin_contributions_ranked.csv')
    tail=[]
    total_d=sum(r['D_same_minus_other'] for r in dominant)
    for k in (1,3,5):
        removed=sum(r['D_same_minus_other'] for r in dominant[:k])
        tail.append(dict(removed_top_absolute_contributions=k,
            removed_basins=';'.join(r['basin_id'] for r in dominant[:k]),
            fraction_of_full_signed_sum=removed/total_d,
            remaining_mean_D=(total_d-removed)/(107-k),
            interpretation='post-hoc descriptive tail diagnostic; not a replacement primary endpoint'))
    save(tail,'tail_influence_descriptive.csv')
    criterion=primary['D_same_minus_other']; reference=protocol['practical_reference_NSE']
    old=read_json(ROOT/'outputs/confirmation_preparation_20261029/design_checks.json')['old_cohort_only']
    report=dict(status='completed',new_cohort_only=True,primary=primary,
        supports_material_attenuation=criterion['estimate']>=reference and criterion['low95']>0,
        lower_bound_above_practical_reference=criterion['low95']>reference,
        reference_nse=reference,old_exploratory_cohort_separate=old,seeds=seeds,basins=107,HUC2_groups=len(set(regions)),
        same_weight_gain_positive_both_baselines=primary['same_vs_B0']['low95']>0 and primary['same_vs_M1']['low95']>0,
        point_estimate_other_positive_vs_B0=primary['other_vs_B0']['estimate']>0,
        point_estimate_other_positive_vs_M1=primary['other_vs_M1']['estimate']>0,
        distributions=distributions,
        LOO_seed_D_range=[float(min(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_seed_out')),
                          float(max(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_seed_out'))],
        LOO_HUC2_D_range=[float(min(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_HUC2_out')),
                          float(max(r['D_same_minus_other'] for r in loo if r['kind']=='leave_one_HUC2_out'))],
        bootstrap_expansion_checks=30,bootstrap_max_absolute_difference=max_boot_error,
        same_sign_at_least9_later_basins=int((((q>0).sum(0)>=9)|((q<0).sum(0)>=9)).sum()),
        boundary=protocol['boundary'],uncertainty=protocol['uncertainty'],elapsed_seconds=time.perf_counter()-started)
    save_json(OUT/'summary.json',report)
    np.savez_compressed(OUT/'primary_arrays.npz',early_delta_nse=p,later_delta_nse=q,basins=np.array(basins),seeds=seeds,regions=regions.astype(str))
    sources=[ARCHIVE,ATTRIBUTES,ROOT/'protocol/confirmation.json',ROOT/'analysis/selection_transfer.py',Path(__file__)]
    save_json(OUT/'source_hashes.json',{p.relative_to(ROOT).as_posix():digest(p) for p in sources})
    lock=read_json(ROOT/'outputs/final_mechanism_audit_20261028/protocol_lock.json')
    require(all(digest(ROOT/p)==h for p,h in lock['submission'].items()),'Manuscript changed')
    print(json.dumps(dict(primary=primary,supports_material_attenuation=report['supports_material_attenuation'],
        LOO_seed_D_range=report['LOO_seed_D_range'],LOO_HUC2_D_range=report['LOO_HUC2_D_range'],
        D_basin_distribution=distribution(detail['D_same_minus_other'].mean(0))),indent=2),flush=True)


if __name__=='__main__':main()
