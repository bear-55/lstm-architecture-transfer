"""A/C/E: aligned periods, policy failure concentration and bounded comparison cases."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,pandas as pd
from scipy.stats import rankdata
import weakness_stats as ws
ROOT=ws.ROOT;OLD=ws.OUT;OUT=ROOT/'outputs/training_domain_followup_20261102'

def load_arrays(scenario):
    with np.load(OLD/scenario/'metric_arrays.npz',allow_pickle=False) as z:return {k:z[k] for k in z.files}

def policy_audit(dest,label):
    p=pd.read_csv(dest/'policy_hydrology_per_basin.csv',dtype={'basin_id':str});summary=[];details=[];shares=[]
    for comp,g in p.groupby('comparison'):
        base=g[g.policy=='mixed_ensemble'].set_index('basin_id')
        for policy,gg in g.groupby('policy'):
            x=gg.set_index('basin_id').loc[base.index];n=x.nse.to_numpy();b=base.nse.to_numpy();gain=n-b
            summary.append(dict(scenario=label,comparison=comp,policy=policy,n_basins=len(n),mean_nse=n.mean(),median_nse=np.median(n),
                                mean_nse_star=np.mean(n/(2-n)),median_kge=x.kge.median(),nse_below_zero=int((n<0).sum()),nse_below_minus1=int((n<-1).sum()),
                                gain_vs_mix_mean=gain.mean(),gain_vs_mix_median=np.median(gain),positive_count=int((gain>0).sum()),
                                rescued_from_mix_negative=int(((b<0)&(n>=0)).sum()),new_failures_vs_mix=int(((b>=0)&(n<0)).sum()),
                                flv_median=x.flv.median()))
            if policy not in ['library_local','select_arch_then_ensemble']:continue
            order=np.argsort(-gain);positive=np.maximum(gain,0).sum();total=gain.sum()
            for k in [1,3,5,10]:
                top=order[:k];keep=order[k:]
                shares.append(dict(scenario=label,comparison=comp,policy=policy,top_k=k,net_gain_sum=total,
                                   top_gain=gain[top].sum(),share_of_positive_gain=gain[top].sum()/positive if positive else np.nan,
                                   share_of_net_gain=gain[top].sum()/total if total else np.nan,
                                   remaining_mean_gain=gain[keep].mean(),remaining_positive_count=int((gain[keep]>0).sum())))
            for i,bid in enumerate(base.index):
                details.append(dict(scenario=label,comparison=comp,policy=policy,basin_id=bid,selected_nse=n[i],mixed_nse=b[i],gain=gain[i],
                                    selected_kge=x.iloc[i].kge,mixed_kge=base.iloc[i].kge,rescued=bool(b[i]<0<=n[i]),new_failure=bool(n[i]<0<=b[i])))
    pd.DataFrame(summary).to_csv(OUT/f'{label}_policy_summary.csv',index=False)
    pd.DataFrame(details).sort_values(['comparison','policy','gain'],ascending=[True,True,False]).to_csv(OUT/f'{label}_basin_gains.csv',index=False)
    pd.DataFrame(shares).to_csv(OUT/f'{label}_gain_concentration.csv',index=False)
    return summary

def existing():
    OUT.mkdir(parents=True,exist_ok=True)
    for scenario in ['local_heldout','local_internal','external_NSE','external_MSE']:
        policy_audit(OLD/scenario,scenario)
        p=pd.read_csv(OLD/scenario/'policy_hydrology_per_basin.csv',dtype={'basin_id':str})
        p.groupby(['comparison','policy']).agg({k:['median'] if k=='flv' else ['mean','median'] for k in ws.METRICS}).to_csv(OLD/scenario/'policy_hydrology_summary.csv')
    # Paired 4-year windows within the already analysed 10-year period.
    data=ws.read_local('heldout');late=load_arrays('local_heldout');basins=data['basins'];models=data['models'];seeds=data['seeds']
    roots={m:ROOT/'runs'/('ea_evaluation_20261101' if m=='EA' else 'confirmation_evaluation_20261030') for m in models}
    with np.load(roots['B0']/'earlier/B0/seed_1009/predictions_test.npz') as z:dates=z['dates'];bids=z['basin_ids']
    dates_by_basin=[dates[bids==b] for b in basins];rows=[];allp=[]
    for year in range(1989,1996):
        start=np.datetime64(f'{year}-10-01');end=np.datetime64(f'{year+4}-10-01')
        early=np.empty((3,10,len(basins)))
        for bi,(o,p) in enumerate(data['series']['earlier']):
            mask=(dates_by_basin[bi]>=start)&(dates_by_basin[bi]<end);oo=o[mask];pp=p[:,:,mask]
            early[:,:,bi]=1-((pp-oo)**2).sum(2)/((oo-oo.mean())**2).sum()
        allp.append(early)
        for mi,m in enumerate(models[1:],1):
            ps=early[mi]-early[0];qs=late['later'][mi,:,:,0]-late['later'][0,:,:,0]
            sums,draws=ws.selection_boot(ps,qs,late['regions'])
            for r in sums:
                if r['scheme']=='seed_HUC2':rows.append(dict(selection_start=str(start),selection_end=str(end-np.timedelta64(1,'D')),comparison=m+'-B0',**r))
        print('FOUR_YEAR_WINDOW',year,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'historical_four_year_windows.csv',index=False)
    np.savez_compressed(OUT/'historical_four_year_scores.npz',early=np.array(allp),starts=np.arange(1989,1996),models=models,seeds=seeds,basins=basins)
    # E: same scales for architecture, objective and asymmetric conceptual cases.
    n=load_arrays('external_NSE');m=load_arrays('external_MSE');records=[]
    def add(a,b,period,case,category,scenario):
        for scale,x in [('NSE',a-b),('NSE_star',a/(2-a)-b/(2-b)),('rank',rankdata(a-b,axis=1))]:
            records.append(dict(period=period,case=case,category=category,scenario=scenario,scale=scale,**ws.reml(x)))
    for scenario in ['local_heldout','local_internal','external_NSE','external_MSE']:
        z=load_arrays(scenario)
        for era in ['earlier','later']:
            for j,mod in enumerate(z['models'][1:],1):
                if mod=='LSTM_dynamic':continue
                add(z[era][j,:,:,0],z[era][0,:,:,0],era,str(mod)+'-B0','architecture',scenario)
    for era in ['earlier','later']:
        for j,mod in enumerate(n['models']):add(n[era][j,:,:,0],m[era][j,:,:,0],era,str(mod)+' NSEloss-MSEloss','loss','external')
    conceptual=pd.read_csv(OLD/'external/conceptual_metrics.csv',dtype={'basin_id':str})
    for (case,era),g in conceptual.groupby(['model','period']):
        g=g.set_index('basin_id');ids=[b for b in n['basins'] if b in g.index and np.isfinite(g.loc[b,'nse'])]
        ix=[list(n['basins']).index(b) for b in ids];b=n[era][0,:,ix,0].T;a=np.broadcast_to(g.loc[ids,'nse'].to_numpy()[None,:],b.shape)
        add(a,b,era,case+'-LSTM','fixed_calibration_reference','external')
    pd.DataFrame(records).to_csv(OUT/'case_comparison_scales.csv',index=False)
    print('EXISTING_ANALYSES_COMPLETED',flush=True)

def joint_boot(scenarios,comparisons):
    # Synchronize original seed identities and HUC2 draws across scenarios.
    groups=np.unique(np.concatenate([v[2] for v in scenarios.values()]));rng=np.random.default_rng(20261102);n=10;rows=[];points={}
    def calc(p,q,sw,bw):
        bw=bw/bw.sum();cross=np.einsum('ib,jb,b->ij',p>0,q,bw);w=np.outer(sw,sw);np.fill_diagonal(w,0)
        a=sw@np.diag(cross)/sw.sum();b=(w*cross).sum()/w.sum();return np.array([a,b,a-b])
    for key,(p,q,huc) in scenarios.items():points[key]=calc(p,q,np.ones(n),np.ones(len(huc)))
    for draw in range(2000):
        while True:
            sw=rng.multinomial(n,np.full(n,1/n))
            if np.count_nonzero(sw)>1:break
        while True:
            gw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)))
            weights={key:np.array([gw[np.flatnonzero(groups==g)[0]] for g in h]) for key,(_,_,h) in scenarios.items()}
            if all(w.sum()>0 for w in weights.values()):break
        vals={key:calc(p,q,sw,weights[key]) for key,(p,q,h) in scenarios.items()}
        for label,a,b in comparisons:
            for endpoint,v in zip(['same','other','D'],vals[a]-vals[b]):rows.append(dict(draw=draw,contrast=label,endpoint=endpoint,value=v))
    df=pd.DataFrame(rows);summary=[]
    for label,a,b in comparisons:
        for endpoint,point in zip(['same','other','D'],points[a]-points[b]):
            draws=df[(df.contrast==label)&(df.endpoint==endpoint)].value
            lo,hi=np.quantile(draws,[.025,.975]);summary.append(dict(contrast=label,endpoint=endpoint,estimate=point,low95=lo,high95=hi))
    return pd.DataFrame(summary),df

def matched():
    results=list((ROOT/'runs/domain_matched_20261102/earlier').glob('*/seed_*/result.json'));assert len(results)==30
    checks=[]
    for p in results:
        r=json.loads(p.read_text());assert hashlib.sha256((p.parent/'predictions_test.npz').read_bytes()).hexdigest()==r['prediction_sha256'];checks.append(r)
    (ROOT/'runs/domain_matched_20261102/completed.json').write_text(json.dumps(dict(status='completed',evaluations=30,basins=107,training=False,results=checks),indent=2))
    data=ws.read_local('heldout');new=[];ref=None;pred=[]
    for model in data['models']:
        sims=[]
        for seed in data['seeds']:
            with np.load(ROOT/f'runs/domain_matched_20261102/earlier/{model}/seed_{seed}/predictions_test.npz',allow_pickle=False) as z:
                if ref is None:ref={k:z[k] for k in ['basin_ids','dates','q_observed_mm_day']}
                else:
                    for k in ref:np.testing.assert_array_equal(z[k],ref[k])
                sims.append(z['q_predicted_mm_day'])
        pred.append(sims)
    pred=np.array(pred)
    for b in data['basins']:
        mask=ref['basin_ids']==b;new.append((ref['q_observed_mm_day'][mask],pred[:,:,mask]))
    data['series']['earlier']=new;data['name']='unseen_aligned';ws.OUT=OUT;ws.analyze(data)
    policy_audit(OUT/'unseen_aligned','unseen_aligned')
    with np.load(OUT/'unseen_aligned/metric_arrays.npz') as z:aligned={k:z[k] for k in z.files}
    old=load_arrays('local_heldout');train=load_arrays('local_internal');summ=[]
    for j,model in enumerate(aligned['models'][1:],1):
        cases={name:(z['earlier'][j,:,:,0]-z['earlier'][0,:,:,0],z['later'][j,:,:,0]-z['later'][0,:,:,0],z['regions']) for name,z in [('aligned',aligned),('original',old),('trained',train)]}
        s,d=joint_boot(cases,[('aligned107_minus_trained370','aligned','trained'),('original107_minus_aligned107','original','aligned')]);s['comparison']=model+'-B0';summ.append(s)
        d.to_csv(OUT/f'joint_bootstrap_{model}.csv',index=False)
    pd.concat(summ).to_csv(OUT/'matched_contrasts.csv',index=False)
    print('MATCHED_ANALYSIS_COMPLETED',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['existing','matched']);a=ap.parse_args()
    existing() if a.stage=='existing' else matched()
