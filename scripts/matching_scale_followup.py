"""P1: synchronized uncertainty, scale/influence sensitivity and early-score matching."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/selection_transfer_P1_20261102';PREV=ROOT/'outputs/training_domain_followup_20261102'
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def weighted_median(values,w):
    order=np.argsort(values,axis=1);v=np.take_along_axis(values,order,axis=1);ww=np.take_along_axis(w,order,axis=1);cs=ww.cumsum(1);half=ww.sum(1)/2
    lo=(cs<half[:,None]).sum(1);hi=(cs<=half[:,None]).sum(1);hi=np.minimum(hi,values.shape[1]-1)
    return (v[np.arange(len(v)),lo]+v[np.arange(len(v)),hi])/2
def main():
    z={'unseen107':load(PREV/'unseen_aligned/metric_arrays.npz'),'trained370':load(ROOT/'outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz')}
    for k in ['models','seeds']:np.testing.assert_array_equal(z['unseen107'][k],z['trained370'][k])
    rng=np.random.default_rng(20261102);sw=np.ones((2001,10));sw[1:]=rng.multinomial(10,np.full(10,.1),size=2000)
    for i in np.flatnonzero(np.count_nonzero(sw,axis=1)<2):
        while np.count_nonzero(sw[i])<2:sw[i]=rng.multinomial(10,np.full(10,.1))
    groups=np.unique(np.concatenate([a['regions'] for a in z.values()]));gw=np.ones((2001,len(groups)));gw[1:]=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    bw={key:gw[:,[list(groups).index(r) for r in a['regions']]] for key,a in z.items()}
    def outcomes(p,q):
        choose=(p>0).astype(float);same=sw@(choose*q)/sw.sum(1,keepdims=True)
        other=((sw@choose)*(sw@q)-(sw**2)@(choose*q))/(sw.sum(1)**2-(sw**2).sum(1))[:,None]
        return {'same':same,'other':other,'D':same-other}
    def aggregate(v,w,median=False):
        assert (w.sum(1)>0).all()
        return weighted_median(v,w) if median else (v*w).sum(1)/w.sum(1)
    records=[];deletions=[];perbasin=[];cache={};draws={}
    def report(a,**tags):
        lo,hi=np.quantile(a[1:],[.025,.975]);records.append(dict(**tags,estimate=a[0],low95=lo,high95=hi))
    oldgain=pd.read_csv(PREV/'unseen_aligned_basin_gains.csv',dtype={'basin_id':str});cg=oldgain[(oldgain.comparison=='EA-B0')&(oldgain.policy=='library_local')].sort_values('gain',ascending=False).basin_id.tolist()
    for mi,model in enumerate(['M1','EA'],1):
        for scale in ['NSE','NSE_star']:
            for key,a in z.items():
                e=a['earlier'][:,:,:,0];l=a['later'][:,:,:,0]
                if scale=='NSE_star':e=e/(2-e);l=l/(2-l)
                cache[(model,scale,key)]=outcomes(e[mi]-e[0],l[mi]-l[0])
                for bi,b in enumerate(a['basins']):perbasin.append(dict(scenario=key,comparison=model+'-B0',scale=scale,basin_id=b,**{k:v[0,bi] for k,v in cache[(model,scale,key)].items()}))
            for variant in ['mean','basin_median','drop_largest_D_3','drop_largest_D_7','drop_C_3','drop_C_7']:
                vals={}
                for key,a in z.items():
                    o=cache[(model,scale,key)];keep=np.ones(len(a['basins']),bool)
                    if variant.startswith('drop_largest_D'):
                        k=int(variant.split('_')[-1]);ix=np.argsort(-o['D'][0],kind='stable')[:k];keep[ix]=False
                    elif variant.startswith('drop_C') and key=='unseen107':keep=~np.isin(a['basins'],cg[:int(variant.split('_')[-1])])
                    for b in a['basins'][~keep]:deletions.append(dict(scenario=key,comparison=model+'-B0',scale=scale,variant=variant,basin_id=b))
                    vals[key]={endpoint:aggregate(v[:,keep],bw[key][:,keep],variant=='basin_median') for endpoint,v in o.items()}
                    for endpoint,v in vals[key].items():report(v,analysis='scale_influence',scenario=key,comparison=model+'-B0',scale=scale,variant=variant,endpoint=endpoint,n_basins=int(keep.sum()))
                for endpoint in ['same','other','D']:
                    v=vals['unseen107'][endpoint]-vals['trained370'][endpoint];report(v,analysis='scale_influence',scenario='unseen_minus_trained',comparison=model+'-B0',scale=scale,variant=variant,endpoint=endpoint,n_basins=477)
                    draws[f'{model}_{scale}_{variant}_{endpoint}']=v
    # Covariate and assignment use early B0 only, common to both model comparisons.
    raw={key:a['earlier'][0,:,:,0].mean(0) for key,a in z.items()};cov={key:x/(2-x) for key,x in raw.items()}
    caliper=.2*np.std(np.concatenate(list(cov.values())),ddof=1);u,t=cov['unseen107'],cov['trained370'];distance=np.abs(u[:,None]-t[None,:]);pairrows=[];balance=[];coverage=[]
    pooled=np.sqrt((np.var(u,ddof=1)+np.var(t,ddof=1))/2)
    def balance_row(label,ui,ti):
        xx=u[ui];yy=t[ti];v=dict(matching=label,n_pairs=len(ui),caliper=caliper,mean_distance=np.mean(np.abs(xx-yy)) if len(ui)==len(ti) else np.nan,SMD=(xx.mean()-yy.mean())/pooled,KS=ks_2samp(xx,yy).statistic)
        for name,x in [('unseen',raw['unseen107'][ui]),('trained',raw['trained370'][ti])]:
            v[name+'_raw_mean']=x.mean()
            for q in [0,.1,.25,.5,.75,.9,1]:v[f'{name}_raw_q{q}']=np.quantile(x,q)
        balance.append(v)
    balance_row('unmatched',np.arange(len(u)),np.arange(len(t)))
    for method in ['within_HUC2','across_HUC2']:
        allowed=distance<=caliper
        if method=='within_HUC2':allowed &= z['unseen107']['regions'][:,None]==z['trained370']['regions'][None,:]
        ui,ti=linear_sum_assignment(np.where(allowed,distance,1e6+distance));ok=allowed[ui,ti];ui=ui[ok];ti=ti[ok]
        assert len(set(ui))==len(ui)==len(set(ti));balance_row(method,ui,ti)
        for a,b in zip(ui,ti):pairrows.append(dict(matching=method,unseen_basin=z['unseen107']['basins'][a],trained_basin=z['trained370']['basins'][b],unseen_huc=z['unseen107']['regions'][a],trained_huc=z['trained370']['regions'][b],unseen_early_B0_NSE=raw['unseen107'][a],trained_early_B0_NSE=raw['trained370'][b],distance=distance[a,b],caliper=caliper))
        for i,b in enumerate(z['unseen107']['basins']):coverage.append(dict(matching=method,basin_id=b,matched=i in ui,early_B0_NSE=raw['unseen107'][i]))
        if method=='within_HUC2':pw=bw['unseen107'][:,ui]
        else:
            pw=np.ones((2001,len(ui)));pw[1:]=rng.multinomial(len(ui),np.full(len(ui),1/len(ui)),size=2000)
        for model in ['M1','EA']:
            for scale in ['NSE','NSE_star']:
                for endpoint in ['same','other','D']:
                    au=aggregate(cache[(model,scale,'unseen107')][endpoint][:,ui],pw);at=aggregate(cache[(model,scale,'trained370')][endpoint][:,ti],pw)
                    for label,v in [('unseen107',au),('trained370',at),('unseen_minus_trained',au-at)]:report(v,analysis='matching',scenario=label,comparison=model+'-B0',scale=scale,variant=method,endpoint=endpoint,n_basins=len(ui))
                    draws[f'match_{method}_{model}_{scale}_{endpoint}']=au-at
        print('MATCHED',method,len(ui),'of107',flush=True)
    pd.DataFrame(records).to_csv(OUT/'P1_endpoints.csv',index=False);pd.DataFrame(perbasin).to_csv(OUT/'P1_per_basin.csv',index=False);pd.DataFrame(deletions).to_csv(OUT/'P1_deleted_basins.csv',index=False)
    pd.DataFrame(pairrows).to_csv(OUT/'matching_pairs.csv',index=False);pd.DataFrame(balance).to_csv(OUT/'matching_balance.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'matching_coverage.csv',index=False)
    np.savez_compressed(OUT/'P1_contrast_draws.npz',**draws)
    print('P1_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
