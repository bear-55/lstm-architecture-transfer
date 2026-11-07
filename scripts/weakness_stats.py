"""Shared CPU diagnostics for the 2026-11-02 checklist. Balanced REML, not SS ratios."""
from pathlib import Path
import itertools,json,math
import numpy as np
import pandas as pd
from scipy.stats import rankdata,f as fdist

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/weakness_resolution_20261102'
SEEDS=[1009,1021,1033,1051,1063,1087,1097,1109,1123,1151]
METRICS=['nse','kge','r','alpha','beta','fhv','flv','mae_low','mae_high','peak_day_bias_pct']

def score(obs,sim):
    ok=np.isfinite(obs)&np.isfinite(sim)&(obs>=0);o=np.asarray(obs[ok],float);s=np.asarray(sim[ok],float)
    if len(o)<2 or np.var(o)<=1e-12:return np.full(len(METRICS),np.nan)
    mu=o.mean();v=o.std();ss=s.std();r=float(np.corrcoef(o,s)[0,1]) if ss>0 else np.nan
    a=ss/v;b=s.mean()/mu if mu>0 else np.nan
    nse=1-np.sum((s-o)**2)/np.sum((o-mu)**2);kge=1-np.sqrt((r-1)**2+(a-1)**2+(b-1)**2)
    so=np.sort(o)[::-1];sp=np.sort(s)[::-1];nh=max(1,round(.02*len(o)));nl=max(1,round(.3*len(o)))
    fhv=100*(sp[:nh].sum()-so[:nh].sum())/so[:nh].sum()
    ol=np.log(np.where(so[-nl:]>0,so[-nl:],1e-6));sl=np.log(np.where(sp[-nl:]>0,sp[-nl:],1e-6))
    flv=-100*((sl-sl.min()).sum()-(ol-ol.min()).sum())/((ol-ol.min()).sum()+1e-6)
    lo=o<=np.quantile(o,.2);hi=o>=np.quantile(o,.95);peaks=o>=np.quantile(o,.98)
    peak=100*np.sum(s[peaks]-o[peaks])/np.sum(o[peaks])
    return np.array([nse,kge,r,a,b,fhv,flv,np.abs(s[lo]-o[lo]).mean(),np.abs(s[hi]-o[hi]).mean(),peak])

def reml(d):
    """Exact balanced intercept+crossed random factors REML by active-set enumeration.
    Centered eigenspaces: seed, basin, residual; variance eigenvalues e+B*s,
    e+S*b, e. The four possible boundary sets exhaust the feasible optimum.
    """
    s,b=d.shape;mu=d.mean();dfs=s-1;dfb=b-1;dfe=dfs*dfb
    sss=b*np.sum((d.mean(1)-mu)**2);ssb=s*np.sum((d.mean(0)-mu)**2)
    res=d-d.mean(1,keepdims=True)-d.mean(0,keepdims=True)+mu;sse=np.sum(res**2)
    mss,msb,mse=sss/dfs,ssb/dfb,sse/dfe
    options=[]
    for active_s,active_b in itertools.product([False,True],repeat=2):
        ve=(sse+(0 if active_s else sss)+(0 if active_b else ssb))/(dfe+(0 if active_s else dfs)+(0 if active_b else dfb))
        ve=max(ve,1e-15);vs=(mss-ve)/b if active_s else 0;vb=(msb-ve)/s if active_b else 0
        if min(vs,vb)<0:continue
        ls,lb=ve+b*vs,ve+s*vb
        ll=.5*(dfs*np.log(ls)+sss/ls+dfb*np.log(lb)+ssb/lb+dfe*np.log(ve)+sse/ve)
        options.append((ll,vb,vs,ve))
    ll,vb,vs,ve=min(options);total=vb+vs+ve
    return dict(v_basin=float(vb),v_seed=float(vs),v_residual=float(ve),basin_fraction=float(vb/total),
                seed_fraction=float(vs/total),residual_fraction=float(ve/total),mean_delta=float(mu),
                F_basin=float(msb/mse),F_p_iid_reference=float(fdist.sf(msb/mse,dfb,dfe)),
                restricted_nll=float(ll),n_seed=s,n_basin=b)

def anova3(x):
    components={0:x.mean(keepdims=True)};total=np.sum((x-x.mean())**2);rows=[]
    for mask in sorted(range(1,8),key=int.bit_count):
        axes=tuple(i for i in range(3) if not mask&(1<<i));c=x.mean(axis=axes,keepdims=True) if axes else x.copy()
        for sub,e in components.items():
            if sub&mask==sub:c=c-e
        components[mask]=c;ss=float(np.sum(np.broadcast_to(c,x.shape)**2))
        rows.append(dict(component=' x '.join(['period','seed','basin'][i] for i in range(3) if mask&(1<<i)),ss=ss,fraction=ss/total))
    np.testing.assert_allclose(sum(r['ss'] for r in rows),total,rtol=1e-10)
    return rows

def region_ids(basins):
    f=pd.read_csv(ROOT/'data/manifests/source_files.csv',dtype={'basin_id':str})
    mapping={r.basin_id:Path(r.path).parent.name for r in f[f.kind=='daymet'].itertuples()}
    return np.array([mapping[b] for b in basins])

def bootstrap_reml(d,regions,nboot=2000):
    rng=np.random.default_rng(20261102);s,b=d.shape;groups=np.unique(regions);out=[]
    for scheme in ['seed_basin','seed_HUC2']:
        for i in range(nboot):
            sr=rng.integers(s,size=s)
            br=rng.integers(b,size=b) if scheme=='seed_basin' else np.concatenate([np.flatnonzero(regions==g) for g in rng.choice(groups,size=len(groups))])
            v=reml(d[np.ix_(sr,br)]);v.update(scheme=scheme,draw=i);out.append(v)
    return pd.DataFrame(out)

def selection_boot(p,q,regions,nboot=2000):
    # Same original seed IDs are excluded even if bootstrap repeats a source row.
    n,b=p.shape;groups=np.unique(regions);rng=np.random.default_rng(20261102);rows=[]
    def calc(sw,bw):
        bw=bw/bw.sum();prob=p>0;cross=np.einsum('ib,jb,b->ij',prob,q,bw)
        weights=np.outer(sw,sw);np.fill_diagonal(weights,0)
        same=sw@np.diag(cross)/sw.sum();other=np.sum(weights*cross)/weights.sum()
        return same,other,same-other
    point=calc(np.ones(n),np.ones(b))
    for scheme in ['seed_basin','seed_HUC2']:
        for i in range(nboot):
            while True:
                sw=rng.multinomial(n,np.full(n,1/n))
                if np.count_nonzero(sw)>1:break
            if scheme=='seed_basin':bw=rng.multinomial(b,np.full(b,1/b))
            else:
                gw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)));bw=np.array([gw[np.flatnonzero(groups==g)[0]] for g in regions])
            rows.append((scheme,*calc(sw,bw)))
    df=pd.DataFrame(rows,columns=['scheme','same','other','D']);summary=[]
    for scheme,g in df.groupby('scheme'):
        for key,v in zip(['same','other','D'],point):
            lo,hi=np.quantile(g[key],[.025,.975]);summary.append(dict(scheme=scheme,endpoint=key,estimate=v,low95=lo,high95=hi))
    return summary,df

def read_local(which):
    if which=='internal':roots={m:ROOT/'runs/internal_temporal_extension_20261102' for m in ['B0','M1','EA']}
    else:roots={m:ROOT/'runs'/('ea_evaluation_20261101' if m=='EA' else 'confirmation_evaluation_20261030') for m in ['B0','M1','EA']}
    data={}
    for era in ['earlier','later']:
        ref=None;arrays=[]
        for model,root in roots.items():
            sims=[]
            for seed in SEEDS:
                with np.load(root/era/model/f'seed_{seed}'/'predictions_test.npz',allow_pickle=False) as z:
                    if ref is None:ref={k:z[k] for k in ['basin_ids','dates','q_observed_mm_day']}
                    else:
                        for k in ref:np.testing.assert_array_equal(ref[k],z[k])
                    sims.append(z['q_predicted_mm_day'])
            arrays.append(sims)
        p=np.asarray(arrays);basins=sorted(np.unique(ref['basin_ids']));series=[]
        for b in basins:
            ix=ref['basin_ids']==b;series.append((ref['q_observed_mm_day'][ix],p[:,:,ix]))
        data[era]=series
    return dict(name='local_'+which,models=['B0','M1','EA'],seeds=SEEDS,basins=basins,series=data)

def analyze(data,nboot=2000):
    name=data['name'];dest=OUT/name;dest.mkdir(parents=True,exist_ok=True)
    models=data['models'];seeds=data['seeds'];basins=data['basins'];regions=region_ids(basins);scores={};rows=[]
    for era,series in data['series'].items():
        a=np.empty((len(models),len(seeds),len(basins),len(METRICS)))
        for b,(obs,pred) in enumerate(series):
            # Freeze the common observation/simulation mask within period.
            mask=np.isfinite(obs)&(obs>=0)&np.isfinite(pred).all(axis=(0,1))
            for m,model in enumerate(models):
                for s,seed in enumerate(seeds):
                    a[m,s,b]=score(obs[mask],pred[m,s,mask]);rows.append(dict(era=era,model=model,seed=seed,basin_id=basins[b],n_days=int(mask.sum()),**dict(zip(METRICS,a[m,s,b]))))
        scores[era]=a
    pd.DataFrame(rows).to_csv(dest/'individual_metrics.csv',index=False)
    np.savez_compressed(dest/'metric_arrays.npz',earlier=scores['earlier'],later=scores['later'],metrics=METRICS,models=models,seeds=seeds,basins=basins,regions=regions)
    variance=[];cis=[];reliability=[];three=[];endpoints=[]
    for m,model in enumerate(models[1:],1):
        comparison=model+'-'+models[0]
        for era,a in scores.items():
            p=a[m,:,:,0]-a[0,:,:,0]
            e=scores['earlier'];pe=e[m,:,:,0]-e[0,:,:,0]
            good=(e[[0,m],:,:,0]>=0).all(axis=(0,1))
            drop=np.argsort(np.abs(pe.mean(0)))[:-5]
            variants={'NSE':p,'NSE_star':a[m,:,:,0]/(2-a[m,:,:,0])-a[0,:,:,0]/(2-a[0,:,:,0]),
                      'rank':rankdata(p,axis=1),'KGE':a[m,:,:,1]-a[0,:,:,1],
                      'early_nonnegative':p[:,good],'drop_early_top5':p[:,drop]}
            for scale,d in variants.items():
                finite=np.isfinite(d).all(axis=0);d=d[:,finite]
                v=reml(d);v.update(comparison=comparison,period=era,scale=scale);variance.append(v)
                if scale in ['NSE','NSE_star','rank','KGE']:
                    draws=bootstrap_reml(d,regions[finite],nboot)
                    draws.to_csv(dest/f'bootstrap_{model}_{era}_{scale}.csv',index=False)
                    for scheme,g in draws.groupby('scheme'):
                        for key in ['basin_fraction','seed_fraction','residual_fraction','v_basin','v_seed','v_residual']:
                            lo,hi=np.quantile(g[key],[.025,.975]);cis.append(dict(comparison=comparison,period=era,scale=scale,scheme=scheme,component=key,estimate=v[key],low95=lo,high95=hi))
                        for n in [1,2,3,5,8,10,16,20,30,50]:
                            r=v['v_basin']/(v['v_basin']+v['v_residual']/n)
                            rd=g.v_basin/(g.v_basin+g.v_residual/n);lo,hi=np.quantile(rd,[.025,.975])
                            n80=np.where(g.v_basin>0,4*g.v_residual/np.maximum(g.v_basin,1e-300),np.inf)
                            reliability.append(dict(comparison=comparison,period=era,scale=scale,scheme=scheme,n=n,rho=float(r),low95=lo,high95=hi,
                                                    n80_point=math.ceil(4*v['v_residual']/v['v_basin']) if v['v_basin']>0 else np.inf,
                                                    n80_finite_boot_fraction=float(np.isfinite(n80).mean()),extrapolated=n>len(seeds)))
            print('REML',name,comparison,era,flush=True)
        for scale in ['NSE','NSE_star']:
            a=np.stack([scores[e][m,:,:,0] for e in ['earlier','later']]);b=np.stack([scores[e][0,:,:,0] for e in ['earlier','later']])
            x=a-b if scale=='NSE' else a/(2-a)-b/(2-b)
            for r in anova3(x):r.update(comparison=comparison,scale=scale);three.append(r)
        p=scores['earlier'][m,:,:,0]-scores['earlier'][0,:,:,0];q=scores['later'][m,:,:,0]-scores['later'][0,:,:,0]
        summaries,draws=selection_boot(p,q,regions,nboot);draws.to_csv(dest/f'selection_bootstrap_{model}.csv',index=False)
        for r in summaries:r.update(comparison=comparison);endpoints.append(r)
    for fn,records in [('reml.csv',variance),('reml_intervals.csv',cis),('reliability.csv',reliability),('three_factor.csv',three),('selection_endpoints.csv',endpoints)]:pd.DataFrame(records).to_csv(dest/fn,index=False)
    policies=[]
    for m,model in enumerate(models[1:],1):
        pool_early=scores['earlier'][[0,m],:,:,0].reshape(2*len(seeds),len(basins))
        local=pool_early.argmax(0);glob=pool_early.mean(1).argmax();arch=pool_early.reshape(2,len(seeds),len(basins)).mean(1).argmax(0)
        for b,(obs,pred) in enumerate(data['series']['later']):
            p=pred[[0,m]].reshape(2*len(seeds),-1);means=pred[[0,m]].mean(1)
            strategies={'library_local':p[local[b]],'library_global':p[glob],'select_arch_then_ensemble':means[arch[b]],
                        'mixed_ensemble':p.mean(0),'B0_ensemble':means[0],model+'_ensemble':means[1]}
            for key,values in strategies.items():policies.append(dict(comparison=model+'-'+models[0],policy=key,basin_id=basins[b],**dict(zip(METRICS,score(obs,values)))))
            for j,mod in enumerate([models[0],model]):
                policies.append(dict(comparison=model+'-'+models[0],policy=mod+'_individual_seed_mean',basin_id=basins[b],**dict(zip(METRICS,scores['later'][[0,m][j],:,b,:].mean(0)))))
    pol=pd.DataFrame(policies);pol.to_csv(dest/'policy_hydrology_per_basin.csv',index=False)
    pol.groupby(['comparison','policy']).agg({key:['median'] if key=='flv' else ['mean','median'] for key in METRICS}).to_csv(dest/'policy_hydrology_summary.csv')
    (dest/'completed.json').write_text(json.dumps(dict(name=name,models=models,seeds=seeds,basins=len(basins),bootstrap_repetitions=nboot,status='completed',training=False),indent=2))
    print('ANALYSIS_COMPLETE',name,flush=True)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('scenario',choices=['heldout','internal']);ap.add_argument('--bootstrap',type=int,default=2000);args=ap.parse_args()
    analyze(read_local(args.scenario),args.bootstrap)
