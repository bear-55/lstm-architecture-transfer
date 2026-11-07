"""Matched same-information global baseline, seed spread, early-defined failure groups."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import weakness_stats as ws
ROOT=ws.ROOT;OLD=ws.OUT;OUT=ROOT/'outputs/training_domain_followup_20261102'
CASES={'original107':OLD/'local_heldout','aligned107':OUT/'unseen_aligned','trained370':OLD/'local_internal'}

def main():
    reports=[];spreads=[];periods=[]
    for label,dest in CASES.items():
        with np.load(dest/'metric_arrays.npz') as z:a={k:z[k] for k in z.files}
        for j,model in enumerate(a['models'][1:],1):
            p=a['earlier'][j,:,:,0]-a['earlier'][0,:,:,0];q=a['later'][j,:,:,0]-a['later'][0,:,:,0]
            huc=a['regions'];groups=np.unique(huc);rng=np.random.default_rng(20261102);n=len(p)
            def calc(sw,bw):
                bw=bw/bw.sum();cross=(p>0)@(q*bw).T;global_rule=(p@bw)>0;g=np.outer(global_rule,q@bw)
                weights=np.outer(sw,sw);np.fill_diagonal(weights,0)
                same_local=sw@np.diag(cross)/sw.sum();other_local=(weights*cross).sum()/weights.sum()
                same_global=sw@np.diag(g)/sw.sum();other_global=(weights*g).sum()/weights.sum()
                return np.array([same_global,other_global,same_local-same_global,other_local-other_global])
            point=calc(np.ones(n),np.ones(len(huc)));draws=[]
            for _ in range(2000):
                while True:
                    sw=rng.multinomial(n,np.full(n,1/n))
                    if np.count_nonzero(sw)>1:break
                gw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)));bw=np.array([gw[np.flatnonzero(groups==g)[0]] for g in huc]);draws.append(calc(sw,bw))
            for k,key in enumerate(['same_global_vs_B0','other_global_vs_B0','same_local_minus_global','other_local_minus_global']):
                lo,hi=np.quantile(np.array(draws)[:,k],[.025,.975]);reports.append(dict(scenario=label,comparison=model+'-B0',endpoint=key,estimate=point[k],low95=lo,high95=hi))
            for era in ['earlier','later']:
                x=a[era][j,:,:,0];y=a[era][0,:,:,0]
                for scale,d in [('NSE',x-y),('NSE_star',x/(2-x)-y/(2-y))]:
                    sd=d.std(0,ddof=1);spreads.append(dict(scenario=label,comparison=model+'-B0',period=era,scale=scale,median_seed_sd=np.median(sd),mean_seed_sd=sd.mean(),max_seed_sd=sd.max()))
        t=pd.read_csv(dest/'three_factor.csv')
        for (c,s),g in t.groupby(['comparison','scale']):
            g=g.set_index('component');periods.append(dict(scenario=label,comparison=c,scale=s,
                triple=g.loc['period x seed x basin','fraction'],period_involving_sum=g.loc[[k for k in g.index if k.startswith('period')],'fraction'].sum(),seed_basin=g.loc['seed x basin','fraction']))
    pd.DataFrame(reports).to_csv(OUT/'global_baseline_endpoints.csv',index=False);pd.DataFrame(spreads).to_csv(OUT/'seed_spread.csv',index=False);pd.DataFrame(periods).to_csv(OUT/'period_components_aligned.csv',index=False)
    # Risk groups use only early mixed-ensemble performance.
    risk=[]
    for label,folder in [('local_heldout',None),('unseen_aligned',ROOT/'runs/domain_matched_20261102')]:
        ref=None;sims=[]
        for model in ['B0','M1','EA']:
            root=folder or ROOT/'runs'/('ea_evaluation_20261101' if model=='EA' else 'confirmation_evaluation_20261030')
            row=[]
            for seed in ws.SEEDS:
                with np.load(root/f'earlier/{model}/seed_{seed}/predictions_test.npz') as z:
                    if ref is None:ref={k:z[k] for k in ['basin_ids','q_observed_mm_day']}
                    row.append(z['q_predicted_mm_day'])
            sims.append(row)
        sims=np.array(sims);gains=pd.read_csv(OUT/f'{label}_basin_gains.csv',dtype={'basin_id':str})
        for j,model in enumerate(['M1','EA'],1):
            for bid in np.unique(ref['basin_ids']):
                ix=ref['basin_ids']==bid;o=ref['q_observed_mm_day'][ix].astype(float);sim=sims[[0,j]][:,:,ix].astype(float).mean((0,1));nse=1-((sim-o)**2).sum()/((o-o.mean())**2).sum()
                for r in gains[(gains.comparison==model+'-B0')&(gains.basin_id==bid)].itertuples():
                    risk.append(dict(scenario=label,comparison=model+'-B0',policy=r.policy,basin_id=bid,early_mixed_nse=nse,early_risk=bool(nse<0),later_gain=r.gain,later_selected_nse=r.selected_nse,later_mixed_nse=r.mixed_nse))
    risk=pd.DataFrame(risk);risk.to_csv(OUT/'early_defined_risk_per_basin.csv',index=False)
    risk.groupby(['scenario','comparison','policy','early_risk']).agg(n_basins=('basin_id','count'),mean_later_gain=('later_gain','mean'),median_later_gain=('later_gain','median')).to_csv(OUT/'early_defined_risk_summary.csv')
    print('GLOBAL_BASELINE_AND_RISK_DIAGNOSTICS_COMPLETE')

if __name__=='__main__':main()
