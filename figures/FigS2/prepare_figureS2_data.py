"""Six catchments by three metrics: true seed contributions and their means."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
DATA = HERE / "data"
METRICS = ["NSE_star", "KGE", "nMAE_high"]
COMPARISONS = ["M1", "EA"]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    score_path = ARTICLE / "deliverables/data_request_20261103/01_scores_long.csv"
    array_path = ARTICLE / "outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz"
    reference_path = ARTICLE / "outputs/multimetric_transfer_20261102/transfer_per_basin.csv"
    metadata_path = ARTICLE / "deliverables/data_request_20261103/03_failure_cases.csv"
    reference = pd.read_csv(reference_path, dtype={"basin_id":str}, float_precision="round_trip")
    reference = reference[(reference.scenario=="unseen107") & (reference.selection_rule=="fixed_early_NSE")]
    with np.load(array_path,allow_pickle=False) as z:
        original={k:z[k] for k in z.files}
    seeds=list(original["seeds"])
    basins=list(original["basins"])
    assert len(seeds)==10 and len(basins)==107
    scores=pd.read_csv(score_path,dtype={"basin_id":str},float_precision="round_trip")
    scores=scores[(scores.scenario=="heldout107") & scores.period.isin(["WY2000-2003","WY2004-2008"])]
    assert len(scores)==6420

    def mat(model,period,metric):
        sub=scores[(scores.model==model)&(scores.period==period)]
        return sub.pivot(index="seed",columns="basin_id",values=metric).loc[seeds,basins].to_numpy()

    early_B0=mat("B0","WY2000-2003","NSE")
    selections={model:mat(model,"WY2000-2003","NSE")>early_B0 for model in COMPARISONS}
    matrices={}
    for model in ["B0"]+COMPARISONS:
        mi=list(original["models"]).index(model)
        np.testing.assert_allclose(mat(model,"WY2000-2003","NSE"),original["earlier"][mi,...,0],atol=1e-11)
        for metric in METRICS:
            values=mat(model,"WY2004-2008",metric)
            if metric=="NSE_star":
                nse=original["later"][mi,...,0]
                expected=nse/(2-nse)
            elif metric=="KGE":
                expected=original["later"][mi,...,1]
            else:
                expected=original["later"][mi,...,8]/mat(model,"WY2004-2008","observed_mean_mm_day")
            np.testing.assert_allclose(values,expected,atol=1e-11,rtol=1e-12)
            matrices[model,metric]=(-values if metric=="nMAE_high" else values)

    # The catchment rule uses all 107 EA/B0 NSE* losses, before plotting other metrics.
    q=matrices["EA","NSE_star"]-matrices["B0","NSE_star"]
    selected=selections["EA"]
    r=selected*q
    n=selected*(q.sum(0)-q)/9
    losses=(r-n).mean(0)
    losses[(selected==selected[0]).all(0)]=0
    loss_series=pd.Series(losses,index=basins)
    rr=reference[(reference.comparison=="EA-B0") & (reference.metric=="NSE_star")].set_index("basin_id").loc[basins]
    np.testing.assert_allclose(losses,rr.D,atol=1e-12)
    cases=[]
    for quantile in [.10,.25,.50,.75,.90]:
        target=float(np.quantile(losses,quantile))
        distance=(loss_series-target).abs()
        nearest=sorted(distance[distance<=distance.min()+1e-12].index)[0]
        cases.append(dict(row=len(cases)+1,basin_id=nearest,rule=f"Nearest P{int(100*quantile)}",
                          quantile_target=quantile,target_D=target,EA_NSE_star_D=float(loss_series[nearest])))
    meta=pd.read_csv(metadata_path,dtype={"basin_id":str})
    top3=meta.loc[meta.is_top3_net_gain_case,"basin_id"].tolist()
    failure=str(loss_series.loc[top3].idxmax())
    cases.append(dict(row=6,basin_id=failure,rule="Largest D among the three leading library-gain cases",
                      quantile_target=None,target_D=None,EA_NSE_star_D=float(loss_series[failure])))
    assert len(set(c["basin_id"] for c in cases))==6
    assert [c["basin_id"] for c in cases][2:3]==["12092000"]
    assert cases[4]["basin_id"]=="07083000" and failure=="08324000"

    contributions,means,pairs,checks=[],[],[],[]
    for case in cases:
        basin=case["basin_id"]
        bi=basins.index(basin)
        for metric in METRICS:
            for model in COMPARISONS:
                chosen=selections[model][:,bi]
                delta=matrices[model,metric][:,bi]-matrices["B0",metric][:,bi]
                full=chosen[:,None]*delta[None,:]
                same=np.diag(full)
                new=np.array([np.delete(full[s],s).mean() for s in range(10)])
                difference=same-new
                estimate=np.array([same.mean(),new.mean(),difference.mean()])
                if np.all(chosen==chosen[0]):
                    estimate[1]=estimate[0]
                    estimate[2]=0
                ref=reference[(reference.basin_id==basin)&(reference.metric==metric)&
                              (reference.comparison==model+"-B0")].iloc[0]
                np.testing.assert_allclose(estimate,[ref.same,ref.other,ref.D],atol=1e-11,rtol=1e-12)
                np.testing.assert_allclose(new.mean(),full[~np.eye(10,dtype=bool)].mean(),atol=1e-14)
                np.testing.assert_allclose(estimate[2],estimate[0]-estimate[1],atol=1e-14)
                np.testing.assert_allclose(estimate[2],np.cov(chosen.astype(float),delta,ddof=1)[0,1],atol=1e-14)
                for endpoint,values,point in zip(["G_same","G_new","D_b"],[same,new,difference],estimate):
                    means.append(dict(row=case["row"],basin_id=basin,metric=metric,comparison=model+"/B0",
                                      endpoint=endpoint,estimate=float(point),source_values=10,
                                      n_EA_or_M1_choices=int(chosen.sum())))
                    for si,(seed,value) in enumerate(zip(seeds,values)):
                        contributions.append(dict(row=case["row"],basin_id=basin,metric=metric,comparison=model+"/B0",
                                                  endpoint=endpoint,source_index=si+1,source_seed=int(seed),
                                                  select_candidate=bool(chosen[si]),value=float(value)))
                for si,s in enumerate(seeds):
                    for ti,t in enumerate(seeds):
                        pairs.append(dict(row=case["row"],basin_id=basin,metric=metric,comparison=model+"/B0",
                                          source_seed=int(s),evaluation_seed=int(t),same_seed=si==ti,
                                          selection_gain=float(full[si,ti])))
                checks.append(dict(basin_id=basin,metric=metric,comparison=model+"/B0",
                                   stored_results="PASS",distinct_pair_average="PASS",covariance_identity="PASS"))
    tables={"catchment_selection.csv":pd.DataFrame(cases),"mean_endpoints.csv":pd.DataFrame(means),
            "source_contributions.csv":pd.DataFrame(contributions),"all_seed_pairs.csv":pd.DataFrame(pairs),
            "all_catchment_selection_D.csv":pd.DataFrame({"basin_id":basins,"EA_NSE_star_D":losses})}
    assert len(means)==108 and len(contributions)==1080 and len(pairs)==3600
    for name,table in tables.items():
        table.to_csv(DATA/name,index=False,float_format="%.17g")
    inputs=[score_path,array_path,reference_path,metadata_path]
    audit=dict(status="PASS",no_synthetic_data=True,new_training=False,layout="6 catchments x 3 metrics",
               panels=18,mean_bars=108,source_values=1080,seed_pair_values=3600,metrics=METRICS,
               comparisons=[m+"/B0" for m in COMPARISONS],seeds=[int(s) for s in seeds],selected_catchments=cases,
               case_selection_basis="EA/B0 D_b on NSE* over all 107 held-out catchments; ties within 1e-12 broken by basin ID",
               selection_period="WY2000-2003",evaluation_period="WY2004-2008",fixed_selection_metric="NSE",
               error_metric_sign="Negated nMAE_high before computing gain: positive G means an error reduction",
               source_value_definitions={"G_same":"I_s * Delta_s","G_new":"I_s * mean(Delta_t, t != s)",
                                         "D_b":"Source contribution d_s = I_s * (Delta_s - mean(Delta_t, t != s)); mean(d_s) = D_b"},
               summary_rule="Arithmetic mean over all 10 sources, retaining choices of B0 as zero contributions",
               new_weight_rule="Each source averages nine distinct evaluation seeds; all-source mean equals the mean of 90 ordered pairs",
               confidence_intervals=False,checks=checks,
               input_hashes={str(path.relative_to(ARTICLE)):sha(path) for path in inputs},
               data_hashes={name:sha(DATA/name) for name in tables})
    (HERE/"data_verification.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ["status","panels","mean_bars","source_values","selected_catchments"]},indent=2))


if __name__=="__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
