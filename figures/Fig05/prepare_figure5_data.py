"""Validate and prepare final source-pool curves using the archived 20,000 draws."""
from hashlib import sha256
from itertools import combinations
from math import comb
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
OLD = ROOT / "outputs/multiseed_budget_curve_20261104"
ARRAYS = ROOT / "outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz"
SCORES = ROOT / "deliverables/data_request_20261103/01_scores_long.csv"
REVIEW = ROOT / "outputs/word_v4_review_revision_20261104/numeric_review.json"
ENDPOINTS = ["G_same", "G_new", "G_global_new", "D", "increment_new"]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    spec = json.loads((OLD/"analysis_spec.json").read_text(encoding="utf-8"))
    verified = json.loads((OLD/"verification.json").read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert verified["status"] == review["status"] == "PASS"
    assert spec["bootstrap_draws"] == 20000 and spec["target_budget"] == 5
    assert sha(ARRAYS) == spec["metric_arrays_sha256"]
    for name, digest in review["source_hashes"].items():
        assert sha(OLD/name) == digest
    with np.load(ARRAYS, allow_pickle=False) as z:
        early, late = z["earlier"][:,:,:,0], z["later"][:,:,:,0]
        seeds, basins, regions = z["seeds"], z["basins"], z["regions"]
        assert z["models"].tolist() == ["B0", "M1", "EA"]
    assert early.shape == late.shape == (3, 10, 107)
    raw = pd.read_csv(SCORES, dtype={"basin_id": str}, float_precision="round_trip")
    for mi, model in enumerate(["B0", "M1", "EA"]):
        for period, values in [("WY2000-2003", early), ("WY2004-2008", late)]:
            sub = raw[(raw.scenario == "heldout107") & (raw.model == model) & (raw.period == period)]
            table = sub.pivot(index="seed", columns="basin_id", values="NSE").loc[seeds, basins]
            np.testing.assert_allclose(table.to_numpy(), values[mi], atol=1e-12, rtol=1e-12)
    stored = pd.read_csv(OLD/"budget_curve_endpoints.csv", float_precision="round_trip")
    stored_subsets = pd.read_csv(OLD/"source_subset_results.csv", float_precision="round_trip")
    pools = pd.read_csv(OLD/"pool_definitions.csv")
    groups, group_ids = np.unique(regions, return_inverse=True)
    assert len(groups) == 16
    counts = np.random.default_rng(spec["bootstrap_seed"]).multinomial(16, np.full(16, 1/16), size=20000)
    rows, source_rows, fits, changes, summaries, validations = [], [], [], [], [], []
    point_lookup, draw_lookup = {}, {}
    explicit_allocations = 0
    with np.load(OLD/"bootstrap_draws.npz", allow_pickle=False) as boot:
        names = list(boot["endpoint_names"])
        for mi, model in [(1, "M1"), (2, "EA")]:
            for k in range(1, 6):
                source_sets = np.array(list(combinations(range(10), k)))
                pool_rows = pools[pools.source_budget == k].sort_values("source_id")
                assert len(source_sets) == len(pool_rows) == comb(10, k)
                ed = (early[mi][source_sets] - early[0][source_sets]).mean(1)
                retained = (late[mi][source_sets] - late[0][source_sets]).mean(1)
                evaluation = ((late[mi] - late[0]).sum(0) - k * retained) / (10-k)
                choose, global_choose = ed > 0, ed.mean(1) > 0
                gs, gn = choose * retained, choose * evaluation
                gg = global_choose[:,None] * evaluation
                source_values = np.column_stack([gs.mean(1), gn.mean(1), gg.mean(1),
                                                  (gs-gn).mean(1), (gn-gg).mean(1)])
                point = source_values.mean(0)
                point_lookup[model, k] = dict(zip(ENDPOINTS, point))
                old_sub = stored_subsets[(stored_subsets.comparison == model+"-B0")
                                        & (stored_subsets.source_budget == k) & (stored_subsets.metric == "NSE")].sort_values("source_id")
                np.testing.assert_allclose(source_values, old_sub[ENDPOINTS].to_numpy(), atol=1e-12, rtol=1e-12)
                # Enumerate all actual five-instance evaluation subsets. Their
                # arithmetic mean is exactly the non-source complement mean.
                for sid, source in enumerate(source_sets):
                    remain = [j for j in range(10) if j not in source]
                    targets = np.array(list(combinations(remain, 5)))
                    assert not np.isin(targets, source).any()
                    definition = pool_rows.iloc[sid]
                    assert definition.source_seeds == ";".join(str(int(seeds[j])) for j in source)
                    assert definition.available_target_seeds == ";".join(str(int(seeds[j])) for j in remain)
                    assert definition.target_budget == 5 and definition.target_subsets_per_source == len(targets)
                    explicit = (late[mi][targets] - late[0][targets]).mean(1)
                    np.testing.assert_allclose(explicit.mean(0), evaluation[sid], atol=1e-12, rtol=0)
                    explicit_allocations += len(targets)
                    source_rows.append(dict(comparison=model+"/B0", k=k, source_id=sid,
                                            source_seeds=definition.source_seeds,
                                            n_evaluation_subsets=len(targets),
                                            **dict(zip(ENDPOINTS, source_values[sid]))))
                values = boot[f"{model}__m{k}__NSE"]
                assert values.shape == (20000, 8) and np.isfinite(values).all()
                selected = values[:, [names.index(ep) for ep in ENDPOINTS]]
                draw_lookup[model, k] = selected
                np.testing.assert_allclose(selected[:,3], selected[:,0]-selected[:,1], atol=1e-12)
                np.testing.assert_allclose(selected[:,4], selected[:,1]-selected[:,2], atol=1e-12)
                # Independent direct witnesses of spatial resampling and
                # reselecting the global rule; no seed resampling is performed.
                for draw in [0, 9999, 19999]:
                    weights = counts[draw, group_ids].astype(float)
                    weights /= weights.sum()
                    reselected = (ed @ weights) > 0
                    a, b = np.mean(gs @ weights), np.mean(gn @ weights)
                    c = np.mean(reselected * (evaluation @ weights))
                    np.testing.assert_allclose([a, b, c, a-b, b-c], selected[draw], atol=1e-12, rtol=1e-12)
                for ei, ep in enumerate(ENDPOINTS):
                    lo, hi = np.quantile(selected[:,ei], [.025, .975])
                    upper = np.quantile(selected[:,ei], .95)
                    ref = stored[(stored.comparison == model+"-B0") & (stored.source_budget == k)
                                 & (stored.metric == "NSE") & (stored.endpoint == ep)].iloc[0]
                    np.testing.assert_allclose([point[ei], lo, hi, upper],
                                               ref[["estimate", "low95", "high95", "one_sided_upper95"]].to_numpy(float),
                                               atol=1e-12, rtol=1e-12)
                    rows.append(dict(comparison=model+"/B0", k=k, evaluation_instances=5,
                                     endpoint=ep, estimate=point[ei], low95=lo, high95=hi,
                                     one_sided_upper95=upper, n_source_subsets=comb(10,k),
                                     n_directed_allocations=comb(10,k)*comb(10-k,5), n_bootstrap=20000))
                validations.append(dict(comparison=model+"/B0", k=k, source_subsets=len(source_sets),
                                        directed_allocations=comb(10,k)*comb(10-k,5),
                                        explicit_target_enumeration="PASS", spatial_bootstrap_witnesses=3))
            kvals = np.arange(1,6,dtype=float)
            d = np.array([point_lookup[model,k]["D"] for k in range(1,6)])
            assert (d > 0).all() and (np.diff(d) < 0).all()
            intercept, slope = np.linalg.lstsq(np.column_stack([np.ones(5), np.log(kvals)]), np.log(d), rcond=None)[0]
            a, alpha = float(np.exp(intercept)), float(-slope)
            old_fit = review["results"][model]["decay"]["NSE"]
            np.testing.assert_allclose([a, alpha], [old_fit["a"], old_fit["alpha"]], atol=1e-12)
            fits.append(dict(comparison=model+"/B0", amplitude=a, alpha=alpha, k_min=1, k_max=5,
                             fit="Unweighted least squares of log(D) on log(k), including intercept"))
            p1, p5 = point_lookup[model,1], point_lookup[model,5]
            for ei, ep in enumerate(ENDPOINTS):
                diff = draw_lookup[model,5][:,ei] - draw_lookup[model,1][:,ei]
                lo, hi = np.quantile(diff, [.025,.975])
                changes.append(dict(comparison=model+"/B0", endpoint=ep, k_from=1, k_to=5,
                                    estimate=p5[ep]-p1[ep], low95=lo, high95=hi))
            decrease = p1["D"]-p5["D"]
            summaries.append(dict(comparison=model+"/B0", D_reduction_fraction=decrease/p1["D"],
                                  retained_gain_decrease=p1["G_same"]-p5["G_same"],
                                  transferred_gain_increase=p5["G_new"]-p1["G_new"],
                                  D_reduction_from_retained_fraction=(p1["G_same"]-p5["G_same"])/decrease,
                                  global_share_at_k5=p5["G_global_new"]/p5["G_new"]))
    table = pd.DataFrame(rows)
    assert len(table) == 50
    assert (table[table.endpoint == "D"].low95 > 0).all()
    for filename, values in [("curve_endpoints.csv", rows), ("power_fits.csv", fits),
                             ("source_subset_results.csv", source_rows), ("paired_changes_k1_k5.csv", changes),
                             ("summary.csv", summaries)]:
        pd.DataFrame(values).to_csv(DATA/filename, index=False, float_format="%.17g")
    inputs = [ARRAYS, SCORES, REVIEW, OLD/"analysis_spec.json", OLD/"verification.json",
              OLD/"budget_curve_endpoints.csv", OLD/"bootstrap_draws.npz",
              OLD/"pool_definitions.csv", OLD/"source_subset_results.csv"]
    report = dict(status="PASS", no_synthetic_data=True, n_catchments=107, n_huc2=16,
                  n_seeds_per_architecture=10, source_instances=[1,2,3,4,5], evaluation_instances=5,
                  selection_period="WY2000-2003", evaluation_period="WY2004-2008",
                  verified_endpoints=50, verified_bootstrap_replicates=20000,
                  explicit_directed_allocations_checked_across_both_comparisons=explicit_allocations,
                  all_source_subsets_kept_together=True, global_rule_reselected=True,
                  score_definition="Mean individual NSE; not NSE of mean discharge",
                  interval_definition="Pointwise 95% HUC2 percentile intervals conditional on the fixed fitted library; no seed resampling",
                  fit_definition="Descriptive power fit to five correlated means only within k=1..5; no extrapolation",
                  all_D_intervals_positive=True, validations=validations,
                  sources={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
                  data_hashes={p.name:sha(p) for p in sorted(DATA.glob("*.csv"))})
    (HERE/"data_verification.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(table[table.k.isin([1,5])][["comparison","k","endpoint","estimate","low95","high95","one_sided_upper95"]].to_string(index=False))
    print(pd.DataFrame(fits).to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))
    print("PASS: 50 endpoints and 20,000-draw intervals;",explicit_allocations,"explicit directed allocations checked.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
