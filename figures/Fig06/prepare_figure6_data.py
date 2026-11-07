"""Recompute Figure 6 from individual scores and verify against original results."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
DATA = HERE / "data"
SCORES = ARTICLE / "deliverables/data_request_20261103/01_scores_long.csv"
ARRAYS = ARTICLE / "outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz"
RESULTS = ARTICLE / "outputs/multimetric_transfer_20261102"
METRICS = ["NSE_star", "KGE", "abs_FHV", "nMAE_high", "nMAE_low"]
UNITS = ["NSE*", "KGE", "percentage points", "dimensionless", "dimensionless"]
CSV_METRIC = dict(zip(METRICS, ["NSE_star", "KGE", "abs_FHV_pct", "nMAE_high", "nMAE_low"]))


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with np.load(ARRAYS, allow_pickle=False) as z:
        original = {k: z[k] for k in z.files}
    basins, seeds = original["basins"], original["seeds"]
    assert list(original["models"]) == ["B0", "M1", "EA"]
    assert len(basins) == 107 and len(seeds) == 10
    n = len(seeds)
    scores = pd.read_csv(SCORES, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip")
    scores = scores[(scores.scenario == "heldout107") & scores.period.isin(["WY2000-2003", "WY2004-2008"])]
    assert len(scores) == 6420 and scores.role.eq("test").all()
    assert not scores.duplicated(["model", "seed", "basin_id", "period"]).any()
    np.testing.assert_array_equal(
        scores.drop_duplicates("basin_id").set_index("basin_id").loc[basins, "huc2"], original["regions"])

    def matrix(model, period, metric):
        t = scores[(scores.model == model) & (scores.period == period)]
        a = t.pivot(index="seed", columns="basin_id", values=metric).loc[seeds, basins].to_numpy()
        assert a.shape == (10, 107) and np.isfinite(a).all()
        return a

    # Verify score transformations before averaging instances or catchments.
    normalizers = pd.read_csv(RESULTS / "observation_normalizers.csv", dtype={"basin_id": str}, float_precision="round_trip")
    for mi, model in enumerate(original["models"]):
        for era, period in [("earlier", "WY2000-2003"), ("later", "WY2004-2008")]:
            a = original[era][mi]
            mean_q = normalizers[(normalizers.scenario == "unseen107") & (normalizers.era == era)]
            mean_q = mean_q.set_index("basin_id").loc[basins, "mean_q_mm_day"].to_numpy()
            mapping = {"NSE": a[..., 0], "NSE_star": a[..., 0] / (2 - a[..., 0]),
                       "KGE": a[..., 1], "abs_FHV_pct": np.abs(a[..., 5]),
                       "nMAE_high": a[..., 8] / mean_q, "nMAE_low": a[..., 7] / mean_q}
            for metric, values in mapping.items():
                np.testing.assert_allclose(matrix(model, period, metric), values, atol=1e-11, rtol=1e-12)

    stored = pd.read_csv(RESULTS / "transfer_per_basin.csv", dtype={"basin_id": str}, float_precision="round_trip")
    endpoints = pd.read_csv(RESULTS / "transfer_endpoints.csv", float_precision="round_trip")
    outcomes, summaries, distribution_stats, selections, checks, zero_corrections = [], [], [], [], [], []
    for mi, model in enumerate(["M1", "EA"], 1):
        x = matrix(model, "WY2000-2003", "NSE") - matrix("B0", "WY2000-2003", "NSE")
        choose = (x > 0).astype(float)
        np.testing.assert_array_equal(choose, original["earlier"][mi, ..., 0] > original["earlier"][0, ..., 0])
        constant_choice = (choose == choose[0]).all(axis=0)
        for si, seed in enumerate(seeds):
            for bi, basin in enumerate(basins):
                selections.append(dict(comparison=model + "/B0", seed=int(seed), basin_id=basin,
                                       historical_NSE_difference=x[si, bi], select_candidate=bool(choose[si, bi])))
        for metric, unit in zip(METRICS, UNITS):
            q = matrix(model, "WY2004-2008", CSV_METRIC[metric]) - matrix("B0", "WY2004-2008", CSV_METRIC[metric])
            if metric in ["abs_FHV", "nMAE_high", "nMAE_low"]:
                q = -q
            same = (choose * q).mean(axis=0)
            other = (choose.sum(0) * q.sum(0) - (choose * q).sum(0)) / (n * (n - 1))
            explicit = np.mean([choose[s] * q[t] for s in range(n) for t in range(n) if s != t], axis=0)
            covariance = ((choose - choose.mean(0)) * (q - q.mean(0))).sum(0) / (n - 1)
            np.testing.assert_allclose(other, explicit, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(same - other, covariance, atol=1e-12, rtol=1e-12)

            # If every source instance makes the same choice, same == other
            # algebraically. Enforce that identity before counting D_b > 0,
            # so roundoff near 1e-17 cannot become a positive-loss catchment.
            uncorrected_D = same - other
            other[constant_choice] = same[constant_choice]
            loss = same - other
            assert np.all(loss[constant_choice] == 0)
            assert np.all(np.abs(loss[~constant_choice]) > 1e-12)
            ref = stored[(stored.scenario == "unseen107") & (stored.selection_rule == "fixed_early_NSE")
                         & (stored.comparison == model + "-B0") & (stored.metric == metric)]
            ref = ref.set_index("basin_id").loc[basins]
            computed = np.column_stack([same, other, loss])
            np.testing.assert_allclose(computed, ref[["same", "other", "D"]], atol=1e-11, rtol=1e-12)
            refs = endpoints[(endpoints.scenario == "unseen107") & (endpoints.selection_rule == "fixed_early_NSE")
                             & (endpoints.comparison == model + "-B0") & (endpoints.metric == metric)]
            summary = dict(comparison=model + "/B0", metric=metric, unit=unit, n_catchments=len(basins),
                           G_same_mean=same.mean(), G_new_mean=other.mean(), D_mean=loss.mean(), D_median=np.median(loss),
                           n_D_positive=int((loss > 0).sum()), n_D_zero=int((loss == 0).sum()),
                           n_D_negative=int((loss < 0).sum()), share_D_positive=float((loss > 0).mean()))
            for name, vals in [("same", same), ("other", other), ("D", loss)]:
                for aggregation, reducer in [("mean", np.mean), ("basin_median", np.median)]:
                    row = refs[(refs.endpoint == name) & (refs.aggregation == aggregation)]
                    assert len(row) == 1
                    np.testing.assert_allclose(reducer(vals), row.estimate.iloc[0], atol=1e-11, rtol=1e-12)
                    if name == "D":
                        summary[f"D_{aggregation}_low95"] = row.low95.iloc[0]
                        summary[f"D_{aggregation}_high95"] = row.high95.iloc[0]
            summaries.append(summary)
            for bi, basin in enumerate(basins):
                outcomes.append(dict(comparison=model + "/B0", metric=metric, basin_id=basin,
                                     huc2=original["regions"][bi], G_same=same[bi], G_new=other[bi], D=loss[bi],
                                     candidate_selection_count=int(choose[:, bi].sum()),
                                     constant_choice=bool(constant_choice[bi])))
            for endpoint, values in [("G_same", same), ("G_new", other)]:
                quantiles = np.quantile(values, [0, .01, .25, .5, .75, .99, 1])
                distribution_stats.append(dict(comparison=model + "/B0", metric=metric, endpoint=endpoint,
                    **dict(zip(["min", "q01", "q25", "median", "q75", "q99", "max"], quantiles)),
                    n_below_display=int((values < quantiles[1]).sum()), n_above_display=int((values > quantiles[-2]).sum())))
            checks.append(dict(comparison=model + "/B0", metric=metric, catchments=107,
                               explicit_distinct_pairs=90, covariance_identity="PASS",
                               stored_results_max_abs_difference=float(np.max(np.abs(computed - ref[["same", "other", "D"]].to_numpy())))))
            zero_corrections.append(dict(comparison=model + "/B0", metric=metric,
                constant_choice_basins=basins[constant_choice].tolist(),
                raw_formula_positive_count=int((uncorrected_D > 0).sum()),
                stored_positive_count=int((ref.D > 0).sum()),
                exact_identity_positive_count=int((loss > 0).sum()),
                max_roundoff_removed=float(np.max(np.abs(uncorrected_D[constant_choice]), initial=0))))

    for filename, rows in [("per_catchment_gains.csv", outcomes), ("metric_summary.csv", summaries),
                           ("distribution_statistics.csv", distribution_stats), ("historical_choices.csv", selections)]:
        pd.DataFrame(rows).to_csv(DATA / filename, index=False)
    assert len(outcomes) == 1070 and len(distribution_stats) == 20 and len(selections) == 2140
    sources = [SCORES, ARRAYS, RESULTS / "transfer_per_basin.csv", RESULTS / "transfer_endpoints.csv",
               RESULTS / "observation_normalizers.csv"]
    report = dict(status="PASS", no_synthetic_data=True, new_training=False, n_catchments=107,
                  n_seeds_per_architecture=10, selection_metric="NSE", selection_period="WY2000-2003",
                  evaluation_period="WY2004-2008", mean_same_seed_pairs=10, mean_distinct_seed_pairs=90,
                  metric_transform_before_averaging=True, fixed_choice_for_all_metrics=True,
                  structural_zero_policy="D_b = 0 exactly whenever all source seeds select the same architecture",
                  uncertainty="No uncertainty bands in Figure 6; quartiles show between-catchment dispersion",
                  validation=checks, structural_zero_audit=zero_corrections,
                  input_hashes={str(p.relative_to(ARTICLE)): sha(p) for p in sources},
                  data_hashes={name: sha(DATA / name) for name in ["per_catchment_gains.csv", "metric_summary.csv",
                               "distribution_statistics.csv", "historical_choices.csv"]})
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries)[["comparison", "metric", "D_mean", "D_median", "n_D_positive", "share_D_positive"]].to_string(index=False))
    print("PASS: 1,070 per-catchment outcomes, 20 distributions and 60 aggregate endpoints verified.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
