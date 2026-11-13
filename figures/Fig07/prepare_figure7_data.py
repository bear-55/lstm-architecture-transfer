"""Extract measured and simulated daily discharge for the catchments in Figure 7."""
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
DATA = HERE / "data"
SOURCE = ARTICLE / "outputs/training_domain_followup_20261102/unseen_aligned"
sys.path.insert(0, str(ARTICLE / "scripts"))
from weakness_stats import score, METRICS


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def prediction_path(era, model, seed):
    prefix = "domain_matched_20261102" if era == "earlier" else (
        "ea_evaluation_20261101" if model == "EA" else "confirmation_evaluation_20261030")
    return ARTICLE / f"runs/{prefix}/{era}/{model}/seed_{seed}/predictions_test.npz"


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE / "metric_arrays.npz", allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    seeds, basins = a["seeds"], a["basins"]
    assert list(a["models"]) == ["B0", "M1", "EA"] and len(seeds) == 10 and len(basins) == 107
    early = a["earlier"][[0, 2], ..., 0]
    later = a["later"][[0, 2], ..., 0]
    selected = early[1] > early[0]
    d_tables = {}
    for scale in ["NSE", "NSE_star"]:
        values = later if scale == "NSE" else later / (2 - later)
        delta = values[1] - values[0]
        same = np.mean(selected * delta, axis=0)
        other = (selected.sum(0) * delta.sum(0) - (selected * delta).sum(0)) / 90
        other[(selected == selected[0]).all(0)] = same[(selected == selected[0]).all(0)]
        loss = same - other
        explicit_other = np.mean([selected[s] * delta[t] for s in range(10) for t in range(10) if s != t], axis=0)
        np.testing.assert_allclose(other, explicit_other, atol=1e-12)
        d_tables[scale] = pd.DataFrame({"basin_id": basins, "G_same": same, "G_new": other, "D": loss})
    reference_path = ARTICLE / "outputs/multimetric_transfer_20261102/transfer_per_basin.csv"
    stored = pd.read_csv(reference_path, dtype={"basin_id": str}, float_precision="round_trip")
    for scale, table in d_tables.items():
        ref = stored[(stored.scenario == "unseen107") & (stored.comparison == "EA-B0") &
                     (stored.selection_rule == "fixed_early_NSE") & (stored.metric == scale)]
        ref = ref.set_index("basin_id").loc[basins]
        np.testing.assert_allclose(table[["G_same", "G_new", "D"]], ref[["same", "other", "D"]], atol=1e-11, rtol=1e-12)
    d = d_tables["NSE_star"].set_index("basin_id")
    quantiles = np.quantile(d.D, [.5, .9])
    meta_path = ARTICLE / "deliverables/data_request_20261103/03_failure_cases.csv"
    failure_cases = pd.read_csv(meta_path, dtype={"basin_id": str}, float_precision="round_trip")
    top3 = failure_cases.loc[failure_cases.is_top3_net_gain_case, "basin_id"].tolist()
    assert set(top3) == {"04127997", "08324000", "07301410"}
    ids = [str((d.D - quantiles[0]).abs().idxmin()), str((d.D - quantiles[1]).abs().idxmin()),
           str(d.loc[top3, "D"].idxmax())]
    assert ids == ["12092000", "07083000", "08324000"]
    roles = ["median_D", "p90_D", "failure_case"]
    input_paths = [SOURCE / "metric_arrays.npz", reference_path, meta_path,
                   SOURCE / "policy_hydrology_per_basin.csv", ARTICLE / "scripts/weakness_stats.py"]

    # Select the displayed source on early scores. Seed 1009 provides a separate individual-score check.
    choices, source_seed, bi_list = [], {}, []
    evaluation_seed = 1009
    for ci, basin in enumerate(ids):
        bi = list(basins).index(basin)
        bi_list.append(bi)
        si = int(np.argmax(early[1, :, bi]))
        source_seed[basin] = int(seeds[si])
        assert source_seed[basin] != evaluation_seed
        assert early[1, si, bi] > early[0, si, bi]
        choices.append(dict(case=roles[ci], basin_id=basin, D_scale="NSE_star", D=float(d.loc[basin, "D"]),
                            D_NSE=float(d_tables["NSE"].set_index("basin_id").loc[basin, "D"]),
                            target_quantile=float(quantiles[ci]) if ci < 2 else None,
                            source_seed=source_seed[basin], evaluation_seed=evaluation_seed,
                            early_EA_NSE=float(early[1, si, bi]), early_paired_B0_NSE=float(early[0, si, bi]),
                            historical_choice="EA"))

    raw, validation, coverage, early_records = {}, [], [], []
    for era in ["earlier", "later"]:
        ref = None
        predictions = {}
        for model in ["B0", "EA"]:
            mi = list(a["models"]).index(model)
            for si, seed in enumerate(seeds):
                path = prediction_path(era, model, int(seed))
                input_paths.append(path)
                with np.load(path, allow_pickle=False) as z:
                    mask = np.isin(z["basin_ids"], ids)
                    sample = {k: z[k][mask] for k in ["basin_ids", "dates", "q_observed_mm_day"]}
                    if ref is None:
                        ref = sample
                    else:
                        for key in sample:
                            np.testing.assert_array_equal(ref[key], sample[key])
                    predictions[model, int(seed)] = z["q_predicted_mm_day"][mask]
                for basin, bi in zip(ids, bi_list):
                    m = ref["basin_ids"] == basin
                    actual = score(ref["q_observed_mm_day"][m], predictions[model, int(seed)][m])
                    target = a[era][mi, si, bi]
                    np.testing.assert_allclose(actual, target, atol=1e-10, rtol=1e-11)
                    validation.append(dict(era=era, model=model, seed=int(seed), basin_id=basin,
                                           n_days=int(m.sum()), max_abs_score_difference=float(np.max(np.abs(actual-target)))))
                    if era == "earlier":
                        early_records.append(dict(basin_id=basin, model=model, seed=int(seed),
                                                  nse=float(actual[0]), source_instance=int(seed) == source_seed[basin]))
        raw[era] = dict(reference=ref, predictions=predictions)

    ref, predictions = raw["later"]["reference"], raw["later"]["predictions"]
    for basin in ids:
        mask = ref["basin_ids"] == basin
        dates = pd.DatetimeIndex(ref["dates"][mask])
        assert dates.is_unique and dates.is_monotonic_increasing
        for year in range(2004, 2009):
            start, end = pd.Timestamp(year-1, 10, 1), pd.Timestamp(year, 9, 30)
            expected = pd.date_range(start, end)
            actual = dates[(dates >= start) & (dates <= end)]
            coverage.append(dict(basin_id=basin, water_year=year, n_days=len(actual), expected_days=len(expected),
                                 complete=np.array_equal(actual.to_numpy().astype("datetime64[D]"),
                                                         expected.to_numpy().astype("datetime64[D]"))))
    coverage = pd.DataFrame(coverage)
    common_complete_years = [year for year in range(2004, 2009) if coverage[coverage.water_year == year].complete.all()]
    hydrograph_year = min(common_complete_years)
    assert hydrograph_year == 2004

    policy = pd.read_csv(SOURCE / "policy_hydrology_per_basin.csv", dtype={"basin_id": str}, float_precision="round_trip")
    policy = policy[(policy.comparison == "EA-B0") & (policy.policy == "mixed_ensemble")].set_index("basin_id")
    # Original pipeline averages the twenty float32 predictions before scoring.
    pool = np.stack([predictions[model, int(seed)] for model in ["B0", "EA"] for seed in seeds]).mean(axis=0)
    rows, metric_rows, fdc_rows, ensemble_checks, high_flow_cutoffs = [], [], [], [], {}
    for basin in ids:
        mask = ref["basin_ids"] == basin
        dates = pd.DatetimeIndex(ref["dates"][mask])
        values = {
            "observed": ref["q_observed_mm_day"][mask],
            "B0_source": predictions["B0", source_seed[basin]][mask],
            "EA_retained": predictions["EA", source_seed[basin]][mask],
            "EA_new": predictions["EA", evaluation_seed][mask],
            "pooled_ensemble": pool[mask],
            "B0_evaluation_not_plotted": predictions["B0", evaluation_seed][mask],
        }
        frame = pd.DataFrame(dict(basin_id=basin, date=dates, **values))
        assert np.isfinite(frame[list(values)]).all().all()
        assert (frame[list(values)] > 0).all().all(), "A log-flow axis requires explicit handling of non-positive discharge"
        frame["water_year"] = dates.year + (dates.month >= 10)
        frame["in_hydrograph"] = frame.water_year == hydrograph_year
        rows.append(frame)
        full_pool_metrics = score(values["observed"], values["pooled_ensemble"])
        np.testing.assert_allclose(full_pool_metrics, policy.loc[basin, METRICS].to_numpy(dtype=float), atol=1e-10, rtol=1e-11)
        ensemble_checks.append(dict(basin_id=basin, n_instances=20, n_days=len(frame),
                                    max_abs_score_difference=float(np.max(np.abs(full_pool_metrics-policy.loc[basin, METRICS].to_numpy(dtype=float))))))
        for window, selected_frame in [("WY2004-2008", frame), (f"WY{hydrograph_year}", frame[frame.in_hydrograph])]:
            o = selected_frame.observed.to_numpy()
            for series in values:
                v = selected_frame[series].to_numpy()
                calculated = score(o, v)
                metric_rows.append(dict(basin_id=basin, period=window, series=series, n_days=len(v),
                                        **dict(zip(METRICS, calculated))))
        # Weibull plotting positions retain every available daily value.
        exceedance = np.arange(1, len(frame)+1) / (len(frame)+1) * 100
        high_count = max(1, round(.02 * len(frame)))
        high_flow_cutoffs[basin] = dict(n_high_flow_days=high_count, exceedance_pct=float(exceedance[high_count-1]))
        for series in list(values)[:5]:
            sorted_flow = np.sort(frame[series].to_numpy())[::-1]
            assert np.all(np.diff(sorted_flow) <= 0)
            for rank, (p, q) in enumerate(zip(exceedance, sorted_flow), start=1):
                fdc_rows.append(dict(basin_id=basin, series=series, rank=rank, n_days=len(frame),
                                     exceedance_pct=float(p), q_mm_day=float(q), top_2pct=rank <= high_count))
    combined = pd.concat(rows, ignore_index=True)
    assert combined[combined.in_hydrograph].groupby("basin_id").size().eq(366).all()
    datasets = {
        "case_selection.csv": pd.DataFrame(choices),
        "D_by_catchment.csv": pd.concat([table.assign(scale=scale) for scale, table in d_tables.items()], ignore_index=True),
        "early_instance_scores.csv": pd.DataFrame(early_records),
        "coverage_by_water_year.csv": coverage,
        "daily_predictions.csv": combined,
        "flow_duration_curves.csv": pd.DataFrame(fdc_rows),
        "case_metrics.csv": pd.DataFrame(metric_rows),
        "raw_score_validation.csv": pd.DataFrame(validation),
    }
    for name, frame in datasets.items():
        frame.to_csv(DATA / name, index=False, float_format="%.17g")
    audit = dict(
        status="PASS", no_synthetic_data=True, new_training=False, manuscript_modified=False,
        contrast="EA/B0", case_selection_scale="NSE_star", case_rules=["Closest to held-out median D_b",
        "Closest to held-out 90th percentile D_b", "Largest D_b among the three largest library-gain catchments"],
        selected_cases=choices, source_instance_rule="Highest WY2000-2003 EA NSE, then same-seed B0 reference",
        individual_score_check="EA seed 1009 in all three cases", displayed_independent_instances=9,
        evaluation_seed_selection_uses_late_results=False, seed_candidates=[int(s) for s in seeds],
        selection_period="WY2000-2003", hydrograph_year=hydrograph_year,
        hydrograph_year_rule="Earliest complete water year common to the three catchments in WY2004-2008",
        hydrograph_days_per_catchment=366, fdc_period="WY2004-2008", fdc_days_per_catchment=combined.groupby("basin_id").size().to_dict(),
        fdc_position="100 * rank / (n + 1), descending discharge, separately sorted for each series",
        fdc_top2_count_matches_FHV=True, fdc_high_flow_cutoffs=high_flow_cutoffs,
        discharge_units="mm/day", temporal_smoothing=False,
        missing_values_filled=False, transformed_or_clipped_discharge=False,
        all_discharge_positive=True, observed_and_simulated_dates_aligned=True,
        original_individual_score_checks=len(validation), individual_metrics_per_check=len(METRICS),
        individual_max_abs_score_difference=max(v["max_abs_score_difference"] for v in validation),
        pooled_ensemble_score_checks=ensemble_checks,
        input_hashes={str(path.relative_to(ARTICLE)): sha(path) for path in dict.fromkeys(input_paths)},
        data_hashes={name: sha(DATA / name) for name in datasets})
    (HERE / "data_verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:audit[k] for k in ["status", "selected_cases", "hydrograph_year", "fdc_days_per_catchment",
          "individual_max_abs_score_difference", "pooled_ensemble_score_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
