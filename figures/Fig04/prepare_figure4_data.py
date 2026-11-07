"""Prepare Figure 4 from measured scores and the existing paired bootstrap draws."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
SEEDS = [1009, 1021, 1033, 1051, 1063, 1087, 1097, 1109, 1123, 1151]
SOURCES = {
    "scores": "deliverables/data_request_20261103/01_scores_long.csv",
    "main": "outputs/jh_predraft_audit_20261102/main_NSE_endpoints.csv",
    "main_draws": "outputs/jh_predraft_audit_20261102/baseline_bootstrap_draws.npz",
    "multimetric": "outputs/multimetric_transfer_20261102/transfer_endpoints.csv",
    "multimetric_draws": "outputs/multimetric_transfer_20261102/bootstrap_draws.npz",
    "ratios": "outputs/evaluation_framing_20261102/descriptive_transfer_ratios.csv",
    "external_scores": "outputs/weakness_resolution_20261102/external_NSE/individual_metrics.csv",
    "external_arrays": "outputs/weakness_resolution_20261102/external_NSE/metric_arrays.npz",
    "external_endpoints": "outputs/weakness_resolution_20261102/external_NSE/selection_endpoints.csv",
    "external_draws": "outputs/weakness_resolution_20261102/external_NSE/selection_bootstrap_EA.csv",
    "external_eligibility": "outputs/weakness_resolution_20261102/external/eligibility.json",
    "windows": "outputs/training_domain_followup_20261102/historical_four_year_windows.csv",
    "decade": "outputs/weakness_resolution_20261102/local_heldout/selection_endpoints.csv",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readcsv(key):
    return pd.read_csv(ROOT / SOURCES[key], dtype={"basin_id": str}, float_precision="round_trip")


def stats(x, y):
    """Equal-catchment, equal-seed means; exclude identical original seed IDs."""
    n = x.shape[0]
    choose = x > 0
    same_b = (choose * y).mean(0)
    new_b = (choose.sum(0) * y.sum(0) - (choose * y).sum(0)) / (n * (n - 1))
    explicit_new = np.mean([choose[s] * y[t] for s in range(n) for t in range(n) if s != t], axis=0)
    np.testing.assert_allclose(new_b, explicit_new, rtol=1e-12, atol=1e-12)
    cov = ((choose - choose.mean(0)) * (y - y.mean(0))).sum(0) / (n - 1)
    np.testing.assert_allclose(same_b - new_b, cov, rtol=1e-12, atol=1e-12)
    global_choice = x.mean(1) > 0
    gs = np.mean(global_choice * y.mean(1))
    gn = np.mean([global_choice[s] * y[t].mean() for s in range(n) for t in range(n) if s != t])
    same, new = same_b.mean(), new_b.mean()
    return dict(same=same, other=new, D=same-new, global_same=gs, global_other=gn,
                incremental_same=same-gs, incremental_other=new-gn), same_b, new_b


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    scores, main_table, multi = readcsv("scores"), readcsv("main"), readcsv("multimetric")
    assert not scores.duplicated(["scenario", "model", "seed", "basin_id", "period"]).any()
    external = readcsv("external_scores")
    ext_end = readcsv("external_endpoints")
    ext_draws = readcsv("external_draws").query("scheme == 'seed_HUC2'")
    assert len(ext_draws) == 2000
    eligibility = json.loads((ROOT / SOURCES["external_eligibility"]).read_text(encoding="utf-8"))
    assert eligibility["selection_period"] == ["1989-10-01", "1994-09-30"]
    assert eligibility["evaluation_period"] == ["1994-10-01", "1999-09-30"]
    gain_rows, inc_rows, ratio_rows, basin_rows, window_rows, ref_rows, precision_rows = [], [], [], [], [], [], []
    saved_draws, points, metadata = {}, {}, []
    checked_intervals = 0

    def local_matrix(scenario, model, period, column="NSE"):
        sub = scores[(scores.scenario == scenario) & (scores.model == model) & (scores.period == period)]
        frame = sub.pivot(index="seed", columns="basin_id", values=column).sort_index().sort_index(axis=1)
        assert frame.index.tolist() == SEEDS
        assert np.isfinite(frame.to_numpy()).all()
        return frame

    def endpoint_row(group, model, endpoint, point, ref, draws, nbasin, nseed, key):
        nonlocal checked_intervals
        assert draws.shape == (2000,) and np.isfinite(draws).all()
        lo, hi = np.quantile(draws, [.025, .975])
        np.testing.assert_allclose([point, lo, hi], ref[["estimate", "low95", "high95"]].to_numpy(float), atol=1e-12, rtol=1e-10)
        checked_intervals += 1
        saved_draws[key] = draws
        return dict(group=group, comparison=model+"/B0", endpoint=endpoint, estimate=point,
                    low95=lo, high95=hi, n_catchments=nbasin, n_seeds=nseed, n_bootstrap=2000,
                    bootstrap_scheme="seed_HUC2", bootstrap_key=key)

    with np.load(ROOT / SOURCES["main_draws"], allow_pickle=False) as main_boot, \
            np.load(ROOT / SOURCES["multimetric_draws"], allow_pickle=False) as multi_boot, \
            np.load(ROOT / SOURCES["external_arrays"], allow_pickle=False) as ext_arrays:
        for group, scenario, nbasin, models in [("Held-out", "heldout107", 107, ["M1", "EA"]),
                                                ("Training", "trained370", 370, ["M1", "EA"]),
                                                ("External", "external_NSE", 531, ["EA"])]:
            for model in models:
                matrices = {}
                for mod in ["B0", model]:
                    for era, period in [("earlier", "WY2000-2003"), ("later", "WY2004-2008")]:
                        if group == "External":
                            sub = external[(external.model == mod) & (external.era == era)]
                            frame = sub.pivot(index="seed", columns="basin_id", values="nse").sort_index().sort_index(axis=1)
                            np.testing.assert_array_equal(frame.index, ext_arrays["seeds"])
                            np.testing.assert_array_equal(frame.columns, ext_arrays["basins"])
                            mi = list(ext_arrays["models"]).index(mod)
                            np.testing.assert_allclose(frame.to_numpy(), ext_arrays[era][mi, :, :, 0], atol=1e-12, rtol=1e-12)
                        else:
                            frame = local_matrix(scenario, mod, period)
                        matrices[mod, era] = frame
                ref_frame = matrices["B0", "earlier"]
                for frame in matrices.values():
                    assert frame.index.equals(ref_frame.index) and frame.columns.equals(ref_frame.columns)
                    assert np.isfinite(frame.to_numpy()).all()
                nseed = len(ref_frame)
                assert ref_frame.shape == (8 if group == "External" else 10, nbasin)
                x = (matrices[model, "earlier"] - matrices["B0", "earlier"]).to_numpy()
                y = (matrices[model, "later"] - matrices["B0", "later"]).to_numpy()
                point, same_b, new_b = stats(x, y)
                points[group, model] = point
                for bid, same, new in zip(ref_frame.columns, same_b, new_b):
                    basin_rows.append(dict(group=group, comparison=model+"/B0", basin_id=bid,
                                           G_same=same, G_new=new, D=same-new))
                for endpoint in ["same", "other"]:
                    key = group.replace("-", "_") + "__" + model + "__" + endpoint
                    if group == "Held-out":
                        ref = main_table[(main_table.comparison == model+"-B0") & (main_table.endpoint == endpoint)].iloc[0]
                        all_draws = main_boot[f"{model}__reselect_global_in_bootstrap__{endpoint}"]
                        np.testing.assert_allclose(all_draws[0], point[endpoint], atol=1e-12)
                        draws = all_draws[1:]
                    elif group == "Training":
                        ref = multi[(multi.scenario == "trained370") & (multi.comparison == model+"-B0")
                                    & (multi.selection_rule == "fixed_early_NSE") & (multi.metric == "NSE")
                                    & (multi.aggregation == "mean") & (multi.endpoint == endpoint)].iloc[0]
                        all_draws = multi_boot[f"trained370__{model}__fixed_early_NSE__NSE__mean__{endpoint}"]
                        np.testing.assert_allclose(all_draws[0], point[endpoint], atol=1e-12)
                        draws = all_draws[1:]
                    else:
                        ref = ext_end[(ext_end.comparison == "EA-B0") & (ext_end.scheme == "seed_HUC2") & (ext_end.endpoint == endpoint)].iloc[0]
                        draws = ext_draws[endpoint].to_numpy()
                    gain_rows.append(endpoint_row(group, model, endpoint, point[endpoint], ref, draws, nbasin, nseed, key))
                if group == "Held-out":
                    for endpoint in ["incremental_same", "incremental_other"]:
                        ref = main_table[(main_table.comparison == model+"-B0") & (main_table.endpoint == endpoint)].iloc[0]
                        draws = main_boot[f"{model}__reselect_global_in_bootstrap__{endpoint}"]
                        np.testing.assert_allclose(draws[0], point[endpoint], atol=1e-12)
                        inc_rows.append(endpoint_row(group, model, endpoint, point[endpoint], ref, draws[1:], nbasin, nseed, f"increment__{model}__{endpoint}"))
                    retained = saved_draws[f"Held_out__{model}__same"]
                    retrained = saved_draws[f"Held_out__{model}__other"]
                    assert np.all(retained > 0)
                    ratios = retrained / retained
                    ratio_ref = readcsv("ratios")
                    ratio_ref = ratio_ref[ratio_ref.comparison == model+"-B0"].iloc[0]
                    row = endpoint_row(group, model, "transfer_ratio", point["other"]/point["same"], ratio_ref,
                                       ratios, nbasin, nseed, f"ratio__{model}")
                    row["nonpositive_bootstrap_denominators"] = int(np.count_nonzero(retained <= 0))
                    ratio_rows.append(row)
                metadata.append(dict(group=group, comparison=model+"/B0", n_catchments=nbasin,
                                     n_seeds=nseed, seeds=ref_frame.index.tolist()))

    # Recompute windows in float64, as in the supplied plotting template.
    # The archive used float32 window NSE; record its reproducible differences.
    old_windows = readcsv("windows")
    decade = readcsv("decade")
    periods = sorted(scores.loc[scores.period_kind == "overlapping_four_year", "period"].unique())
    assert len(periods) == 7
    for model in ["M1", "EA"]:
        y = (local_matrix("heldout107", model, "WY2004-2008") - local_matrix("heldout107", "B0", "WY2004-2008")).to_numpy()
        for period in periods:
            x = (local_matrix("heldout107", model, period) - local_matrix("heldout107", "B0", period)).to_numpy()
            p, _, _ = stats(x, y)
            oldx = (local_matrix("heldout107", model, period, "NSE_legacy_window_float32")
                    - local_matrix("heldout107", "B0", period, "NSE_legacy_window_float32")).to_numpy()
            oldp, _, _ = stats(oldx, y)
            start = str(int(period[2:6])-1) + "-10-01"
            oldref = old_windows[(old_windows.selection_start == start) & (old_windows.comparison == model+"-B0") & (old_windows.endpoint == "D")].iloc[0]
            np.testing.assert_allclose(oldp["D"], oldref.estimate, atol=1e-12, rtol=1e-10)
            window_rows.append(dict(comparison=model+"/B0", selection_window=period, G_same=p["same"], G_new=p["other"], D=p["D"]))
            precision_rows.append(dict(comparison=model+"/B0", selection_window=period, plotted_float64_D=p["D"],
                                       legacy_float32_D=oldp["D"], choice_changes=int(np.count_nonzero((x>0) != (oldx>0)))))
        x = (local_matrix("heldout107", model, "WY1990-1999") - local_matrix("heldout107", "B0", "WY1990-1999")).to_numpy()
        p, _, _ = stats(x, y)
        ref = decade[(decade.comparison == model+"-B0") & (decade.scheme == "seed_HUC2") & (decade.endpoint == "D")].iloc[0]
        np.testing.assert_allclose(p["D"], ref.estimate, atol=1e-12)
        ref_rows.append(dict(comparison=model+"/B0", selection_window="WY1990-1999", G_same=p["same"], G_new=p["other"], D=p["D"]))
    tables = {"gains.csv": gain_rows, "global_increments.csv": inc_rows,
              "transfer_ratios.csv": ratio_rows, "window_losses.csv": window_rows,
              "window_references.csv": ref_rows, "per_catchment_gains.csv": basin_rows,
              "window_precision_check.csv": precision_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(DATA / name, index=False, float_format="%.17g")
    np.savez_compressed(DATA / "bootstrap_draws.npz", **saved_draws)
    pd.read_csv(ROOT / SOURCES["main"], float_precision="round_trip").to_csv(DATA / "endpoints.csv", index=False)
    assert len(gain_rows) == 10 and len(inc_rows) == 4 and len(window_rows) == 14 and len(ratio_rows) == 2
    assert all(row["D"] > 0 for row in window_rows)
    report = dict(status="PASS", no_synthetic_data=True, groups=metadata, gains=10, increments=4,
                  window_estimates=14, decade_references=2, ratios=2, bootstrap_interval_checks=checked_intervals,
                  bootstrap_draws_per_interval=2000, saved_draws_exclude_point_estimates=True,
                  primary_intervals="Existing synchronized seed/HUC2 bootstrap; global choices reselected within each resample for panel b",
                  training_intervals="Existing multimetric seed/HUC2 draws, synchronized with the local held-out results",
                  external_intervals="Existing external NSE-loss archive, eight seeds, 531 catchments, seed/HUC2 draws",
                  ratio_intervals="Ratios of paired retrained/retained bootstrap gains; no nonpositive denominators",
                  point_checks="Explicit source/evaluation seed pairs and covariance identity checked; points and interval bounds match stored results",
                  selection_period="WY2000-2003", evaluation_period="WY2004-2008",
                  external_eligibility=eligibility, window_precision=precision_rows,
                  display_notes="Panel c shows point estimates; overlapping windows are not independent replicates",
                  sources={key: dict(path=relative, sha256=sha(ROOT/relative)) for key, relative in SOURCES.items()},
                  data_hashes={p.name: sha(p) for p in sorted(DATA.iterdir()) if p.is_file()})
    (HERE/"data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(gain_rows)[["group", "comparison", "endpoint", "estimate", "low95", "high95"]].to_string(index=False))
    print(pd.DataFrame(inc_rows)[["comparison", "endpoint", "estimate", "low95", "high95"]].to_string(index=False))
    print(pd.DataFrame(ratio_rows)[["comparison", "estimate", "low95", "high95"]].to_string(index=False))
    print(pd.DataFrame(window_rows).to_string(index=False))
    print(f"PASS: {checked_intervals} intervals verified from existing paired bootstrap draws.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
