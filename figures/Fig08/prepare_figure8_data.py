"""Real-data profiles and failure diagnostics for the fixed EA/B0 library."""
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
DATA = HERE / "data"
FOLLOWUP = ARTICLE / "outputs/training_domain_followup_20261102"
MULTI = ARTICLE / "outputs/multimetric_transfer_20261102"
WEAK = ARTICLE / "outputs/weakness_resolution_20261102"
FOLDERS = {"heldout": FOLLOWUP / "unseen_aligned", "training": WEAK / "local_internal", "external": WEAK / "external_NSE"}
POLICIES = ["library_local", "select_arch_then_ensemble", "mixed_ensemble"]
METRICS = ["nse", "kge", "r", "alpha_closeness", "beta_closeness", "abs_fhv", "mae_high", "mae_low"]
RANGES = [(0, .85), (0, .85), (.6, .95), (.5, 1), (0, 1), (50, 10), (4, 0), (.2, 0)]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    inputs, all_policy, selections, checks, summaries = [], [], [], [], []
    multi = pd.read_csv(MULTI / "policy_metrics_per_basin.csv", dtype={"basin_id": str}, float_precision="round_trip")
    inputs += [MULTI / "policy_metrics_per_basin.csv", MULTI / "policy_metric_summary.csv"]
    for role, folder in FOLDERS.items():
        a = load(folder / "metric_arrays.npz")
        table = pd.read_csv(folder / "policy_hydrology_per_basin.csv", dtype={"basin_id": str}, float_precision="round_trip")
        table = table[table.comparison == "EA-B0"].copy()
        assert not table.duplicated(["policy", "basin_id"]).any()
        n, basins = len(a["seeds"]), a["basins"]
        assert (n, len(basins)) == {"heldout": (10, 107), "training": (10, 370), "external": (8, 531)}[role]
        mi = list(a["models"]).index("EA")
        early = a["earlier"][[0, mi], ..., 0]
        local = early.reshape(2 * n, len(basins)).argmax(0)
        architecture = early.mean(1).argmax(0)
        later = a["later"][[0, mi]].reshape(2 * n, len(basins), -1)
        single = table[table.policy == "library_local"].set_index("basin_id").loc[basins]
        np.testing.assert_allclose(single[a["metrics"]], later[local, np.arange(len(basins))], atol=1e-11, rtol=1e-12)
        ensemble_rows = np.stack([table[table.policy == policy].set_index("basin_id").loc[basins, a["metrics"]].to_numpy()
                                  for policy in ["B0_ensemble", "EA_ensemble"]])
        chosen = table[table.policy == "select_arch_then_ensemble"].set_index("basin_id").loc[basins]
        np.testing.assert_allclose(chosen[a["metrics"]], ensemble_rows[architecture, np.arange(len(basins))], atol=1e-11, rtol=1e-12)
        for bi, basin in enumerate(basins):
            selections.append(dict(group=role, basin_id=basin, selected_model=["B0", "EA"][local[bi] // n],
                                   selected_seed=int(a["seeds"][local[bi] % n]), ensemble_architecture=["B0", "EA"][architecture[bi]],
                                   instances_per_architecture=n, library_size=2 * n))
        subset = table[table.policy.isin(POLICIES)].copy()
        assert len(subset) == len(basins) * 3 and np.isfinite(subset[a["metrics"]].to_numpy()).all()
        if role != "external":
            scenario = "unseen107" if role == "heldout" else "trained370"
            ref = multi[(multi.scenario == scenario) & (multi.comparison == "EA-B0") & multi.policy.isin(POLICIES)]
            columns = list(a["metrics"])
            np.testing.assert_allclose(subset.set_index(["policy", "basin_id"])[columns].sort_index(),
                                       ref.set_index(["policy", "basin_id"])[columns].sort_index(), atol=1e-11, rtol=1e-12)
        summary_path = FOLLOWUP / {"heldout": "unseen_aligned_policy_summary.csv", "training": "local_internal_policy_summary.csv", "external": "external_NSE_policy_summary.csv"}[role]
        reference = pd.read_csv(summary_path, float_precision="round_trip")
        reference = reference[(reference.comparison == "EA-B0") & reference.policy.isin(POLICIES)].set_index("policy")
        for policy in POLICIES:
            sub = subset[subset.policy == policy]
            row = dict(group=role, policy=policy, n_catchments=len(sub), mean_nse=sub.nse.mean(),
                       median_nse=sub.nse.median(), median_kge=sub.kge.median(), failures=int((sub.nse < 0).sum()))
            rr = reference.loc[policy]
            np.testing.assert_allclose([row["mean_nse"], row["median_nse"], row["median_kge"]],
                                       [rr.mean_nse, rr.median_nse, rr.median_kge], atol=1e-11, rtol=1e-12)
            assert row["failures"] == rr.nse_below_zero
            summaries.append(row)
        subset["group"] = role
        all_policy.append(subset)
        checks.append(dict(group=role, n_catchments=len(basins), instances_per_architecture=n,
                           selected_individual_all_metrics="PASS", selected_architecture_ensemble_all_metrics="PASS",
                           stored_summary="PASS"))
        inputs += [folder / "metric_arrays.npz", folder / "policy_hydrology_per_basin.csv", summary_path]

    policy_data = pd.concat(all_policy, ignore_index=True)
    policy_data["alpha_closeness"] = 1 - np.abs(policy_data.alpha - 1)
    policy_data["beta_closeness"] = 1 - np.abs(policy_data.beta - 1)
    policy_data["abs_fhv"] = np.abs(policy_data.fhv)
    risk_path = FOLLOWUP / "early_defined_risk_per_basin.csv"
    risk = pd.read_csv(risk_path, dtype={"basin_id": str}, float_precision="round_trip")
    risk = risk[(risk.scenario == "unseen_aligned") & (risk.comparison == "EA-B0") & (risk.policy == "library_local")].copy()
    assert len(risk) == risk.basin_id.nunique() == 107
    np.testing.assert_array_equal(risk.early_risk, risk.early_mixed_nse < 0)
    risk_ids = risk.loc[risk.early_risk, "basin_id"].tolist()
    assert len(risk_ids) == 7
    groups = {"typical_heldout": policy_data[(policy_data.group == "heldout") & ~policy_data.basin_id.isin(risk_ids)],
              "early_risk": policy_data[(policy_data.group == "heldout") & policy_data.basin_id.isin(risk_ids)],
              "training": policy_data[policy_data.group == "training"], "external": policy_data[policy_data.group == "external"]}
    profiles, range_rows = [], []
    for group, group_data in groups.items():
        for policy in POLICIES:
            sub = group_data[group_data.policy == policy]
            assert len(sub) == {"typical_heldout": 100, "early_risk": 7, "training": 370, "external": 531}[group]
            for metric, (inner, outer) in zip(METRICS, RANGES):
                value = float(sub[metric].median())
                normalized = (value - inner) / (outer - inner)
                assert 0 <= normalized <= 1, (group, policy, metric, value)
                profiles.append(dict(group=group, policy=policy, metric=metric, n_catchments=len(sub),
                                     median=value, axis_inner=inner, axis_outer=outer, radius=normalized))
    profile_table = pd.DataFrame(profiles)
    for metric, (inner, outer) in zip(METRICS, RANGES):
        sub = profile_table[profile_table.metric == metric]
        range_rows.append(dict(metric=metric, axis_inner=inner, axis_outer=outer, observed_profile_min=sub["median"].min(),
                               observed_profile_max=sub["median"].max(), clipping=False))

    heldout = policy_data[policy_data.group == "heldout"].pivot(index="basin_id", columns="policy", values="nse")
    gain = heldout.library_local - heldout.mixed_ensemble
    ordered = gain.sort_values(ascending=False, kind="stable")
    net_gain = float(gain.sum())
    assert net_gain > 0
    cumulative = ordered.cumsum() / net_gain * 100
    np.testing.assert_allclose(cumulative.iloc[-1], 100, atol=1e-10)
    top3 = ordered.index[:3].tolist()
    assert top3 == ["04127997", "08324000", "07301410"]
    concentration_ref = pd.read_csv(FOLLOWUP / "unseen_aligned_gain_concentration.csv", float_precision="round_trip")
    concentration_ref = concentration_ref[(concentration_ref.comparison == "EA-B0") & (concentration_ref.policy == "library_local")]
    for k in [1, 3, 5, 10]:
        rr = concentration_ref[concentration_ref.top_k == k].iloc[0]
        np.testing.assert_allclose([net_gain, ordered.iloc[:k].sum(), cumulative.iloc[k-1] / 100],
                                   [rr.net_gain_sum, rr.top_gain, rr.share_of_net_gain], atol=1e-11, rtol=1e-12)
    meta_path = ARTICLE / "deliverables/data_request_20261103/03_basin_metadata_531.csv"
    meta = pd.read_csv(meta_path, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip").set_index("basin_id").loc[heldout.index]
    np.testing.assert_allclose(heldout.library_local, meta.EA_library_selected_nse, atol=1e-11)
    np.testing.assert_allclose(heldout.mixed_ensemble, meta.EA_library_mixed_nse, atol=1e-11)
    np.testing.assert_allclose(gain, meta.EA_library_gain, atol=1e-11)
    rref = risk.set_index("basin_id").loc[heldout.index]
    np.testing.assert_allclose(rref.later_gain, gain, atol=1e-11)
    np.testing.assert_allclose(rref.later_selected_nse, heldout.library_local, atol=1e-11)
    np.testing.assert_allclose(rref.later_mixed_nse, heldout.mixed_ensemble, atol=1e-11)
    curve = pd.DataFrame(dict(rank=np.arange(1, 108), basin_id=ordered.index, gain=ordered.to_numpy(),
                              cumulative_gain=ordered.cumsum().to_numpy(), cumulative_net_share_pct=cumulative.to_numpy(),
                              top3=ordered.index.isin(top3), early_risk=ordered.index.isin(risk_ids)))
    failures = heldout[(heldout < 0).any(axis=1)].sort_values("mixed_ensemble")
    assert len(failures) == 3
    failures = failures[POLICIES].copy()
    selection_table = pd.DataFrame(selections)
    failure_choices = selection_table[selection_table.group == "heldout"].set_index("basin_id").loc[failures.index]
    failure_counts = []
    for policy in POLICIES:
        failure_counts.append(dict(policy=policy, n_catchments=107, failures=int((heldout[policy] < 0).sum()),
            rescued_vs_pool=int(((heldout[policy] >= 0) & (heldout.mixed_ensemble < 0)).sum()),
            new_failures_vs_pool=int(((heldout[policy] < 0) & (heldout.mixed_ensemble >= 0)).sum())))
    assert [r["failures"] for r in failure_counts] == [1, 3, 3]
    assert [r["rescued_vs_pool"] for r in failure_counts] == [2, 0, 0]
    assert [r["new_failures_vs_pool"] for r in failure_counts] == [0, 0, 0]
    for metric, source, transform in [("alpha_closeness", "alpha", lambda a: 1 - np.abs(a - 1)),
                                      ("beta_closeness", "beta", lambda a: 1 - np.abs(a - 1)),
                                      ("abs_fhv", "fhv", np.abs)]:
        np.testing.assert_allclose(policy_data[metric], transform(policy_data[source]), atol=0, rtol=0)

    policy_data.to_csv(DATA / "strategy_metrics_per_catchment.csv", index=False)
    profile_table.to_csv(DATA / "radar_profiles.csv", index=False)
    pd.DataFrame(range_rows).to_csv(DATA / "radar_axis_ranges.csv", index=False)
    pd.DataFrame(summaries).to_csv(DATA / "strategy_summary.csv", index=False)
    selection_table.to_csv(DATA / "historical_selections.csv", index=False)
    risk[["basin_id", "early_mixed_nse", "early_risk"]].to_csv(DATA / "early_risk_groups.csv", index=False)
    curve.to_csv(DATA / "gain_concentration.csv", index=False)
    failures.rename_axis("basin_id").reset_index().to_csv(DATA / "failure_matrix.csv", index=False)
    failure_choices.rename_axis("basin_id").reset_index().to_csv(DATA / "failure_case_selections.csv", index=False)
    pd.DataFrame(failure_counts).to_csv(DATA / "failure_counts.csv", index=False)
    external_eligibility = WEAK / "external/eligibility.json"
    eligibility = json.loads(external_eligibility.read_text(encoding="utf-8"))
    inputs += [risk_path, FOLLOWUP / "early_defined_risk_summary.csv", FOLLOWUP / "unseen_aligned_gain_concentration.csv",
               meta_path, external_eligibility, WEAK / "external_NSE/completed.json"]
    report = dict(status="PASS", no_synthetic_data=True, new_training=False, contrast="EA/B0 fixed fitted library",
                  selection_rule_single="Maximum early NSE among all available instances",
                  selection_rule_architecture="Maximum mean individual early NSE, followed by daily-discharge ensemble of that architecture",
                  pooled_rule="Mean daily discharge over all EA and B0 instances, before scoring",
                  local_instances_per_architecture=10, external_instances_per_architecture=8,
                  local_selection_period="WY2000-2003", local_evaluation_period="WY2004-2008",
                  external_selection_period="WY1990-1994", external_evaluation_period="WY1995-1999",
                  external_source=eligibility["source"], external_license=eligibility["license"],
                  groups={k: len(v) // 3 for k, v in groups.items()}, early_risk_criterion="Early EA/B0 pooled-ensemble NSE < 0",
                  early_risk_ids=risk_ids, radar_summary="Catchment median of each metric; transforms applied per catchment before medians",
                  radar_values=len(profiles), radar_clipping=False, radar_ranges=range_rows,
                  early_risk_share_of_net_gain_pct=float(gain.loc[risk_ids].sum() / net_gain * 100),
                  net_gain_mean=net_gain / 107, net_gain_sum=net_gain, top3_ids=top3,
                  top3_net_gain_share_pct=float(cumulative.iloc[2]), cumulative_peak_pct=float(cumulative.max()),
                  cumulative_peak_rank=int(np.argmax(cumulative.to_numpy()) + 1), cumulative_final_pct=float(cumulative.iloc[-1]),
                  failure_matrix_cells=9, failure_counts=failure_counts, source_validation=checks,
                  uncertainty="Descriptive medians and observed scores; no confidence bands in this figure",
                  input_hashes={str(p.relative_to(ARTICLE)): sha(p) for p in dict.fromkeys(inputs)},
                  data_hashes={p.name: sha(p) for p in sorted(DATA.iterdir()) if p.is_file()})
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(profile_table.pivot(index=["group", "policy"], columns="metric", values="median").to_string())
    print(failures.to_string())
    print(json.dumps({k: report[k] for k in ["status", "radar_values", "top3_net_gain_share_pct", "early_risk_share_of_net_gain_pct", "cumulative_peak_pct", "failure_counts"]}, indent=2))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
