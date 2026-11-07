"""Compute catchment matching and covariate balance for Figures S1 and 7c."""
from hashlib import sha256
from pathlib import Path
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
DATA = HERE / "data"
P1 = ARTICLE / "outputs/selection_transfer_P1_20261102"
SCORES = ARTICLE / "deliverables/data_request_20261103/01_scores_long.csv"
META = ARTICLE / "deliverables/data_request_20261103/03_basin_metadata_531.csv"
ATTR_FEATHER = ARTICLE / "data/attributes/camels_attributes_v2.0.feather"
ATTR_CSV = ARTICLE / "data/attributes/static_attributes.csv"
ARRAY_PATHS = {
    "heldout": ARTICLE / "outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz",
    "training": ARTICLE / "outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz",
}
ATTRIBUTES = ["early_B0_NSEstar", "aridity", "frac_snow", "log_area", "baseflow_index", "p_mean", "elev_mean"]
SCHEMES = ["unmatched", "across_HUC2", "within_HUC2"]
N_BOOT, RNG_SEED = 2000, 20261102


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    arrays = {k: load(p) for k, p in ARRAY_PATHS.items()}
    h, t = arrays["heldout"], arrays["training"]
    assert len(h["basins"]) == 107 and len(t["basins"]) == 370
    assert not set(h["basins"]) & set(t["basins"])
    np.testing.assert_array_equal(h["seeds"], t["seeds"])
    assert list(h["models"]) == list(t["models"]) == ["B0", "M1", "EA"]
    ids = np.concatenate([h["basins"], t["basins"]])
    scores = pd.read_csv(SCORES, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip")
    scores = scores[scores.scenario.isin(["heldout107", "trained370"]) & scores.period.isin(["WY2000-2003", "WY2004-2008"])]
    assert len(scores) == 28620 and not scores.duplicated(["model", "seed", "basin_id", "period"]).any()
    for role, a in arrays.items():
        for mi, model in enumerate(a["models"]):
            for era, period in [("earlier", "WY2000-2003"), ("later", "WY2004-2008")]:
                sub = scores[(scores.model == model) & (scores.period == period) & scores.basin_id.isin(a["basins"])]
                raw = sub.pivot(index="seed", columns="basin_id", values="NSE").loc[a["seeds"], a["basins"]].to_numpy()
                np.testing.assert_allclose(raw, a[era][mi, ..., 0], atol=1e-11, rtol=1e-12)

    meta = pd.read_csv(META, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip").set_index("basin_id").loc[ids]
    original_attr = pd.read_feather(ATTR_FEATHER)
    aligned_attr = pd.read_csv(ATTR_CSV, dtype={"gauge_id": str, "huc_02": str}, float_precision="round_trip").set_index("gauge_id")
    attr_columns = ["aridity", "frac_snow", "area_geospa_fabric", "baseflow_index", "p_mean", "elev_mean", "gauge_lat", "gauge_lon"]
    attributes = original_attr.loc[ids, attr_columns].copy()
    assert attributes.index.is_unique and np.isfinite(attributes.to_numpy()).all()
    np.testing.assert_allclose(attributes, aligned_attr.loc[ids, attr_columns], atol=1e-9, rtol=1e-12)
    np.testing.assert_allclose(attributes[["gauge_lat", "gauge_lon"]], meta[["gauge_lat", "gauge_lon"]], atol=1e-7)
    assert (attributes.area_geospa_fabric > 0).all()
    attributes["log_area"] = np.log(attributes.area_geospa_fabric)
    attributes["huc2"] = meta.huc2
    raw_early, covariate = {}, {}
    for role, a in arrays.items():
        raw_early[role] = a["earlier"][0, ..., 0].mean(0)
        # Matching covariate = transform(mean NSE); outcome utility below is
        # instead transformed per instance before any averaging.
        covariate[role] = raw_early[role] / (2 - raw_early[role])
        attributes.loc[a["basins"], "role"] = role
        attributes.loc[a["basins"], "early_B0_mean_NSE"] = raw_early[role]
        attributes.loc[a["basins"], "early_B0_NSEstar"] = covariate[role]
        np.testing.assert_array_equal(attributes.loc[a["basins"], "huc2"], a["regions"])

    pairs = pd.read_csv(P1 / "matching_pairs.csv", dtype={"unseen_basin": str, "trained_basin": str, "unseen_huc": str, "trained_huc": str}, float_precision="round_trip")
    coverage = pd.read_csv(P1 / "matching_coverage.csv", dtype={"basin_id": str}, float_precision="round_trip")
    old_balance = pd.read_csv(P1 / "matching_balance.csv", float_precision="round_trip").set_index("matching")
    caliper = .2 * np.std(np.concatenate(list(covariate.values())), ddof=1)
    distance = np.abs(covariate["heldout"][:, None] - covariate["training"][None, :])
    indices = {"unmatched": (np.arange(107), np.arange(370))}
    mapping_checks = []
    for scheme, expected_n in [("within_HUC2", 77), ("across_HUC2", 101)]:
        pp = pairs[pairs.matching == scheme]
        assert len(pp) == pp.unseen_basin.nunique() == pp.trained_basin.nunique() == expected_n
        hi = np.array([list(h["basins"]).index(b) for b in pp.unseen_basin])
        ti = np.array([list(t["basins"]).index(b) for b in pp.trained_basin])
        indices[scheme] = hi, ti
        np.testing.assert_allclose(raw_early["heldout"][hi], pp.unseen_early_B0_NSE, atol=1e-12)
        np.testing.assert_allclose(raw_early["training"][ti], pp.trained_early_B0_NSE, atol=1e-12)
        np.testing.assert_allclose(distance[hi, ti], pp.distance, atol=1e-12)
        np.testing.assert_allclose(pp.caliper, caliper, atol=1e-12)
        assert (distance[hi, ti] <= caliper).all()
        allowed = distance <= caliper
        if scheme == "within_HUC2":
            allowed &= h["regions"][:, None] == t["regions"][None, :]
            np.testing.assert_array_equal(h["regions"][hi], t["regions"][ti])
        ui, vi = linear_sum_assignment(np.where(allowed, distance, 1e6 + distance))
        ok = allowed[ui, vi]
        assert set(zip(ui[ok], vi[ok])) == set(zip(hi, ti))
        cc = coverage[coverage.matching == scheme].set_index("basin_id").loc[h["basins"]]
        np.testing.assert_array_equal(cc.matched, np.isin(np.arange(107), hi))
        mapping_checks.append(dict(scheme=scheme, n_pairs=expected_n, unique_without_replacement=True,
                                   caliper_verified=True, original_assignment_reproduced=True))

    balances, distributions = [], []
    for attribute in ATTRIBUTES:
        u = attributes.loc[h["basins"], attribute].to_numpy()
        v = attributes.loc[t["basins"], attribute].to_numpy()
        pooled_sd = np.sqrt((u.var(ddof=1) + v.var(ddof=1)) / 2)
        assert pooled_sd > 0
        for scheme in SCHEMES:
            hi, ti = indices[scheme]
            xx, yy = u[hi], v[ti]
            smd = (xx.mean() - yy.mean()) / pooled_sd
            balances.append(dict(attribute=attribute, matching=scheme, n_heldout=len(hi), n_training=len(ti),
                                 heldout_mean=xx.mean(), training_mean=yy.mean(), full_sample_pooled_sd=pooled_sd, SMD=smd))
            if attribute == "early_B0_NSEstar":
                ref = old_balance.loc[scheme]
                np.testing.assert_allclose([smd, ks_2samp(xx, yy).statistic, caliper], [ref.SMD, ref.KS, ref.caliper], atol=1e-12)
                for role, selected in [("heldout", hi), ("training", ti)]:
                    vals = covariate[role][selected]
                    selected_ids = arrays[role]["basins"][selected]
                    order = np.argsort(vals, kind="stable")
                    for rank, index in enumerate(order, 1):
                        distributions.append(dict(matching=scheme, role=role, basin_id=selected_ids[index],
                                                  early_B0_NSEstar=vals[index], ecdf=rank / len(vals), n=len(vals)))

    # Reproduce all original synchronized bootstrap draws, including the
    # separate paired resampling used by the across-region 101-pair analysis.
    rng = np.random.default_rng(RNG_SEED)
    sw = np.ones((N_BOOT + 1, 10))
    sw[1:] = rng.multinomial(10, np.full(10, .1), size=N_BOOT)
    for i in np.flatnonzero(np.count_nonzero(sw, axis=1) < 2):
        while np.count_nonzero(sw[i]) < 2:
            sw[i] = rng.multinomial(10, np.full(10, .1))
    groups = np.unique(np.r_[h["regions"], t["regions"]])
    gw = np.ones((N_BOOT + 1, len(groups)))
    gw[1:] = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=N_BOOT)
    bw = {role: gw[:, [list(groups).index(r) for r in a["regions"]]] for role, a in arrays.items()}
    pair_weights = np.ones((N_BOOT + 1, 101))
    pair_weights[1:] = rng.multinomial(101, np.full(101, 1 / 101), size=N_BOOT)
    denom = sw.sum(1) ** 2 - (sw ** 2).sum(1)
    cached, per_basin = {}, []
    reference_basins = pd.read_csv(P1 / "P1_per_basin.csv", dtype={"basin_id": str}, float_precision="round_trip")
    for mi, model in enumerate(["M1", "EA"], 1):
        for role, a in arrays.items():
            early = a["earlier"][..., 0]
            late = a["later"][..., 0]
            choice = (early[mi] > early[0]).astype(float)
            transformed_early = early / (2 - early)
            np.testing.assert_array_equal(choice, transformed_early[mi] > transformed_early[0])
            utility = late / (2 - late)
            q = utility[mi] - utility[0]
            same = sw @ (choice * q) / sw.sum(1, keepdims=True)
            other = ((sw @ choice) * (sw @ q) - (sw ** 2) @ (choice * q)) / denom[:, None]
            loss = same - other
            explicit = np.mean([choice[s] * q[e] for s in range(10) for e in range(10) if s != e], axis=0)
            np.testing.assert_allclose(other[0], explicit, atol=1e-12)
            scenario = "unseen107" if role == "heldout" else "trained370"
            refs = reference_basins[(reference_basins.scenario == scenario) & (reference_basins.comparison == model + "-B0")
                                    & (reference_basins.scale == "NSE_star")].set_index("basin_id").loc[a["basins"]]
            np.testing.assert_allclose(np.c_[same[0], other[0], loss[0]], refs[["same", "other", "D"]], atol=1e-11, rtol=1e-12)
            cached[(model, role)] = dict(G_same=same, D=loss)
            for bi, basin in enumerate(a["basins"]):
                per_basin.append(dict(comparison=model + "/B0", role=role, basin_id=basin,
                                      G_same=same[0, bi], G_new=other[0, bi], D=loss[0, bi]))

    reference = pd.read_csv(P1 / "P1_endpoints.csv", float_precision="round_trip")
    saved = load(P1 / "P1_contrast_draws.npz")
    rows, group_rows, draws, aggregate_checks = [], [], {}, []
    for scheme in SCHEMES:
        hi, ti = indices[scheme]
        if scheme == "unmatched":
            hw, tw = bw["heldout"], bw["training"]
        elif scheme == "within_HUC2":
            hw = tw = bw["heldout"][:, hi]
        else:
            hw = tw = pair_weights
        for model in ["M1", "EA"]:
            for endpoint in ["D", "G_same"]:
                out = {}
                for role, selected, weights in [("heldout", hi, hw), ("training", ti, tw)]:
                    values = cached[(model, role)][endpoint][:, selected]
                    assert values.shape == weights.shape and (weights.sum(1) > 0).all()
                    out[role] = (values * weights).sum(1) / weights.sum(1)
                out["contrast"] = out["heldout"] - out["training"]
                old_endpoint = "same" if endpoint == "G_same" else "D"
                for role, values in out.items():
                    lo, high = np.quantile(values[1:], [.025, .975])
                    scenario = {"heldout": "unseen107", "training": "trained370", "contrast": "unseen_minus_trained"}[role]
                    analysis, variant = ("scale_influence", "mean") if scheme == "unmatched" else ("matching", scheme)
                    ref = reference[(reference.analysis == analysis) & (reference.variant == variant) & (reference.scenario == scenario)
                                    & (reference.comparison == model + "-B0") & (reference.scale == "NSE_star") & (reference.endpoint == old_endpoint)]
                    assert len(ref) == 1
                    np.testing.assert_allclose([values[0], lo, high], ref[["estimate", "low95", "high95"]].iloc[0], atol=1e-11, rtol=1e-12)
                    row = dict(comparison=model + "/B0", matching=scheme, endpoint=endpoint, role=role,
                               scale="NSE_star", n_heldout=len(hi), n_training=len(ti), estimate=values[0], low95=lo, high95=high)
                    (rows if role == "contrast" else group_rows).append(row)
                    draws[f"{scheme}__{model}__{endpoint}__{role}"] = values[1:]
                saved_key = f"{model}_NSE_star_mean_{old_endpoint}" if scheme == "unmatched" else f"match_{scheme}_{model}_NSE_star_{old_endpoint}"
                np.testing.assert_allclose(out["contrast"], saved[saved_key], atol=1e-11, rtol=1e-12)
                aggregate_checks.append(dict(matching=scheme, comparison=model + "/B0", endpoint=endpoint,
                                             estimates_and_intervals="PASS", saved_bootstrap_draws=2000))

    attributes.rename_axis("basin_id").reset_index().to_csv(DATA / "catchment_attributes.csv", index=False)
    pd.DataFrame(balances).to_csv(DATA / "covariate_balance.csv", index=False)
    pd.DataFrame(distributions).to_csv(DATA / "score_ecdfs.csv", index=False)
    pairs.to_csv(DATA / "matching_pairs.csv", index=False)
    pd.DataFrame(per_basin).to_csv(DATA / "per_catchment_outcomes.csv", index=False)
    pd.DataFrame(rows).to_csv(DATA / "contrast_endpoints.csv", index=False)
    pd.DataFrame(group_rows).to_csv(DATA / "group_endpoints.csv", index=False)
    np.savez_compressed(DATA / "bootstrap_draws.npz", **draws)
    np.savez_compressed(DATA / "bootstrap_weights.npz", seed_weights=sw[1:].astype(int), region_weights=gw[1:].astype(int),
                        across_region_pair_weights=pair_weights[1:].astype(int), seeds=h["seeds"], regions=groups)
    source_paths = [SCORES, META, ATTR_FEATHER, ATTR_CSV, *ARRAY_PATHS.values(), P1 / "matching_pairs.csv",
                    P1 / "matching_balance.csv", P1 / "matching_coverage.csv", P1 / "P1_per_basin.csv",
                    P1 / "P1_endpoints.csv", P1 / "P1_contrast_draws.npz"]
    report = dict(status="PASS", no_synthetic_data=True, new_training=False, n_heldout=107, n_training=370,
                  n_seeds_per_architecture=10, selection_period="WY2000-2003", evaluation_period="WY2004-2008",
                  matched_samples=dict(across_HUC2=101, within_HUC2=77), matching_covariate="f(mean_seed NSE_B0_early), f(x)=x/(2-x)",
                  matching_caliper=caliper, matching_checks=mapping_checks,
                  smd_definition="(heldout mean - training mean) / sqrt((full-heldout variance + full-training variance)/2), sample variances with ddof=1",
                  area_definition="Natural log of CAMELS area_geospa_fabric (km^2)",
                  other_covariates="CAMELS aridity, frac_snow, baseflow_index, p_mean, elev_mean; descriptive balance checks, not additional matching variables",
                  matching_scores_not_mean_transformed_scores=True, outcome_scale="NSE* transformed per instance before averaging",
                  verified_balance_cells=21, ecdf_curves=6, ecdf_observations=len(distributions),
                  interval_estimates_verified=36, original_contrast_bootstrap_values_verified=24000,
                  n_boot=2000, rng_seed=RNG_SEED, bootstrap_full_and_within="synchronized seed/HUC2 region bootstrap",
                  bootstrap_across="synchronized seed and matched-pair bootstrap", matches_fixed_in_bootstrap=True,
                  validation=aggregate_checks, input_hashes={str(p.relative_to(ARTICLE)): sha(p) for p in source_paths},
                  data_hashes={p.name: sha(p) for p in sorted(DATA.iterdir()) if p.is_file()})
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(balances).pivot(index="attribute", columns="matching", values="SMD").to_string())
    print(pd.DataFrame(rows)[["comparison", "matching", "endpoint", "estimate", "low95", "high95"]].to_string(index=False))
    print("PASS: all 178 original matched pairs, 21 balance cells, 36 endpoints and 24,000 saved contrast draws verified.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
