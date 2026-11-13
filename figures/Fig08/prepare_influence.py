"""Catchment influence estimates in Figure 8 with the original 2,000-draw bootstrap."""
from hashlib import sha256
from pathlib import Path
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARTICLE = HERE.parents[1]
DATA = HERE / "data"
SCORES = ARTICLE / "deliverables/data_request_20261103/01_scores_long.csv"
META = ARTICLE / "deliverables/data_request_20261103/03_basin_metadata_531.csv"
BASE = ARTICLE / "outputs/training_domain_followup_20261102"
MULTI = ARTICLE / "outputs/multimetric_transfer_20261102"
P1 = ARTICLE / "outputs/selection_transfer_P1_20261102"
ARRAY = BASE / "unseen_aligned/metric_arrays.npz"
TRAIN_ARRAY = ARTICLE / "outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz"
METRICS = ["NSE", "NSE_star", "KGE", "abs_FHV", "nMAE_high", "nMAE_low"]
VARIANTS = ["mean", "basin_median", "drop_largest_D_3", "drop_largest_D_7", "drop_C_7"]
LABELS = ["Mean", "Catchment median", "Drop top-3 D", "Drop top-7 D", "Drop top-7 library gain"]
N_BOOT, RNG_SEED = 2000, 20261102


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def weighted_median(values, weights):
    order = np.argsort(values, axis=1)
    v = np.take_along_axis(values, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    cumulative = w.cumsum(1)
    halfway = w.sum(1) / 2
    lo = (cumulative < halfway[:, None]).sum(1)
    hi = np.minimum((cumulative <= halfway[:, None]).sum(1), values.shape[1] - 1)
    return (v[np.arange(len(v)), lo] + v[np.arange(len(v)), hi]) / 2


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    a, tr = load(ARRAY), load(TRAIN_ARRAY)
    basins, seeds, regions = a["basins"], a["seeds"], a["regions"]
    assert len(basins) == 107 and len(seeds) == 10
    np.testing.assert_array_equal(seeds, tr["seeds"])
    assert list(a["models"]) == ["B0", "M1", "EA"]
    raw = pd.read_csv(SCORES, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip")
    raw = raw[(raw.scenario == "heldout107") & raw.period.isin(["WY2000-2003", "WY2004-2008"])]
    assert len(raw) == 6420 and not raw.duplicated(["model", "seed", "basin_id", "period"]).any()
    meta = pd.read_csv(META, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip").set_index("basin_id").loc[basins]
    np.testing.assert_array_equal(meta.huc2, regions)
    normalizers = pd.read_csv(MULTI / "observation_normalizers.csv", dtype={"basin_id": str}, float_precision="round_trip")
    norm = normalizers[(normalizers.scenario == "unseen107") & (normalizers.era == "later")].set_index("basin_id").loc[basins, "mean_q_mm_day"].to_numpy()

    def matrix(model, period, metric):
        t = raw[(raw.model == model) & (raw.period == period)]
        out = t.pivot(index="seed", columns="basin_id", values=metric).loc[seeds, basins].to_numpy()
        assert out.shape == (10, 107) and np.isfinite(out).all()
        return out

    utilities = {}
    for mi, model in enumerate(a["models"]):
        np.testing.assert_allclose(matrix(model, "WY2000-2003", "NSE"), a["earlier"][mi, ..., 0], atol=1e-11, rtol=1e-12)
        v = a["later"][mi]
        expected = dict(NSE=v[..., 0], NSE_star=v[..., 0] / (2 - v[..., 0]), KGE=v[..., 1],
                        abs_FHV=np.abs(v[..., 5]), nMAE_high=v[..., 8] / norm, nMAE_low=v[..., 7] / norm)
        for metric in METRICS:
            arr = matrix(model, "WY2004-2008", "abs_FHV_pct" if metric == "abs_FHV" else metric)
            np.testing.assert_allclose(arr, expected[metric], atol=1e-11, rtol=1e-12)
            utilities[(model, metric)] = -arr if metric in ["abs_FHV", "nMAE_high", "nMAE_low"] else arr

    library = pd.read_csv(BASE / "unseen_aligned_basin_gains.csv", dtype={"basin_id": str}, float_precision="round_trip")
    library = library[(library.comparison == "EA-B0") & (library.policy == "library_local")].set_index("basin_id").loc[basins]
    np.testing.assert_allclose(library.gain, library.selected_nse - library.mixed_nse, atol=1e-12)
    np.testing.assert_allclose(library.gain, meta.EA_library_gain, atol=1e-12)
    lib_order = np.argsort(-library.gain.to_numpy(), kind="stable")
    library_top7 = basins[lib_order[:7]]

    # Reproduce the synchronized seed/HUC2 draws used in P1 and the
    # multi-metric analysis, including the combined 18-region design.
    rng = np.random.default_rng(RNG_SEED)
    sw = np.ones((N_BOOT + 1, 10))
    sw[1:] = rng.multinomial(10, np.full(10, .1), size=N_BOOT)
    for i in np.flatnonzero(np.count_nonzero(sw, axis=1) < 2):
        while np.count_nonzero(sw[i]) < 2:
            sw[i] = rng.multinomial(10, np.full(10, .1))
    groups = np.unique(np.concatenate([regions, tr["regions"]]))
    gw = np.ones((N_BOOT + 1, len(groups)))
    gw[1:] = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=N_BOOT)
    bw = gw[:, [list(groups).index(r) for r in regions]]
    seed_denominator = sw.sum(1) ** 2 - (sw ** 2).sum(1)
    assert np.all(seed_denominator > 0) and np.all(bw.sum(1) > 0)

    p1 = pd.read_csv(P1 / "P1_endpoints.csv", float_precision="round_trip")
    p1_deleted = pd.read_csv(P1 / "P1_deleted_basins.csv", dtype={"basin_id": str})
    multi = pd.read_csv(MULTI / "transfer_endpoints.csv", float_precision="round_trip")
    multi_basin = pd.read_csv(MULTI / "transfer_per_basin.csv", dtype={"basin_id": str}, float_precision="round_trip")
    saved_draws = load(MULTI / "bootstrap_draws.npz")
    rows, per_basin, deleted, draws, checks = [], [], [], {}, []
    draw_checks, p1_checks, endpoint_checks, witness_checks = 0, 0, 0, 0
    for model in ["M1", "EA"]:
        choose = (matrix(model, "WY2000-2003", "NSE") > matrix("B0", "WY2000-2003", "NSE")).astype(float)
        for metric in METRICS:
            q = utilities[(model, metric)] - utilities[("B0", metric)]
            same = sw @ (choose * q) / sw.sum(1, keepdims=True)
            counts = sw @ choose
            other = (counts * (sw @ q) - (sw ** 2) @ (choose * q)) / seed_denominator[:, None]
            # The exact-zero identity applies to the equally weighted point
            # estimate. Bootstrap duplicate weights change the off-diagonal
            # evaluation marginals, so retain the original resampling statistic.
            constant_point = (choose == choose[0]).all(axis=0)
            other[0, constant_point] = same[0, constant_point]
            d = same - other
            covariance = ((choose - choose.mean(0)) * (q - q.mean(0))).sum(0) / 9
            np.testing.assert_allclose(d[0], covariance, atol=1e-12, rtol=1e-12)
            # Independent explicit resampled pairs exclude equal ORIGINAL seed IDs.
            for wi in [0, 1, 67, 511, 2000]:
                seed_ids = np.repeat(np.arange(10), sw[wi].astype(int))
                s_explicit = (choose[seed_ids] * q[seed_ids]).mean(0)
                n_explicit = np.mean([choose[s] * q[t] for s in seed_ids for t in seed_ids if s != t], axis=0)
                np.testing.assert_allclose(d[wi], s_explicit - n_explicit, atol=1e-12, rtol=1e-12)
                witness_checks += 1
            ref_b = multi_basin[(multi_basin.scenario == "unseen107") & (multi_basin.comparison == model + "-B0")
                                & (multi_basin.selection_rule == "fixed_early_NSE") & (multi_basin.metric == metric)]
            ref_b = ref_b.set_index("basin_id").loc[basins]
            np.testing.assert_allclose(d[0], ref_b.D, atol=1e-11, rtol=1e-12)
            for bi, basin in enumerate(basins):
                per_basin.append(dict(comparison=model + "/B0", metric=metric, basin_id=basin, huc2=regions[bi],
                                      G_same=same[0, bi], G_new=other[0, bi], D=d[0, bi]))

            d_order = np.argsort(-d[0], kind="stable")
            mean_ref = d[0].mean()
            for row_index, (variant, label) in enumerate(zip(VARIANTS, LABELS)):
                keep = np.ones(107, dtype=bool)
                if variant.startswith("drop_largest_D"):
                    keep[d_order[:int(variant.split("_")[-1])]] = False
                elif variant == "drop_C_7":
                    keep[lib_order[:7]] = False
                w, vals = bw[:, keep], d[:, keep]
                assert np.all(w.sum(1) > 0)
                out = weighted_median(vals, w) if variant == "basin_median" else (vals * w).sum(1) / w.sum(1)
                point = float(np.median(d[0, keep]) if variant == "basin_median" else d[0, keep].mean())
                np.testing.assert_allclose(out[0], point, atol=1e-12)
                for wi in [0, 1, 67, 511, 2000]:
                    expanded = np.repeat(vals[wi], w[wi].astype(int))
                    expected = np.median(expanded) if variant == "basin_median" else expanded.mean()
                    np.testing.assert_allclose(out[wi], expected, atol=1e-12, rtol=1e-12)
                lo, hi = np.quantile(out[1:], [.025, .975])
                rows.append(dict(comparison=model + "/B0", metric=metric, variant=variant, row_index=row_index,
                                 row_label=label, n_catchments=int(keep.sum()), estimate=point, low95=lo, high95=hi,
                                 ratio_to_mean=point / mean_ref, ci_excludes_zero=bool(lo > 0 or hi < 0),
                                 ci_above_zero=bool(lo > 0), ci_includes_zero=bool(lo <= 0 <= hi), n_boot=N_BOOT))
                draws[f"{model}__{metric}__{variant}"] = out[1:]
                for bi in np.flatnonzero(~keep):
                    deleted.append(dict(comparison=model + "/B0", metric=metric, variant=variant, basin_id=basins[bi],
                                        ranking_value=library.gain.iloc[bi] if variant == "drop_C_7" else d[0, bi],
                                        criterion="EA/B0 late library gain in NSE" if variant == "drop_C_7" else "full-sample D_b for this comparison and metric"))
                if variant in ["mean", "basin_median"]:
                    refs = multi[(multi.scenario == "unseen107") & (multi.comparison == model + "-B0")
                                 & (multi.metric == metric) & (multi.selection_rule == "fixed_early_NSE")
                                 & (multi.aggregation == variant) & (multi.endpoint == "D")]
                    assert len(refs) == 1
                    np.testing.assert_allclose([point, lo, hi], refs[["estimate", "low95", "high95"]].iloc[0], atol=1e-11, rtol=1e-12)
                    key = f"unseen107__{model}__fixed_early_NSE__{metric}__{variant}__D"
                    np.testing.assert_allclose(out, saved_draws[key], atol=1e-11, rtol=1e-12)
                    endpoint_checks += 1
                    draw_checks += N_BOOT
                if metric in ["NSE", "NSE_star"]:
                    refs = p1[(p1.analysis == "scale_influence") & (p1.scenario == "unseen107")
                              & (p1.comparison == model + "-B0") & (p1.scale == metric)
                              & (p1.variant == variant) & (p1.endpoint == "D")]
                    assert len(refs) == 1 and int(refs.n_basins.iloc[0]) == keep.sum()
                    np.testing.assert_allclose([point, lo, hi], refs[["estimate", "low95", "high95"]].iloc[0], atol=1e-11, rtol=1e-12)
                    dd = p1_deleted[(p1_deleted.scenario == "unseen107") & (p1_deleted.comparison == model + "-B0")
                                    & (p1_deleted.scale == metric) & (p1_deleted.variant == variant)]
                    assert set(dd.basin_id) == set(basins[~keep])
                    p1_checks += 1
            checks.append(dict(comparison=model + "/B0", metric=metric, per_catchment_estimates=107,
                               bootstrap_draws=N_BOOT, baseline_results="PASS", covariance_and_explicit_pairs="PASS"))
            print(f"VERIFIED {model}/B0 {metric}", flush=True)

    table = pd.DataFrame(rows)
    assert len(table) == 60 and len(per_basin) == 1284
    table.to_csv(DATA / "robustness_endpoints.csv", index=False)
    pd.DataFrame(per_basin).to_csv(DATA / "per_catchment_losses.csv", index=False)
    pd.DataFrame(deleted).to_csv(DATA / "deleted_catchments.csv", index=False)
    library.reset_index().iloc[lib_order][["basin_id", "selected_nse", "mixed_nse", "gain"]].to_csv(DATA / "EA_library_gain_ranking.csv", index=False)
    np.savez_compressed(DATA / "bootstrap_draws.npz", **draws)
    np.savez_compressed(DATA / "bootstrap_weights.npz", seed_weights=sw[1:].astype(int),
                        region_weights=gw[1:].astype(int), seeds=seeds, regions=groups, basin_ids=basins, basin_regions=regions)
    source_paths = [SCORES, META, ARRAY, TRAIN_ARRAY, BASE / "unseen_aligned_basin_gains.csv",
                    MULTI / "observation_normalizers.csv", MULTI / "transfer_endpoints.csv",
                    MULTI / "transfer_per_basin.csv", MULTI / "bootstrap_draws.npz",
                    P1 / "P1_endpoints.csv", P1 / "P1_deleted_basins.csv"]
    report = dict(status="PASS", no_synthetic_data=True, new_training=False, n_catchments=107,
                  n_seeds_per_architecture=10, n_metrics=6, n_variants=5, n_cells=60,
                  selection_period="WY2000-2003", evaluation_period="WY2004-2008", selection_metric="NSE fixed for all columns",
                  n_boot=N_BOOT, rng_seed=RNG_SEED, bootstrap="Pointwise percentile seed/HUC2 bootstrap, synchronized with existing main analyses",
                  spatial_sampling_regions=groups.tolist(), heldout_regions=np.unique(regions).tolist(),
                  distinct_seed_pair_policy="Exclude matching original seed identities, including repeated bootstrap copies",
                  deletion_policy="Fix full-sample deletion sets within every bootstrap replicate",
                  shared_EA_library_top7=library_top7.tolist(), existing_P1_cells_reproduced=p1_checks,
                  existing_multimetric_cells_reproduced=endpoint_checks, existing_bootstrap_values_checked=draw_checks,
                  explicit_seed_pair_witnesses=witness_checks, all_deletion_sets_match_P1=True,
                  numerical_zero_policy="For the equal-weight point estimate only, set D_b = 0 when all ten seeds make the same choice; bootstrap statistics retain the existing original-seed exclusion rule",
                  validation=checks, cells_with_ci_above_zero=int(table.ci_above_zero.sum()),
                  cells_with_ci_including_zero=int(table.ci_includes_zero.sum()),
                  input_hashes={str(p.relative_to(ARTICLE)): sha(p) for p in source_paths},
                  data_hashes={p.name: sha(p) for p in sorted(DATA.iterdir()) if p.is_file()})
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table[["comparison", "metric", "variant", "estimate", "low95", "high95", "ci_above_zero"]].to_string(index=False))
    print(f"PASS: {p1_checks} existing P1 cells, {endpoint_checks} multi-metric cells, {draw_checks} saved bootstrap values.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
