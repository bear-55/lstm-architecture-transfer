"""Reconstruct Figure 3 trajectories and summaries from real experiment scores."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
DISPLAY_LIMIT = .22
SOURCES = {
    "local_scores": ROOT / "deliverables/data_request_20261103/01_scores_long.csv",
    "external_scores": ROOT / "outputs/weakness_resolution_20261102/external_NSE/individual_metrics.csv",
    "external_arrays": ROOT / "outputs/weakness_resolution_20261102/external_NSE/metric_arrays.npz",
    "external_eligibility": ROOT / "outputs/weakness_resolution_20261102/external/eligibility.json",
    "external_runs": ROOT / "outputs/weakness_resolution_20261102/external_NSE/source_runs.json",
}
PANELS = [("a", "M1", "Held-out", "heldout107", 107),
          ("b", "M1", "Training", "trained370", 370),
          ("d", "EA", "Held-out", "heldout107", 107),
          ("e", "EA", "Training", "trained370", 370),
          ("f", "EA", "External", "external_NSE", 531)]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    local = pd.read_csv(SOURCES["local_scores"], dtype={"basin_id": str})
    external = pd.read_csv(SOURCES["external_scores"], dtype={"basin_id": str})
    eligibility = json.loads(SOURCES["external_eligibility"].read_text(encoding="utf-8"))
    assert eligibility["selection_period"] == ["1989-10-01", "1994-09-30"]
    assert eligibility["evaluation_period"] == ["1994-10-01", "1999-09-30"]
    with np.load(SOURCES["external_arrays"], allow_pickle=False) as z:
        source_arrays = {k: z[k] for k in z.files}
    raw_rows, seed_rows, quintile_rows, validation = [], [], [], []
    for panel, candidate, group, scenario, n_basin in PANELS:
        matrices = {}
        is_external = group == "External"
        periods = ("WY1990–1994", "WY1995–1999") if is_external else ("WY2000–2003", "WY2004–2008")
        for model in ["B0", candidate]:
            for stage, era, period in [("early", "earlier", "WY2000-2003"), ("late", "later", "WY2004-2008")]:
                if is_external:
                    sub = external[(external.model == model) & (external.era == era)].copy()
                    sub["NSE_star"] = sub.nse / (2 - sub.nse)
                    raw = sub.pivot(index="seed", columns="basin_id", values="nse").sort_index().sort_index(axis=1)
                    model_i = source_arrays["models"].tolist().index(model)
                    np.testing.assert_array_equal(raw.index.to_numpy(), source_arrays["seeds"])
                    np.testing.assert_array_equal(raw.columns.to_numpy(), source_arrays["basins"])
                    np.testing.assert_allclose(raw.to_numpy(), source_arrays[era][model_i, :, :, 0], atol=1e-12, rtol=1e-14)
                else:
                    sub = local[(local.scenario == scenario) & (local.model == model) & (local.period == period)].copy()
                    np.testing.assert_allclose(sub.NSE_star, sub.NSE / (2 - sub.NSE), atol=1e-14, rtol=0)
                assert not sub.duplicated(["seed", "basin_id"]).any()
                matrices[model, stage] = sub.pivot(index="seed", columns="basin_id", values="NSE_star").sort_index().sort_index(axis=1)
        ref = matrices["B0", "early"]
        for table in matrices.values():
            assert table.index.equals(ref.index) and table.columns.equals(ref.columns)
            assert np.isfinite(table.to_numpy()).all()
        n_seed = len(ref)
        assert ref.shape == (8 if is_external else 10, n_basin)
        seeds, basins = ref.index.tolist(), ref.columns.tolist()
        x = (matrices[candidate, "early"] - matrices["B0", "early"]).to_numpy()
        y = (matrices[candidate, "late"] - matrices["B0", "late"]).to_numpy()
        z = (y.sum(0)[None, :] - y) / (n_seed - 1)
        literal_z = np.stack([np.mean(np.delete(y, s, axis=0), axis=0) for s in range(n_seed)])
        np.testing.assert_allclose(z, literal_z, atol=1e-15, rtol=0)

        for si, seed in enumerate(seeds):
            edges = np.quantile(x[si], np.linspace(0, 1, 6), method="linear")
            bins = np.clip(np.searchsorted(edges, x[si], side="right") - 1, 0, 4)
            assert np.array_equal(np.unique(bins), np.arange(5))
            correlations = {}
            for name, values in [("retained", y[si]), ("retrained", z[si])]:
                rho = float(spearmanr(x[si], values).statistic)
                # Independent calculation: Pearson correlation of average ranks.
                rank_rho = float(np.corrcoef(pd.Series(x[si]).rank(), pd.Series(values).rank())[0, 1])
                np.testing.assert_allclose(rho, rank_rho, atol=1e-14, rtol=0)
                correlations[name] = rho
            seed_rows.append(dict(panel=panel, comparison=candidate + "/B0", group=group, source_seed=seed,
                                  n_catchments=n_basin, n_evaluation_seeds=n_seed - 1,
                                  rho_retained=correlations["retained"], rho_retrained=correlations["retrained"]))
            for bi, basin in enumerate(basins):
                raw_rows.append(dict(panel=panel, comparison=candidate + "/B0", group=group,
                                     source_seed=seed, basin_id=basin, historical_quintile=int(bins[bi]) + 1,
                                     historical=x[si, bi], late_retained=y[si, bi], late_retrained=z[si, bi]))
            for q in range(5):
                keep = bins == q
                for stage, values in [("historical", x[si]), ("late_retained", y[si]), ("late_retrained", z[si])]:
                    q25, med, q75 = np.percentile(values[keep], [25, 50, 75], method="linear")
                    quintile_rows.append(dict(panel=panel, comparison=candidate + "/B0", group=group,
                                              source_seed=seed, quintile=q + 1, stage=stage,
                                              n_catchments=int(keep.sum()), q25=q25, median=med, q75=q75))
        example = np.stack([x[0], y[0], z[0]], axis=1)
        validation.append(dict(panel=panel, comparison=candidate + "/B0", group=group,
                               n_seed=n_seed, n_catchments=n_basin, seeds=seeds,
                               historical_period=periods[0], late_period=periods[1],
                               example_seed=seeds[0], example_min=float(example.min()), example_max=float(example.max()),
                               example_curves_outside_display=int(np.any(np.abs(example) > DISPLAY_LIMIT, axis=1).sum()),
                               example_values_below_display=(example < -DISPLAY_LIMIT).sum(0).tolist(),
                               example_values_above_display=(example > DISPLAY_LIMIT).sum(0).tolist()))

    trajectories = pd.DataFrame(raw_rows)
    seed_stats = pd.DataFrame(seed_rows)
    seed_quintiles = pd.DataFrame(quintile_rows)
    qgroups = ["panel", "comparison", "group", "quintile", "stage"]
    quintiles = seed_quintiles.groupby(qgroups, sort=False).agg(
        mean_q25=("q25", "mean"), mean_median=("median", "mean"), mean_q75=("q75", "mean"),
        min_quintile_n=("n_catchments", "min"), max_quintile_n=("n_catchments", "max"),
        n_source_seeds=("source_seed", "nunique")).reset_index()
    correlation = seed_stats.groupby(["panel", "comparison", "group"], sort=False).agg(
        rho_retained=("rho_retained", "mean"), rho_retrained=("rho_retrained", "mean"),
        n_source_seeds=("source_seed", "nunique"), n_catchments=("n_catchments", "first")).reset_index()
    assert len(trajectories) == 13788 and len(seed_stats) == 48 and len(quintiles) == 75
    assert ((quintiles.mean_q25 <= quintiles.mean_median) & (quintiles.mean_median <= quintiles.mean_q75)).all()
    assert np.abs(quintiles["mean_median"].to_numpy()).max() < DISPLAY_LIMIT
    expected = {("M1/B0", "Held-out"): (.56, -.05), ("M1/B0", "Training"): (.32, .02),
                ("EA/B0", "Held-out"): (.57, .11), ("EA/B0", "Training"): (.30, .09)}
    for r in correlation.itertuples():
        if r.group != "External":
            assert (round(r.rho_retained, 2), round(r.rho_retrained, 2)) == expected[r.comparison, r.group]
    tables = {"trajectories.csv": trajectories, "correlations_by_seed.csv": seed_stats,
              "quintiles_by_seed.csv": seed_quintiles, "quintile_summary.csv": quintiles,
              "correlation_summary.csv": correlation}
    for filename, table in tables.items():
        table.to_csv(DATA / filename, index=False, float_format="%.17g")
    report = dict(status="PASS", panels=validation, correlation_summary=correlation.to_dict("records"),
                  trajectory_rows=len(trajectories), summary_rows=len(quintiles),
                  summary_definition="Within each source seed, group catchments by historical delta NSE* quintile. Compute each group's q25, median and q75 at each stage; average each statistic over source seeds.",
                  retrained_definition="Arithmetic mean of late individual delta NSE* over all other seeds (nine local, seven external); not a discharge ensemble score.",
                  correlation_definition="Mean of per-source-seed Spearman correlations between historical delta NSE* and each late branch, across catchments.",
                  external_source=eligibility, no_synthetic_data=True, display_ylim=[-DISPLAY_LIMIT, DISPLAY_LIMIT],
                  source_hashes={k: dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for k, p in SOURCES.items()},
                  data_hashes={name: sha(DATA / name) for name in tables})
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(correlation.to_string(index=False))
    print("All foreground medians fit the common [-0.22, 0.22] display; quartile bars are not plotted.")
    print("Verified trajectories:", len(trajectories))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
