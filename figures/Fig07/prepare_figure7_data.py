"""Rebuild the four panels from the local fits and the published resampling schemes."""
import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODELS = ("M1", "EA")
LABELS = {
    "main": "Seed + HUC2 bootstrap", "seed_only": "Seed-only bootstrap",
    "median": "Catchment median", "omit3": "Omit 3 largest losses",
    "omit7": "Omit 7 largest losses", "omitlib": "Omit 7 largest library gains",
    "prespec": "WY1990–1999 selection", "windows": "Seven four-year windows",
    "perm": "Permuted B0 seed pairing", "k5": "Five source instances",
    "train_mean": "Training: mean", "train_median": "Training: median",
    "ext_nse": "External: NSE training loss", "ext_mse": "External: MSE training loss",
}
SHARE_ROWS = ("seed_only", "median", "omit3", "omit7", "omitlib", "prespec",
              "windows", "k5", "train_mean", "train_median")


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def nse(scores, period, model):
    return scores[period][list(scores["models"]).index(model), :, :,
                          list(scores["metrics"]).index("nse")]


def catchment_loss(choice_delta, late_delta):
    choice = choice_delta > 0
    n_seeds = choice.shape[0]
    retained = (choice * late_delta).mean(axis=0)
    independent = (choice * (late_delta.sum(axis=0) - late_delta)).mean(axis=0) / (n_seeds - 1)
    return retained - independent


def five_source_loss(early, late):
    indices = np.arange(len(early))
    catchment_losses = []
    for split in combinations(indices, 5):
        source = list(split)
        evaluation = np.setdiff1d(indices, source)
        choice = early[source].mean(axis=0) > 0
        catchment_losses.append(choice * (late[source].mean(axis=0) - late[evaluation].mean(axis=0)))
    assert len(catchment_losses) == 252
    return np.mean(catchment_losses)


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=3e-13)


def prepare(article):
    data = HERE / "data"
    paths = set()

    def source(rel):
        path = article / rel
        paths.add(path)
        return path

    def arrays(rel):
        with np.load(source(rel), allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    sys.path.insert(0, str(article))
    from analysis.selection_transfer import weighted

    held = arrays("outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz")
    train = arrays("outputs/weakness_resolution_20261102/local_internal/metric_arrays.npz")
    cover = read_csv(data / "forest_endpoints.csv").set_index(["comparison", "label"])
    matches = read_csv(data / "matching_contrasts.csv")
    cases = read_csv(source("deliverables/data_request_20261103/03_basin_metadata_531.csv"),
                     dtype={"basin_id": str}).set_index("basin_id")
    periods = read_csv(source("deliverables/data_request_20261103/01_scores_long.csv"),
                       usecols=["scenario", "period", "model", "seed", "basin_id", "NSE"],
                       dtype={"basin_id": str})
    window_table = read_csv(source("figures/Fig04/data/window_losses.csv"))
    pairs = read_csv(source("outputs/selection_transfer_P1_20261102/matching_pairs.csv"),
                     dtype={"unseen_basin": str, "trained_basin": str})
    regions = held["regions"]
    assert (len(held["basins"]), len(train["basins"]), len(held["seeds"])) == (107, 370, 10)
    meta = cases.loc[held["basins"]]
    omit_library = set(meta.EA_library_gain.nlargest(7).index)
    assert len(omit_library) == 7

    local_draws = np.load(data / "bootstrap_draws.npz", allow_pickle=False)
    seed_weights = np.load(data / "bootstrap_weights.npz", allow_pickle=False)
    check(seed_weights["seeds"], held["seeds"])
    sw = seed_weights["seed_weights"]
    assert sw.shape == (2000, 10) and np.all((sw > 0).sum(axis=1) >= 2)
    training_draws = np.load(source("outputs/multimetric_transfer_20261102/bootstrap_draws.npz"), allow_pickle=False)
    budget = np.load(source("outputs/multiseed_budget_curve_20261104/bootstrap_draws.npz"), allow_pickle=False)
    match_draws = np.load(source("outputs/selection_transfer_P1_20261102/P1_contrast_draws.npz"), allow_pickle=False)
    endpoints = []
    dist = {}
    match_checks = []

    def record(model, key, point, draws):
        draws = np.asarray(draws, dtype=float)
        assert draws.ndim == 1 and np.isfinite(draws).all()
        reference = cover.loc[(model + "/B0", LABELS[key])]
        limits = (draws.min(), draws.max()) if key == "windows" else np.quantile(draws, [.025, .975])
        check([point, *limits], reference[["estimate", "low95", "high95"]].to_numpy(dtype=float))
        endpoints.append(dict(model=model, key=key, estimate=float(point),
                              low=float(limits[0]), high=float(limits[1]), n=len(draws),
                              interval=reference.interval_kind))
        dist[f"{model}__{key}"] = draws

    for model in MODELS:
        early = nse(held, "earlier", model) - nse(held, "earlier", "B0")
        late = nse(held, "later", model) - nse(held, "later", "B0")
        db = catchment_loss(early, late)
        record(model, "main", db.mean(), local_draws[f"{model}__NSE__mean"])
        record(model, "median", np.median(db), local_draws[f"{model}__NSE__basin_median"])
        for count in (3, 7):
            record(model, f"omit{count}", np.sort(db)[:-count].mean(),
                   local_draws[f"{model}__NSE__drop_largest_D_{count}"])
        record(model, "omitlib", db[~meta.index.isin(omit_library)].mean(), local_draws[f"{model}__NSE__drop_C_7"])
        seed_only = [weighted(early, late, weights, np.ones(len(db)))[2] for weights in sw]
        record(model, "seed_only", db.mean(), seed_only)

        folder = ("retraining_confirmation_analysis_20261031" if model == "M1"
                  else "ea_selection_analysis_20261101")
        primary = arrays(f"outputs/{folder}/primary_arrays.npz")
        assert np.array_equal(primary["basins"], held["basins"]) and np.array_equal(primary["regions"], regions)
        check(primary["later_delta_nse"], late)
        ep = primary["early_delta_nse"]
        groups, membership = np.unique(regions, return_inverse=True)
        rng = np.random.default_rng(20261029)
        prescribed = []
        while len(prescribed) < 5000:
            seeds = rng.multinomial(len(early), np.full(len(early), 1 / len(early)))
            if np.count_nonzero(seeds) < 2:
                continue
            cluster = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
            prescribed.append(weighted(ep, primary["later_delta_nse"], seeds, cluster[membership])[2])
        record(model, "prespec", catchment_loss(ep, late).mean(), prescribed)

        windows = []
        for start in range(1990, 1997):
            window = f"WY{start}-{start+3}"
            subset = periods[(periods.scenario == "heldout107") & (periods.period == window)]
            def scores(which):
                return (subset.loc[subset.model == which].pivot(index="seed", columns="basin_id", values="NSE")
                        .loc[held["seeds"], held["basins"]].to_numpy())
            d = catchment_loss(scores(model) - scores("B0"), late).mean()
            published = window_table[(window_table.selection_window == window) &
                                     (window_table.comparison == model + "/B0")]
            assert len(published) == 1
            check(d, published.D.iloc[0])
            windows.append(d)
        record(model, "windows", np.mean(windows), windows)

        draw_column = list(budget["endpoint_names"]).index("D")
        record(model, "k5", five_source_loss(early, late), budget[f"{model}__m5__NSE"][:, draw_column])
        early_train = nse(train, "earlier", model) - nse(train, "earlier", "B0")
        late_train = nse(train, "later", model) - nse(train, "later", "B0")
        train_loss = catchment_loss(early_train, late_train)
        for key, stat, point in (("train_mean", "mean", train_loss.mean()),
                                 ("train_median", "basin_median", np.median(train_loss))):
            archived = training_draws[f"trained370__{model}__fixed_early_NSE__NSE__{stat}__D"]
            check(point, archived[0])
            record(model, key, point, archived[1:])

        def bounded_loss(scores, choice):
            candidate, base = nse(scores, "later", model), nse(scores, "later", "B0")
            return catchment_loss(choice, candidate / (2 - candidate) - base / (2 - base))
        held_star, train_star = bounded_loss(held, early), bounded_loss(train, early_train)
        held_set = pd.Series(held_star, index=held["basins"])
        train_set = pd.Series(train_star, index=train["basins"])
        for scheme in ("unmatched", "across_HUC2", "within_HUC2"):
            if scheme == "unmatched":
                point = held_set.mean() - train_set.mean()
                draw_key = f"{model}_NSE_star_mean_D"
                counts = (len(held_set), len(train_set))
            else:
                pair = pairs[pairs.matching == scheme]
                point = (held_set.loc[pair.unseen_basin].to_numpy()
                         - train_set.loc[pair.trained_basin].to_numpy()).mean()
                draw_key = f"match_{scheme}_{model}_NSE_star_D"
                counts = (len(pair), len(pair))
            values = match_draws[draw_key]
            check(values[0], point)
            limits = np.quantile(values[1:], [.025, .975])
            ref = matches[(matches.comparison == model + "/B0") & (matches.matching == scheme)
                          & (matches.endpoint == "D")].iloc[0]
            check([point, *limits, *counts], [ref.estimate, ref.low95, ref.high95, ref.n_heldout, ref.n_training])
            match_checks.append(dict(model=model, scheme=scheme, point=float(point), low=limits[0],
                                     high=limits[1], n_heldout=counts[0], n_training=counts[1]))

    # Permute the same baseline seed identities in both periods, as in Supplementary Section S3.
    rng = np.random.default_rng(20261102)
    ea_early, ea_late = nse(held, "earlier", "EA"), nse(held, "later", "EA")
    b0_early, b0_late = nse(held, "earlier", "B0"), nse(held, "later", "B0")
    permutations = []
    for _ in range(2000):
        order = rng.permutation(len(ea_early))
        permutations.append(catchment_loss(ea_early - b0_early[order], ea_late - b0_late[order]).mean())
    record("EA", "perm", catchment_loss(ea_early-b0_early, ea_late-b0_late).mean(), permutations)

    for objective in ("NSE", "MSE"):
        folder = f"outputs/weakness_resolution_20261102/external_{objective}"
        scores = arrays(f"{folder}/metric_arrays.npz")
        assert len(scores["seeds"]) == 8 and len(scores["basins"]) == 531
        db = catchment_loss(nse(scores, "earlier", "EA") - nse(scores, "earlier", "B0"),
                            nse(scores, "later", "EA") - nse(scores, "later", "B0"))
        samples = read_csv(source(f"{folder}/selection_bootstrap_EA.csv"))
        record("EA", "ext_" + objective.lower(), db.mean(), samples.loc[samples.scheme == "seed_HUC2", "D"])

    for archive in (local_draws, seed_weights, training_draws, budget, match_draws):
        archive.close()
    panel = pd.DataFrame(endpoints)
    assert len(panel) == len(cover) == 25
    ratios = []
    for model in MODELS:
        values = panel.loc[panel.model == model].set_index("key")
        base = values.loc["main", "estimate"]
        for key in SHARE_ROWS:
            row = values.loc[key]
            ratios.append(dict(model=model, check=key, D=row.estimate, main_D=base,
                               percent=100 * row.estimate / base, bounds_include_zero=row.low <= 0 <= row.high,
                               interval_kind=row.interval))
    shares = pd.DataFrame(ratios)
    assert set(shares.loc[shares.bounds_include_zero, "check"]) == {"train_mean"}
    for model, expected in (("M1", [28, 91, 91]), ("EA", [37, 117, 106])):
        s = shares[shares.model == model].set_index("check")
        assert s.loc[list(SHARE_ROWS[:-2]), "percent"].idxmin() == "median"
        assert s.loc[["median", "prespec", "windows"], "percent"].round().tolist() == expected
    np.savez_compressed(data / "forest_distributions.npz", **dist)
    shares.to_csv(data / "retained_shares.csv", index=False, float_format="%.17g")
    sources = {p.relative_to(article).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(paths) if p.is_file()}
    local_files = ("forest_endpoints.csv", "matching_contrasts.csv", "bootstrap_draws.npz", "bootstrap_weights.npz")
    for f in local_files:
        sources[f"figures/Fig07/data/{f}"] = hashlib.sha256((data/f).read_bytes()).hexdigest()
    report = dict(status="PASS", upper_panel=endpoints, matching=match_checks, ratios=ratios,
                  input_sha256=sources, window_statistic="Seven individual estimates; min–max in a,b and mean in d",
                  five_source="20,000 fixed-library HUC2 draws; 252 ordered allocations",
                  seed_pairing="2,000 B0-label permutations; diamond is the original pairing")
    (HERE / "enhanced_data_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=lambda x: x.item()), encoding="utf-8")
    print(f"Verified {len(panel)} forest rows, {len(match_checks)} matched contrasts and {len(shares)} retained shares.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article", type=Path, default=HERE.parents[1])
    prepare(parser.parse_args().article.resolve())
