"""Compute the balanced period-by-seed-by-catchment decomposition in Figure 3c."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
SOURCE = ROOT / "deliverables/data_request_20261103/01_scores_long.csv"
SEEDS = [1009, 1021, 1033, 1051, 1063, 1087, 1097, 1109, 1123, 1151]
PERIODS = ["WY2000-2003", "WY2004-2008"]
METRICS = ["NSE", "NSE_star", "KGE", "abs_FHV_pct", "nMAE_high", "nMAE_low"]
ERRORS = {"abs_FHV_pct", "nMAE_high", "nMAE_low"}
COMPONENTS = ["Catchment", "Seed", "Seed×Catchment", "Period×Catchment",
              "Period×Seed×Catchment", "Period terms"]
CASES = [
    ("a", "heldout107", "Held-out", 107,
     ROOT / "outputs/training_domain_followup_20261102/unseen_aligned"),
    ("b", "trained370", "Training", 370,
     ROOT / "outputs/weakness_resolution_20261102/local_internal"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(x):
    """Template's seven orthogonal sums of squares for x[period, seed, basin]."""
    mean = x.mean()
    p, s, b = x.mean((1, 2)), x.mean((0, 2)), x.mean((0, 1))
    ps, pb, sb = x.mean(2), x.mean(1), x.mean(0)
    np_, ns, nb = x.shape
    values = {
        "Period": ns * nb * np.sum((p - mean) ** 2),
        "Seed": np_ * nb * np.sum((s - mean) ** 2),
        "Catchment": np_ * ns * np.sum((b - mean) ** 2),
        "Period×Seed": nb * np.sum((ps - p[:, None] - s[None, :] + mean) ** 2),
        "Period×Catchment": ns * np.sum((pb - p[:, None] - b[None, :] + mean) ** 2),
        "Seed×Catchment": np_ * np.sum((sb - s[:, None] - b[None, :] + mean) ** 2),
    }
    residual = (x - ps[:, :, None] - pb[:, None, :] - sb[None, :, :]
                + p[:, None, None] + s[None, :, None] + b[None, None, :] - mean)
    values["Period×Seed×Catchment"] = np.sum(residual ** 2)
    total = np.sum((x - mean) ** 2)
    assert total > 0
    np.testing.assert_allclose(sum(values.values()), total, rtol=1e-12)

    # Independently reconstruct by subtracting all lower-order marginal
    # projections, then check all components are mutually orthogonal.
    effects = {0: x.mean(keepdims=True)}
    independent = {}
    for mask in sorted(range(1, 8), key=int.bit_count):
        axes = tuple(i for i in range(3) if not mask & (1 << i))
        effect = x.mean(axis=axes, keepdims=True) if axes else x.copy()
        for submask, lower in effects.items():
            if submask & mask == submask:
                effect = effect - lower
        effects[mask] = effect
        label = "×".join(["Period", "Seed", "Catchment"][i] for i in range(3) if mask & (1 << i))
        independent[label] = np.sum(np.broadcast_to(effect, x.shape) ** 2)
    for key in values:
        np.testing.assert_allclose(values[key], independent[key], rtol=1e-11, atol=1e-10)
    flat = np.array([np.broadcast_to(effects[k], x.shape).ravel() for k in range(1, 8)])
    gram = flat @ flat.T / total
    orthogonality_error = np.max(np.abs(gram - np.diag(np.diag(gram))))
    assert orthogonality_error < 1e-12
    np.testing.assert_allclose(sum(np.broadcast_to(e, x.shape) for e in effects.values()), x,
                               rtol=1e-11, atol=1e-10)
    return values, total, float(orthogonality_error)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, dtype={"basin_id": str, "huc2": str}, float_precision="round_trip")
    df = df[df.scenario.isin([c[1] for c in CASES]) & df.period.isin(PERIODS)].copy()
    assert len(df) == 28620
    assert not df.duplicated(["scenario", "model", "seed", "basin_id", "period"]).any()
    assert set(df.period_kind) == {"full_period"}
    np.testing.assert_allclose(df.NSE_star, df.NSE / (2 - df.NSE), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(df.abs_FHV_pct, np.abs(df.FHV_signed_pct), rtol=1e-12, atol=1e-12)
    for level in ["high", "low"]:
        np.testing.assert_allclose(df[f"nMAE_{level}"], df[f"MAE_{level}_mm_day"] / df.observed_mean_mm_day,
                                   rtol=1e-12, atol=1e-12)
    inputs = {SOURCE.relative_to(ROOT).as_posix(): sha(SOURCE)}
    full_rows, display_rows, validations, panel_meta = [], [], [], []
    stored_comparisons = 0
    maximum_metric_error = 0.0
    for panel, scenario, group, nbasin, archive in CASES:
        local = df[df.scenario == scenario]
        basins = sorted(local.basin_id.unique())
        assert len(basins) == nbasin
        assert sorted(local.seed.unique()) == SEEDS
        assert set(local.model) == {"B0", "M1", "EA"}
        path = archive / "metric_arrays.npz"
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            raw = {key: z[key] for key in z.files}
        np.testing.assert_array_equal(raw["seeds"], SEEDS)
        np.testing.assert_array_equal(raw["basins"], basins)
        assert list(raw["models"]) == ["B0", "M1", "EA"]

        def matrix(model, period, column):
            sub = local[(local.model == model) & (local.period == period)]
            assert len(sub) == len(SEEDS) * nbasin
            result = sub.pivot(index="seed", columns="basin_id", values=column).loc[SEEDS, basins].to_numpy(float)
            assert np.isfinite(result).all()
            return result

        # Reconcile source fields against the original archived score cubes.
        raw_columns = {"NSE": "nse", "KGE": "kge", "FHV_signed_pct": "fhv",
                       "MAE_high_mm_day": "mae_high", "MAE_low_mm_day": "mae_low"}
        for pi, period in enumerate(PERIODS):
            era = ["earlier", "later"][pi]
            for mi, model in enumerate(raw["models"]):
                for column, key in raw_columns.items():
                    actual = matrix(model, period, column)
                    original = raw[era][mi, :, :, list(raw["metrics"]).index(key)]
                    np.testing.assert_allclose(actual, original, rtol=1e-10, atol=1e-10)
                    maximum_metric_error = max(maximum_metric_error, float(np.abs(actual - original).max()))
                mean = matrix(model, period, "observed_mean_mm_day")
                np.testing.assert_allclose(mean, np.broadcast_to(mean[:1], mean.shape), rtol=0, atol=0)
                np.testing.assert_allclose(mean, matrix("B0", period, "observed_mean_mm_day"), rtol=0, atol=0)
        stored_path = archive / "three_factor.csv"
        inputs[stored_path.relative_to(ROOT).as_posix()] = sha(stored_path)
        stored = pd.read_csv(stored_path)
        for candidate in ["M1", "EA"]:
            for metric in METRICS:
                direction = -1 if metric in ERRORS else 1
                x = np.stack([direction * (matrix(candidate, p, metric) - matrix("B0", p, metric))
                              for p in PERIODS])
                ss, total, orthogonality_error = decompose(x)
                shares = {key: 100 * value / total for key, value in ss.items()}
                np.testing.assert_allclose(sum(shares.values()), 100, rtol=0, atol=1e-10)
                if metric in {"NSE", "NSE_star"}:
                    ref = stored[(stored.comparison == candidate + "-B0") & (stored.scale == metric)].set_index("component")
                    for key, val in shares.items():
                        old_key = key.lower().replace("catchment", "basin").replace("×", " x ")
                        np.testing.assert_allclose(val, 100 * ref.loc[old_key, "fraction"], rtol=1e-10, atol=1e-10)
                        stored_comparisons += 1
                common = dict(panel=panel, scenario=scenario, group=group, comparison=candidate + "/B0",
                              metric=metric, n_periods=2, n_seeds=10, n_catchments=nbasin)
                for key, value in ss.items():
                    full_rows.append(dict(**common, component=key, sum_of_squares=value,
                                          total_sum_of_squares=total, share_pct=shares[key]))
                display = {key: shares[key] for key in COMPONENTS[:-1]}
                display["Period terms"] = shares["Period"] + shares["Period×Seed"]
                for key in COMPONENTS:
                    display_rows.append(dict(**common, component=key, share_pct=display[key]))
                validations.append(dict(**common, share_sum_pct=sum(shares.values()),
                                        orthogonality_error=orthogonality_error))
        panel_meta.append(dict(panel=panel, group=group, n_catchments=nbasin, n_seeds=10,
                               n_periods=2, n_contrasts=2, n_metrics=6, n_display_cells=72))
    full = pd.DataFrame(full_rows)
    display = pd.DataFrame(display_rows)
    full.to_csv(DATA / "variance_components.csv", index=False, float_format="%.17g")
    display.to_csv(DATA / "variance_shares.csv", index=False, float_format="%.17g")
    assert len(full) == 168 and len(display) == 144
    np.testing.assert_allclose(display.groupby(["panel", "comparison", "metric"]).share_pct.sum(), 100, atol=1e-10)
    # Exact current-manuscript three-way percentages rounded to one decimal.
    paper_expected = {("a", "M1/B0", "NSE_star"): 12.8, ("a", "EA/B0", "NSE_star"): 13.1,
                      ("b", "M1/B0", "NSE_star"): 28.1, ("b", "EA/B0", "NSE_star"): 30.5,
                      ("a", "M1/B0", "NSE"): 38.7, ("a", "EA/B0", "NSE"): 31.3}
    for (panel, comparison, metric), expected in paper_expected.items():
        value = full[(full.panel == panel) & (full.comparison == comparison)
                     & (full.metric == metric) & (full.component == "Period×Seed×Catchment")].share_pct.item()
        assert round(value, 1) == expected
    report = dict(status="PASS", source_score_rows=28620, periods=PERIODS, seeds=SEEDS,
                  panels=panel_meta, decompositions=24, full_components=168, heatmap_cells=144,
                  source_metric_max_abs_error=maximum_metric_error,
                  stored_component_comparisons_passed=stored_comparisons,
                  manuscript_rounded_values_checked=len(paper_expected),
                  independent_decomposition="Marginal projection, additivity, and orthogonality checked for all 24 cubes",
                  interpretation="Descriptive proportions of total centered sum of squares; not random-effect variance estimates or significance tests",
                  error_metric_direction="Negative errors before candidate-minus-B0 contrast; a global sign change leaves SS shares unchanged",
                  period_terms="Period main effect plus period-by-seed interaction",
                  min_share_pct=float(display.share_pct.min()), max_share_pct=float(display.share_pct.max()),
                  validations=validations, inputs=inputs,
                  data_hashes={p.name: sha(p) for p in (DATA / "variance_components.csv", DATA / "variance_shares.csv")})
    (HERE / "variance_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"validations", "inputs", "data_hashes"}},
                     ensure_ascii=False, indent=2))
    print(display.pivot(index=["group", "comparison", "metric"], columns="component", values="share_pct").round(1).to_string())


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
