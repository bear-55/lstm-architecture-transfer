"""Rebuild Figure 2 data from verified experimental score tables (no training)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
SOURCES = {
    "scores": ROOT / "deliverables/data_request_20261103/01_scores_long.csv",
    "metadata": ROOT / "deliverables/data_request_20261103/03_basin_metadata_531.csv",
    "assignment": ROOT / "splits/formal_pub_527/basin_assignment.csv",
    "reference_losses": ROOT / "outputs/selection_transfer_P1_20261102/P1_per_basin.csv",
    "early_risk": ROOT / "outputs/training_domain_followup_20261102/early_defined_risk_per_basin.csv",
    "library_gains": ROOT / "outputs/training_domain_followup_20261102/unseen_aligned_basin_gains.csv",
}
EARLY, LATE = "WY2000-2003", "WY2004-2008"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scores = pd.read_csv(SOURCES["scores"], dtype={"basin_id": str, "huc2": str})
    meta = pd.read_csv(SOURCES["metadata"], dtype={"basin_id": str, "huc2": str})
    assignment = pd.read_csv(SOURCES["assignment"], dtype={"basin_id": str, "huc2": str})
    reference = pd.read_csv(SOURCES["reference_losses"], dtype={"basin_id": str})
    risk_table = pd.read_csv(SOURCES["early_risk"], dtype={"basin_id": str})
    gains = pd.read_csv(SOURCES["library_gains"], dtype={"basin_id": str})
    meta = meta.loc[meta.in_local527, ["basin_id", "role", "huc2", "gauge_lat", "gauge_lon"]].copy()
    assert len(meta) == meta.basin_id.nunique() == 527
    assert meta.role.value_counts().to_dict() == {"train": 370, "test": 107, "val": 50}
    assert set(meta.basin_id) == set(assignment.basin_id)
    aa = assignment.set_index("basin_id").loc[meta.basin_id]
    assert np.array_equal(aa.role.values, meta.role.values)
    np.testing.assert_allclose(aa[["gauge_lat", "gauge_lon"]], meta[["gauge_lat", "gauge_lon"]], atol=1e-12, rtol=0)
    seeds = sorted(scores.seed.unique())
    assert len(seeds) == 10

    early_b0 = scores[(scores.period == EARLY) & (scores.model == "B0")]
    assert len(early_b0) == 5270
    assert not early_b0.duplicated(["seed", "basin_id"]).any()
    assert (early_b0.groupby("basin_id").size() == 10).all()
    mu = early_b0.groupby("basin_id").NSE.mean()
    meta["early_B0_mean_NSE"] = meta.basin_id.map(mu)
    # The figure and manuscript specify transform(mean NSE), not mean(transform NSE).
    meta["early_B0_transformed_mean_NSE"] = meta.early_B0_mean_NSE / (2 - meta.early_B0_mean_NSE)
    heldout = sorted(meta.loc[meta.role == "test", "basin_id"])

    def matrix(model, period, metric):
        sub = scores[(scores.scenario == "heldout107") & (scores.model == model) & (scores.period == period)]
        assert len(sub) == 1070
        assert not sub.duplicated(["seed", "basin_id"]).any()
        a = sub.pivot(index="seed", columns="basin_id", values=metric).loc[seeds, heldout].to_numpy(float)
        assert np.isfinite(a).all()
        return a

    stats = {}
    for model in ["M1", "EA"]:
        x = matrix(model, EARLY, "NSE") - matrix("B0", EARLY, "NSE")
        y = matrix(model, LATE, "NSE_star") - matrix("B0", LATE, "NSE_star")
        decision = (x > 0).astype(float)
        same = (decision * y).mean(axis=0)
        new = (decision.sum(0) * y.sum(0) - (decision * y).sum(0)) / 90
        loss = same - new
        covariance = ((decision - decision.mean(0)) * (y - y.mean(0))).sum(0) / 9
        np.testing.assert_allclose(loss, covariance, atol=1e-14, rtol=0)
        # Independent literal enumeration of 90 ordered, distinct seed pairs.
        literal = np.array([decision[s] * y[t] for s in range(10) for t in range(10) if s != t]).mean(0)
        np.testing.assert_allclose(new, literal, atol=1e-14, rtol=0)
        ref = reference[(reference.scenario == "unseen107") & (reference.comparison == model + "-B0") & (reference.scale == "NSE_star")].set_index("basin_id").loc[heldout]
        err = float(np.max(np.abs(loss - ref.D.to_numpy())))
        # Previous window scores used float32; delivered scores use float64 accumulations.
        np.testing.assert_allclose(loss, ref.D.to_numpy(), atol=1e-6, rtol=0)
        for key, values in [("Gsame", same), ("Gnew", new), ("D", loss)]:
            meta[f"{key}_{model}_NSEstar"] = meta.basin_id.map(dict(zip(heldout, values)))
        stats[model] = dict(n=107, positive=int((loss > 0).sum()), positive_percent=float(100 * (loss > 0).mean()),
                            mean=float(loss.mean()), median=float(np.median(loss)), minimum=float(loss.min()), maximum=float(loss.max()),
                            above_template_colour_limit=int((loss > .12).sum()), below_template_colour_limit=int((loss < -.06).sum()),
                            above_template_size_limit=int((np.abs(loss) > .15).sum()), max_difference_vs_P1=err)

    risk = risk_table[(risk_table.scenario == "unseen_aligned") & (risk_table.comparison == "EA-B0") & (risk_table.policy == "library_local")].copy()
    assert len(risk) == risk.basin_id.nunique() == 107
    assert set(risk.basin_id) == set(heldout)
    assert np.array_equal(risk.early_risk.to_numpy(), (risk.early_mixed_nse < 0).to_numpy())
    risk_ids = sorted(risk.loc[risk.early_mixed_nse < 0, "basin_id"])
    assert len(risk_ids) == 7
    meta["early_EA_B0_pooled_NSE"] = meta.basin_id.map(risk.set_index("basin_id").early_mixed_nse)
    meta["early_risk"] = meta.basin_id.isin(risk_ids)

    gains = gains[(gains.scenario == "unseen_aligned") & (gains.comparison == "EA-B0") & (gains.policy == "library_local")]
    assert len(gains) == gains.basin_id.nunique() == 107
    top3 = gains.sort_values("gain", ascending=False).head(3).basin_id.tolist()
    assert set(top3) == {"04127997", "08324000", "07301410"}
    meta["top3_library_gain"] = meta.basin_id.isin(top3)
    meta["late_EA_B0_library_gain_NSE"] = meta.basin_id.map(gains.set_index("basin_id").gain)
    assert np.isfinite(meta[["gauge_lat", "gauge_lon", "early_B0_transformed_mean_NSE"]]).all().all()
    assert meta["early_B0_transformed_mean_NSE"].between(-1, 1).all()
    for model in ["M1", "EA"]:
        assert meta[f"D_{model}_NSEstar"].notna().equals(meta.role.eq("test"))
    meta.sort_values("basin_id").to_csv(DATA / "figure2_catchments.csv", index=False, float_format="%.17g")
    report = dict(status="PASS", source_hashes={key: dict(path=str(path.relative_to(ROOT)), sha256=digest(path)) for key, path in SOURCES.items()},
                  selection_period=EARLY, evaluation_period=LATE, seeds=[int(s) for s in seeds],
                  roles=meta.role.value_counts().to_dict(), early_risk_ids=risk_ids, top3_library_gain_ids=top3,
                  panel_b_definition="transform of mean individual B0 NSE: mean_NSE / (2 - mean_NSE)",
                  panel_b_range=[float(meta.early_B0_transformed_mean_NSE.min()), float(meta.early_B0_transformed_mean_NSE.max())],
                  loss_statistics=stats, ordered_distinct_seed_pairs=90,
                  geometry="CAMELS gauge longitude and latitude, preserved from the assignment table; no synthetic locations",
                  data_file_sha256=digest(DATA / "figure2_catchments.csv"))
    (HERE / "data_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "source_hashes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
