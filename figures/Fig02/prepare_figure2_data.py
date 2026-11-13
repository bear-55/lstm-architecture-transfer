"""Verify and assemble the four Figure 2 panels from the original split and scores."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROLES = {"train": ("trained370", 370), "val": ("validation50", 50), "test": ("heldout107", 107)}
P = "climate_mean_precipitation"
F = "climate_cold_precipitation_fraction"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    split = root / "splits/formal_pub_527"
    inputs = root / "deliverables/data_request_20261103"
    files = [split / "basin_assignment.csv", split / "climate_thresholds.json",
             split / "climate_stratum_counts.csv", inputs / "03_basin_metadata_531.csv",
             inputs / "01_scores_long.csv"]
    def read(path, **kwargs):
        return pd.read_csv(path, dtype={"basin_id": str}, float_precision="round_trip", **kwargs)

    assignment = read(files[0]).set_index("basin_id").sort_index()
    meta = read(files[3]).query("in_local527").set_index("basin_id").sort_index()
    assert assignment.index.is_unique and meta.index.is_unique and len(meta) == 527
    pd.testing.assert_index_equal(assignment.index, meta.index)
    assert (meta.role == assignment.role).all() and (meta.climate_stratum == assignment.climate_stratum).all()
    np.testing.assert_allclose(meta[["gauge_lat","gauge_lon"]], assignment[["gauge_lat","gauge_lon"]], atol=1e-12, rtol=0)
    frame = assignment[["role","climate_stratum","gauge_lon","gauge_lat",P,F]].copy()
    gates = json.loads(files[1].read_text(encoding="utf-8"))
    cuts = np.array(gates["precip_terciles_mm_day"])
    medians = np.array(gates["cold_precip_medians_within_precip_tercile"])
    np.testing.assert_allclose(cuts, np.quantile(frame[P],[1/3,2/3]), atol=1e-14, rtol=0)
    tercile = np.searchsorted(cuts, frame[P], side="left")
    np.testing.assert_allclose(medians, frame[F].groupby(tercile).median(), atol=1e-14, rtol=0)
    np.testing.assert_array_equal(2*tercile + (frame[F].to_numpy() > medians[tercile]), frame.climate_stratum)
    frame["P"] = frame.pop(P)*365.25
    frame["F"] = frame.pop(F)
    sc = read(files[4], usecols=["scenario","basin_id","seed","period","model","role","NSE"])
    early = sc.query("model == 'B0' and period == 'WY2000-2003' and scenario in @scenarios",
                     local_dict={"scenarios":[v[0] for v in ROLES.values()]})
    assert len(early)==5270 and not early.duplicated(["basin_id","seed"]).any()
    assert np.isfinite(early.NSE).all() and early.seed.nunique()==10
    for role,(scenario,n) in ROLES.items():
        correct = early[early.scenario == scenario]
        assert set(correct.role)=={role} and correct.basin_id.nunique()==n
        assert set(correct.basin_id)==set(frame[frame.role==role].index)
        assert correct.groupby("basin_id").seed.nunique().eq(10).all()
    frame["early_B0_mean_NSE"] = early.groupby("basin_id").NSE.mean()
    frame["NSEs"] = frame.early_B0_mean_NSE / (2-frame.early_B0_mean_NSE)
    assert frame.NSEs.between(-1,1).all()
    counts=pd.crosstab(frame.climate_stratum,frame.role)[list(ROLES)]
    np.testing.assert_array_equal(counts,pd.read_csv(files[2],index_col=0)[list(ROLES)])
    share=counts/counts.sum()*100
    dev=share.test - counts.sum(axis=1)/len(frame)*100
    # Independently compare b and d to the underlying unmatched Figure S1 observations.
    reference=root/"figures/FigS1/data/score_ecdfs.csv"
    s1=read(reference).query("matching == 'unmatched'").set_index("basin_id")
    assert set(s1.index)==set(frame[frame.role!="val"].index) and s1.index.is_unique
    np.testing.assert_allclose(frame.loc[s1.index,"NSEs"],s1.early_B0_NSEstar,atol=1e-12,rtol=0)
    DATA.mkdir(exist_ok=True)
    outfile=DATA/"figure2_strata_skill.csv"
    frame.to_csv(outfile,float_format="%.17g")
    report={"status":"PASS","basins":len(frame),"stratum_agreement":527,
            "period_climate":"WY1981–1989","period_scores":"WY2000–2003","seeds":sorted(map(int,early.seed.unique())),
            "precipitation_terciles_mm_per_year":(cuts*365.25).tolist(),
            "cold_day_medians_within_precipitation_terciles":medians.tolist(),"tie_rule":gates["boundary_rule"],
            "stratum_counts":counts.to_dict("index"),"heldout_minus_full_pp":dev.to_dict(),
            "max_absolute_heldout_minus_full_pp":float(dev.abs().max()),"role_medians_NSEstar":frame.groupby("role").NSEs.median().to_dict(),
            "role_negative_counts":frame.NSEs.lt(0).groupby(frame.role).sum().to_dict(),
            "baseline_definition":"Transform of the ten-instance mean raw B0 NSE within each basin.",
            "comparison_to_unmatched_S1_basins":len(s1),"data_file_sha256":digest(outfile),
            "source_sha256":{p.relative_to(root).as_posix():digest(p) for p in files+[reference]}}
    (HERE/"data_verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=HERE.parents[1])
    build(parser.parse_args().root.resolve())
