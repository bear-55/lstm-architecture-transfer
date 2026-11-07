"""Build portable 370/50/107 cache; fit only Train, never open Test Q."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from project import digest, save_json
from dataset import read_forcing, DevelopmentStreamflow, valid_window_ends
from preprocessing import fit_static, apply_static, fit_dynamic, apply_dynamic

OUT = ROOT / "data/formal"
SPLIT = ROOT / "splits/formal_pub_527"


def main():
    if (OUT / "metadata.json").exists():
        from formal import load_formal
        load_formal()
        print("Verified existing formal cache; no regeneration")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    sm = json.loads((SPLIT / "split_manifest.json").read_text(encoding="utf-8"))
    for name, expected in sm["output_hashes"].items():
        assert digest(SPLIT / name.replace("\\", "/")) == expected
    splits = {r: (SPLIT / f"{r}.txt").read_text().split() for r in ("train", "val", "test")}
    assert {r: len(v) for r, v in splits.items()} == {"train": 370, "val": 50, "test": 107}
    ids = sorted(sum(splits.values(), []))
    assert len(set(ids)) == 527
    # Use the formal configuration directly; never recreate it from the historical pilot.
    cfg = json.loads((ROOT / "configs/formal.json").read_text(encoding="utf-8"))
    assert cfg["protocol"] == "formal_random_pub_527" and cfg["split_counts"] == sm["counts"]
    manifest = pd.read_csv(ROOT / "data/manifests/source_files.csv", dtype={"basin_id": str})
    forcing = manifest[manifest.kind == "daymet"].set_index("basin_id")
    qfiles = manifest[(manifest.kind == "streamflow_metadata_only") & manifest.basin_id.isin(splits["train"]+splits["val"])].set_index("basin_id").path.to_dict()
    expected = json.loads((ROOT / "data_contract/formal_531_audit/source_hashes.json").read_text(encoding="utf-8"))
    expected = {x["basin_id"]: x for x in expected["basins"]}
    dates = np.arange(np.datetime64("1989-10-01")-np.timedelta64(364, "D"), np.datetime64("2008-10-01"))
    dynamic = np.empty((527, len(dates), 5), dtype=np.float64)
    valid = np.empty((527, len(dates)), dtype=bool)
    areas = {}
    for i, basin in enumerate(ids):
        path = Path(forcing.loc[basin, "path"])
        assert digest(path) == expected[basin]["forcing_sha256"]
        days, values, mask, area = read_forcing(path)
        pos = np.searchsorted(days, dates)
        assert np.array_equal(days[pos], dates)
        dynamic[i], valid[i], areas[basin] = values[pos], mask[pos], area
    fields = pd.read_csv(ROOT / "data_contract/feature_manifest.csv")
    static_fields = fields[fields.role.isin(["static_climate", "static_land"])]
    names = static_fields.standard_name.tolist()
    land_names = static_fields.loc[static_fields.group == "L", "standard_name"].tolist()
    # These are raw, feature-specific climate aggregates and original land attributes.
    raw_table = pd.read_csv(SPLIT / "candidate_attributes.csv", dtype={"basin_id": str}).set_index("basin_id")
    static = raw_table.loc[ids, names].to_numpy(dtype=np.float64)
    train_rows = [ids.index(b) for b in splits["train"]]
    ss = fit_static(static, names, train_rows, cfg["features"]["static_missing_indicators"], cfg["features"]["static_log1p"])
    ds = fit_dynamic(dynamic, valid, dates, train_rows, *cfg["periods"]["train_labels"])
    xs, xd = apply_static(static, ss), apply_dynamic(dynamic, valid, ds)
    full_names = names + [x+"_missing" for x in cfg["features"]["static_missing_indicators"]]
    full_land_names = land_names + [x+"_missing" for x in cfg["features"]["static_missing_indicators"] if x in land_names]
    label_periods = {"train": cfg["periods"]["train_labels"], "val": cfg["periods"]["development_labels"]}
    reader = DevelopmentStreamflow(qfiles, splits, label_periods)
    sigma, means, windows, arrays = {}, {}, {}, {}
    qc = pd.read_csv(ROOT / "data_contract/formal_531_audit/basin_audit.csv", dtype={"basin_id": str}).set_index("basin_id")
    for role in ("train", "val"):
        arr = np.full((len(splits[role]), len(dates)), np.nan, dtype=np.float32)
        for j, basin in enumerate(splits[role]):
            assert digest(Path(qfiles[basin])) == expected[basin]["q_sha256"]
            days, values = reader.read(basin, role, areas[basin])
            arr[j, np.searchsorted(dates, days)] = values
            row = ids.index(basin)
            ends = valid_window_ends(valid[row], dates, *label_periods[role], 365, arr[j])
            assert len(ends) == qc.loc[basin, role+"_joint_valid_days"]
            assert len(ends) >= .8*qc.loc[basin, role+"_expected_days"]
            windows[basin] = {"role": role, "valid_windows": len(ends)}
            if role == "train":
                observed = values[np.isfinite(values)]
                sigma[basin], means[basin] = float(observed.std(ddof=0)), float(observed.mean())
        arrays["q_"+role] = arr
    assert set(sigma) == set(splits["train"])
    assert {r["basin_id"] for r in reader.accesses}.isdisjoint(splits["test"])
    qref = max(float(np.median(list(means.values()))), .1)
    np.savez_compressed(OUT / "formal.npz", basin_ids=np.array(ids), dates=dates, x_dynamic=xd,
                        dynamic_valid=valid, x_static=xs, **arrays)
    save_json(OUT / "scaler.json", {"source_basin_ids": splits["train"], "source_label_period": label_periods["train"],
              "dynamic": ds, "static": ss, "q_ref_mm_day": qref, "source_sigma_mm_day": sigma, "sigma_ddof": 0})
    save_json(OUT / "label_access.json", {"scope": "formal_train_and_val_only", "test_q_read_by_this_builder": False,
              "prior_test_quality_audit": True, "records": reader.accesses})
    files = ["configs/formal.json", "data/formal/formal.npz", "data/formal/scaler.json", "data/formal/label_access.json",
             "data_contract/feature_manifest.csv", "splits/formal_pub_527/split_manifest.json", "splits/formal_pub_527/candidate_attributes.csv",
             "splits/formal_pub_527/train.txt", "splits/formal_pub_527/val.txt", "splits/formal_pub_527/test.txt", "scripts/build_formal_dataset.py",
             "src/dataset.py", "src/preprocessing.py", "src/model.py"]
    meta = {"splits": splits, "sequence_length": 365, "dynamic_names": cfg["features"]["dynamic"],
            "static_names": full_names, "land_names": full_land_names, "land_indices": [full_names.index(x) for x in full_land_names],
            "label_periods": label_periods, "q_ref_mm_day": qref, "sigma_train_mm": sigma, "basin_windows": windows,
            "target_unit": "mm/day", "test_q_in_cache": False, "test_q_read_by_this_builder": False,
            "prior_test_quality_audit": True, "source_hashes": {f: digest(ROOT / f) for f in files}}
    save_json(OUT / "metadata.json", meta)
    print(json.dumps({"counts": cfg["split_counts"], "static_size": len(full_names), "land_size": len(full_land_names),
            "train_windows": sum(x["valid_windows"] for x in windows.values() if x["role"] == "train"),
            "val_windows": sum(x["valid_windows"] for x in windows.values() if x["role"] == "val"),
            "cache_MiB": (OUT / "formal.npz").stat().st_size/2**20, "test_q_in_cache": False}))


if __name__ == "__main__":
    main()
