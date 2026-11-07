"""Build daily and flow-duration summaries from the nine independent EA runs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"


def main():
    cases = pd.read_csv(DATA/"case_selection.csv", dtype={"basin_id":str})
    daily = pd.read_csv(DATA/"daily_predictions.csv", dtype={"basin_id":str}, parse_dates=["date"],
                        float_precision="round_trip")
    fdc = pd.read_csv(DATA/"flow_duration_curves.csv", dtype={"basin_id":str}, float_precision="round_trip")
    with np.load(ROOT/"outputs/training_domain_followup_20261102/unseen_aligned/metric_arrays.npz",
                 allow_pickle=False) as fits:
        seeds = fits["seeds"].tolist()
    predictions, inputs = {}, []
    for seed in seeds:
        path = ROOT/f"runs/ea_evaluation_20261101/later/EA/seed_{seed}/predictions_test.npz"
        with np.load(path, allow_pickle=False) as z:
            for basin in cases.basin_id:
                mask = z["basin_ids"] == basin
                reference = daily[daily.basin_id == basin]
                np.testing.assert_array_equal(z["dates"][mask].astype("datetime64[D]"),
                                              reference.date.to_numpy(dtype="datetime64[D]"))
                np.testing.assert_allclose(z["q_observed_mm_day"][mask], reference.observed, rtol=0, atol=1e-12)
                predictions[basin, seed] = z["q_predicted_mm_day"][mask].astype(float)
        inputs.append(dict(file=path.relative_to(ROOT).as_posix(), sha256=sha256(path.read_bytes()).hexdigest()))

    days, curves, individual = [], [], []
    for case in cases.itertuples():
        independent = [seed for seed in seeds if seed != case.source_seed]
        assert len(independent) == 9
        values = np.stack([predictions[case.basin_id, seed] for seed in independent])
        frame = daily[daily.basin_id == case.basin_id].copy()
        frame["EA_new"] = values.mean(axis=0)
        frame["EA_new_q25"], frame["EA_new_q75"] = np.quantile(values, [.25, .75], axis=0)
        days.append(frame)
        base = fdc[fdc.basin_id == case.basin_id]
        independent_fdc = base[base.series == "EA_new"].sort_values("rank").copy()
        sorted_values = np.sort(values, axis=1)[:, ::-1]
        independent_fdc["q_mm_day"] = sorted_values.mean(axis=0)
        independent_fdc["q25"], independent_fdc["q75"] = np.quantile(sorted_values, [.25, .75], axis=0)
        curves.extend([base[base.series != "EA_new"], independent_fdc])
        obs = frame.observed.to_numpy()
        denominator = ((obs-obs.mean())**2).sum()
        for seed, prediction in zip(independent, values):
            individual.append(dict(basin_id=case.basin_id, source_seed=case.source_seed, evaluation_seed=seed,
                                   NSE=1-((obs-prediction)**2).sum()/denominator, n_days=len(obs)))

    pd.concat(days).to_csv(DATA/"daily_nine_instances.csv", index=False)
    pd.concat(curves).to_csv(DATA/"fdc_nine_instances.csv", index=False)
    pd.DataFrame(individual).to_csv(DATA/"independent_instance_scores.csv", index=False)
    record = dict(independent_count=9, source_excluded=True, source_files=inputs, status="PASS")
    (HERE/"independent_weights_audit.json").write_text(json.dumps(record,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
