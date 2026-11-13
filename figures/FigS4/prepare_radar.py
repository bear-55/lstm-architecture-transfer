"""Centre each metric on the pooled-ensemble median for its catchment group."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
METRICS = ["nse", "kge", "r", "alpha_closeness", "beta_closeness", "abs_fhv", "mae_high", "mae_low"]
LOWER_IS_BETTER = {"abs_fhv", "mae_high", "mae_low"}


def nice_ceiling(value):
    magnitude = 10 ** np.floor(np.log10(value))
    return next(multiple*magnitude for multiple in (1,2,2.5,5,10) if multiple*magnitude >= value-1e-12)


def main():
    profiles = pd.read_csv(DATA / "radar_profiles.csv", float_precision="round_trip")
    shared = dict(zip(METRICS, [.02, .025, .02, .1, .05, 5, .2, .0025]))
    risk = dict(zip(METRICS, [1, .5, .025, .2, .25, 20, .25, .1]))
    references = profiles[profiles.policy == "mixed_ensemble"].set_index(["group", "metric"])["median"]
    for row in profiles.itertuples():
        difference = abs(row.median - references.loc[(row.group, row.metric)])
        widths = risk if row.group == "early_risk" else shared
        if difference:
            widths[row.metric] = max(widths[row.metric], nice_ceiling(difference))

    records = []
    for row in profiles.itertuples():
        pooled = references.loc[(row.group, row.metric)]
        oriented = (row.median-pooled) * (-1 if row.metric in LOWER_IS_BETTER else 1)
        half_range = (risk if row.group == "early_risk" else shared)[row.metric]
        assert abs(oriented) <= half_range + 1e-12
        records.append(dict(group=row.group, policy=row.policy, metric=row.metric, n_catchments=row.n_catchments,
                            median=row.median, pooled_median=pooled, better_oriented_difference=oriented,
                            axis_half_range=half_range, radius=.5 + .5*oriented/half_range))
    pd.DataFrame(records).to_csv(DATA / "reference_radar_profiles.csv", index=False)


if __name__ == "__main__":
    main()
