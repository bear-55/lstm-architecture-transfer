"""Collect the published interval estimates used to check the independently rebuilt forest data."""
from pathlib import Path
import json
import shutil

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE/"data"


def read(path):
    return pd.read_csv(ROOT/path, float_precision="round_trip")


def main():
    primary = read("outputs/jh_predraft_audit_20261102/main_NSE_endpoints.csv")
    influence = read("figures/Fig07/data/robustness_endpoints.csv")
    training = read("outputs/multimetric_transfer_20261102/transfer_endpoints.csv")
    budgets = read("figures/Fig05/data/curve_endpoints.csv")
    windows = read("figures/Fig04/data/window_losses.csv")
    sensitivity = json.loads((ROOT/"outputs/selection_sensitivity/estimates.json").read_text("utf-8"))
    rows = []

    def add(label, group, model, values, kind="95% interval"):
        rows.append(dict(label=label,section=group,comparison=model+"/B0",estimate=values["estimate"],
                         low95=values["low95"],high95=values["high95"],interval_kind=kind))

    for model, archive in (("M1","retraining_confirmation_analysis_20261031"),("EA","ea_selection_analysis_20261101")):
        contrast = model+"/B0"
        add("Seed + HUC2 bootstrap","Main estimate",model,
            primary[(primary.comparison == model+"-B0") & (primary.endpoint == "D")].iloc[0])
        add("Seed-only bootstrap","Main estimate",model,sensitivity["seed_only_reconstruction"][model]["D"])
        base = influence[(influence.comparison == contrast) & (influence.metric == "NSE")].set_index("variant")
        for key, label in [("basin_median","Catchment median"),("drop_largest_D_3","Omit 3 largest losses"),
                           ("drop_largest_D_7","Omit 7 largest losses"),("drop_C_7","Omit 7 largest library gains")]:
            add(label,"Catchment influence",model,base.loc[key])
        decade = read(f"outputs/{archive}/primary_endpoint.csv").set_index("metric").loc["D_same_minus_other"]
        add("WY1990–1999 selection","Selection period",model,decade)
        window = windows[windows.comparison == contrast].D
        assert len(window) == 7
        add("Seven four-year windows","Selection period",model,dict(estimate=window.mean(),low95=window.min(),
            high95=window.max()),"Observed window range")
        if model == "EA":
            perm = sensitivity["permutation_reconstruction"]["20261102"]
            add("Permuted B0 seed pairing","Seed pairing",model,dict(estimate=perm["observed"],low95=perm["low95"],
                high95=perm["high95"]),"Central 95% of permutations; point = original pairing")
        budget = budgets[(budgets.comparison == contrast) & (budgets.k == 5) & (budgets.endpoint == "D")].iloc[0]
        add("Five source instances","Selection budget",model,budget,"95% HUC2 interval conditional on the fitted library")
        subset = training[(training.scenario == "trained370") & (training.comparison == model+"-B0") &
                          (training.selection_rule == "fixed_early_NSE") & (training.metric == "NSE") &
                          (training.endpoint == "D")].set_index("aggregation")
        for key,label in (("mean","Training: mean"),("basin_median","Training: median")):
            add(label,"Evaluation setting",model,subset.loc[key])
        if model == "EA":
            for loss in ("NSE","MSE"):
                ext = read(f"outputs/weakness_resolution_20261102/external_{loss}/selection_endpoints.csv")
                row = ext[(ext.scheme == "seed_HUC2") & (ext.comparison == "EA-B0") & (ext.endpoint == "D")].iloc[0]
                add(f"External: {loss} training loss","Evaluation setting",model,row)
    pd.DataFrame(rows).to_csv(DATA/"forest_endpoints.csv",index=False)
    shutil.copyfile(HERE.parent/"FigS1/data/contrast_endpoints.csv", DATA/"matching_contrasts.csv")


if __name__ == "__main__":
    main()
