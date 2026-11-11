from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTERNAL = ROOT / "outputs/external_eq7_check_20261111"
CSV = ROOT / "deliverables/data_request_20261103/01_scores_long.csv"
OUT = HERE / "data/catchment_components.csv"
EXT = EXTERNAL / "eq7_basins.csv"
subprocess.run(
    [sys.executable, str(ROOT / "analysis/eq7_external.py"), str(EXTERNAL / "external_scores_long.csv"),
     "--scenarios", "external_nse", "external_mse", "--sel", "selection", "--eval", "evaluation"],
    cwd=EXTERNAL, check=True,
)

df = pd.read_csv(CSV, dtype={"basin_id": str}); S = sorted(df.seed.unique())
def mat(scen, m, p):
    d = df[(df.scenario == scen) & (df.model == m) & (df.period == p)]
    return d.pivot(index="seed", columns="basin_id", values="NSE_star").loc[S].astype(float)
res = {}
for scen, g in [("heldout107", "Held-out"), ("trained370", "Training")]:
    for c in ["M1", "EA"]:
        X = mat(scen, c, "WY2000-2003") - mat(scen, "B0", "WY2000-2003")
        Y = mat(scen, c, "WY2004-2008") - mat(scen, "B0", "WY2004-2008")
        ids = X.columns.to_numpy(); assert X.index.equals(Y.index) and X.columns.equals(Y.columns)
        X, Y = X.to_numpy(), Y.to_numpy()
        I = (X > 0).astype(float); n = X.shape[0]
        D = ((I - I.mean(0)) * (Y - Y.mean(0))).sum(0) / (n - 1)
        mu, sx, sy = X.mean(0), X.std(0, ddof=1), Y.std(0, ddof=1)
        r = ((X - mu) * (Y - Y.mean(0))).sum(0) / (n - 1) / (sx * sy); z = mu / sx
        res[(g, c)] = pd.DataFrame(dict(basin_id=ids, D=D, r=r, sy=sy, z=z, ph=norm.pdf(z), P=r * sy * norm.pdf(z)))
e = pd.read_csv(EXT, dtype={"basin_id": str}); e = e[e.metric == "NSE*"]
for scen, g in [("external_nse", "External NSE loss"), ("external_mse", "External MSE loss")]:
    q = e[e.scenario == scen]
    res[(g, "EA")] = pd.DataFrame(dict(basin_id=q.basin_id.values, D=q.D_observed.values, r=q.r.values, sy=q.sigma_Y.values, z=q.z_X.values,
                                       ph=q.phi.values, P=q.D_predicted.values))
pd.concat(res, names=["setting", "architecture", "row"]).reset_index().drop(columns="row").to_csv(OUT, index=False)
for k, v in res.items(): print(k, len(v), round(v.D.mean(), 4), round(v.P.mean(), 4))
