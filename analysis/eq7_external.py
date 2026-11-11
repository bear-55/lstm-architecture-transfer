"""
Check Eq. (7) of Section 4.2 on the external multi-seed archive (or any multi-seed score table).

Input: a long-format CSV with one row per (scenario, model, seed, basin, period), e.g.
    scenario,model,seed,basin_id,period,NSE
    external_nse,EA,1,01013500,selection,0.71
    external_nse,LSTM,1,01013500,selection,0.69
    ...
Column names can be changed with the options below. NSE* is computed as NSE/(2-NSE) if no NSE* column exists.

Example:
    python3 eq7_external.py external_scores_long.csv \
        --scenarios external_nse external_mse --cand EA --base LSTM \
        --sel <selection-period name> --eval <evaluation-period name>

Outputs (written to the current folder):
    eq7_summary.csv   one row per scenario x metric
    eq7_basins.csv    one row per scenario x metric x basin (observed D_b, predicted D_b, r, sigma_Y, mu_X/sigma_X)
"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--scenarios", nargs="+", required=True)
ap.add_argument("--cand", default="EA"); ap.add_argument("--base", default="LSTM")
ap.add_argument("--sel", required=True, help="period label of the selection (historical) window")
ap.add_argument("--eval", required=True, help="period label of the evaluation (late) window")
ap.add_argument("--col-scenario", default="scenario"); ap.add_argument("--col-model", default="model")
ap.add_argument("--col-seed", default="seed"); ap.add_argument("--col-basin", default="basin_id")
ap.add_argument("--col-period", default="period"); ap.add_argument("--col-nse", default="NSE")
ap.add_argument("--col-nsestar", default=None, help="optional existing NSE* column")
a = ap.parse_args()

df = pd.read_csv(a.csv, dtype={a.col_basin: str})
df["NSE_"] = df[a.col_nse].astype(float)
df["NSEstar_"] = df[a.col_nsestar].astype(float) if a.col_nsestar else df["NSE_"] / (2 - df["NSE_"])

def mat(scen, model, period, col):
    d = df[(df[a.col_scenario] == scen) & (df[a.col_model] == model) & (df[a.col_period] == period)]
    if d.empty:
        raise SystemExit(f"No rows for scenario={scen}, model={model}, period={period}. Check the names.")
    p = d.pivot_table(index=a.col_seed, columns=a.col_basin, values=col, aggfunc="first")
    return p

rows, brows = [], []
for scen in a.scenarios:
    for metric, col in [("NSE*", "NSEstar_"), ("NSE", "NSE_")]:
        mats = [mat(scen, m, p, col) for m in (a.cand, a.base) for p in (a.sel, a.eval)]
        seeds = sorted(set.intersection(*[set(m.index) for m in mats]))
        basins = sorted(set.intersection(*[set(m.columns) for m in mats]))
        cS, cE, bS, bE = [m.loc[seeds, basins].values for m in mats]
        X, Y = cS - bS, cE - bE                      # candidate-minus-baseline advantages (seeds x basins)
        ok = np.isfinite(X).all(0) & np.isfinite(Y).all(0)
        X, Y, basins = X[:, ok], Y[:, ok], [b for b, k in zip(basins, ok) if k]
        n = X.shape[0]; I = (X > 0).astype(float)
        D = ((I - I.mean(0)) * (Y - Y.mean(0))).sum(0) / (n - 1)          # observed D_b (Eq. 5)
        mu, sx, sy = X.mean(0), X.std(0, ddof=1), Y.std(0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = ((X - mu) * (Y - Y.mean(0))).sum(0) / (n - 1) / (sx * sy)
            z = mu / sx
        r = np.nan_to_num(r); z = np.where(sx > 0, z, np.inf)
        ph = norm.pdf(z); P = r * sy * ph                                    # Eq. (7)
        rows.append(dict(scenario=scen, metric=metric, n_seeds=n, n_basins=len(basins),
                         D_observed_mean=D.mean(), D_predicted_mean=P.mean(),
                         spearman_obs_pred=spearmanr(D, P)[0],
                         r_median=np.median(r), r_q1=np.percentile(r, 25), r_q3=np.percentile(r, 75),
                         sigmaY_median=np.median(sy), sigmaY_q1=np.percentile(sy, 25), sigmaY_q3=np.percentile(sy, 75),
                         phi_median=np.median(ph), phi_q1=np.percentile(ph, 25), phi_q3=np.percentile(ph, 75),
                         ceiling_mean_0p40_r_sigmaY=(0.3989 * r * sy).mean()))
        brows.append(pd.DataFrame(dict(scenario=scen, metric=metric, basin_id=basins, D_observed=D, D_predicted=P,
                                       r=r, sigma_Y=sy, z_X=z, phi=ph)))
S = pd.DataFrame(rows); S.to_csv("eq7_summary.csv", index=False)
pd.concat(brows).to_csv("eq7_basins.csv", index=False)
pd.set_option("display.width", 200)
print(S[["scenario", "metric", "n_seeds", "n_basins", "D_observed_mean", "D_predicted_mean", "spearman_obs_pred",
         "r_median", "sigmaY_median", "phi_median"]].round(4).to_string(index=False))
print("\nSanity check: on NSE, D_observed_mean should equal Table S9b (0.00252 for NSE loss, 0.00941 for MSE loss).")
