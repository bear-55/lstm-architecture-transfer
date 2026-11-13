from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, gaussian_kde
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt, gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
HERE = Path(__file__).resolve().parent
MM = 1 / 25.4
W2 = 190 * MM

# ------------------------------------------------------------------ house style
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7.5, "legend.frameon": False,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "font.weight": "normal",
    "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": 1.2, "savefig.dpi": 300, "pdf.fonttype": 42,
})
RED, BLUE, GOLD, GREY, INK = "#D62728", "#1F5FA8", "#E8A317", "#7F7F7F", "#1A1A1A"
LRED, LBLUE, LGOLD = "#F2A7A3", "#9DB9DE", "#F5D48A"
PAIR = {"M1": RED, "EA": BLUE}
PAIRL = {"M1": LRED, "EA": LBLUE}
LAB = {"M1": "M1/B0", "EA": "EA/B0"}
PARCMAP = LinearSegmentedColormap.from_list("patchx_par", ["#2C5DA6", "#8FB3D9", "#F4E28A", "#F28E5C", "#D62728"])


def boxgrid(ax, axis="y"):
    ax.grid(axis=axis, color="#B5B5B5", lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.set_axisbelow(True)


def top_legend(fig, handles, labels, y=0.995, ncol=None):
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
               handlelength=1.6, columnspacing=1.6, prop={"weight": "bold", "size": 7.5})



all_data = pd.read_csv(HERE/"data/catchment_components.csv", dtype={"basin_id":str}, float_precision="round_trip")
R = dict(tuple(all_data.groupby(["setting","architecture"],sort=False)))
rng = np.random.default_rng(10)
ORDER = [("Held-out", "M1"), ("Training", "M1"), ("Held-out", "EA"), ("Training", "EA"),
         ("External NSE loss", "EA"), ("External MSE loss", "EA")]
LAB = {("Held-out", "M1"): "M1/B0 held-out", ("Training", "M1"): "M1/B0 training",
       ("Held-out", "EA"): "EA/B0 held-out", ("Training", "EA"): "EA/B0 training",
       ("External NSE loss", "EA"): "EA/LSTM external, NSE loss", ("External MSE loss", "EA"): "EA/LSTM external, MSE loss"}
COL = {("Held-out", "M1"): RED, ("Training", "M1"): "#F08A7E", ("Held-out", "EA"): BLUE, ("Training", "EA"): "#7FA6D6",
       ("External NSE loss", "EA"): "#3A3A3A", ("External MSE loss", "EA"): GOLD}
MK = {("Held-out", "M1"): "o", ("Training", "M1"): "o", ("Held-out", "EA"): "s", ("Training", "EA"): "s",
      ("External NSE loss", "EA"): "D", ("External MSE loss", "EA"): "D"}

fig = plt.figure(figsize=(W2, 150 * MM))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.15, 1, 1], height_ratios=[1, 1], wspace=0.50, hspace=0.42,
                       left=0.07, right=0.985, top=0.885, bottom=0.075)

# ---------------- (a) data collapse onto the normal density
ax = fig.add_subplot(gs[0, 0]); boxgrid(ax)
zz = np.linspace(-2.6, 2.6, 300)
ax.fill_between(zz, 0, norm.pdf(zz), color="#EFEFEF", zorder=0)
ax.plot(zz, norm.pdf(zz), color=INK, lw=1.6, zorder=4, label="φ(z), Eq. (7)")
bins = np.linspace(-2.25, 2.25, 10); mids = 0.5 * (bins[1:] + bins[:-1])
allr = []
for k in ORDER:
    d = R[k]; d = d[np.isfinite(d.z) & (d.r * d.sy > 0.005)]
    allr.append(d)
    yv = []
    for a, b in zip(bins[:-1], bins[1:]):
        q = d[(d.z > a) & (d.z <= b)]
        yv.append(q.D.sum() / (q.r * q.sy).sum() if len(q) >= 8 else np.nan)
    ax.plot(mids, yv, color=COL[k], lw=0.8, alpha=0.8, marker=MK[k], ms=2.8, mec="none", zorder=2)
A = pd.concat(allr); pm, lo, hi = [], [], []
# Resample distinct basin IDs jointly across settings, preserving repeat observations.
basins = np.asarray(sorted(all_data.basin_id.unique()))
weights = rng.multinomial(len(basins), np.full(len(basins), 1/len(basins)), size=1000)
n_used = 0
for a, b in zip(bins[:-1], bins[1:]):
    q = A[(A.z > a) & (A.z <= b)].copy()
    n_used += len(q)
    q["den"] = q.r*q.sy
    grouped = q.groupby("basin_id")[["D","den"]].sum().reindex(basins,fill_value=0)
    est = grouped.D.sum()/grouped.den.sum()
    den = weights @ grouped.den.to_numpy()
    bs = (weights[den>0] @ grouped.D.to_numpy()) / den[den>0]
    pm.append(est); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
assert n_used == 1329 and len(A) == 1352
ax.errorbar(mids, pm, [np.array(pm) - lo, np.array(hi) - pm], fmt="o", ms=5, color="white", mec=INK, mew=1.0,
            ecolor=INK, elinewidth=0.9, capsize=2, zorder=5)
ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.05, 0.52)
ax.set_xlabel(r"Standardized historical advantage $z=\mu_X/\sigma_X$")
ax.set_ylabel(r"$D_b\,/\,(r_b\,\sigma_{Y,b})$")
ax.text(0.03, 0.96, f"{n_used:,} binned pairs\nall six settings", transform=ax.transAxes, va="top", fontsize=6.4, fontweight="bold")
ax.set_title("(a) Binned losses across settings", fontsize=8.5, fontweight="bold")

# ---------------- (b) catchment-level agreement
ax = fig.add_subplot(gs[0, 1]); boxgrid(ax)
for k in ORDER[::-1]:
    d = R[k]
    ax.scatter(d.P, d.D, s=4 if "External" in k[0] else 7, marker=MK[k], color=COL[k], alpha=0.35 if "External" in k[0] else 0.6,
               lw=0, zorder=2 if "External" in k[0] else 3)
lim = [-0.12, 0.35]; ax.plot(lim, lim, color=INK, lw=0.9, ls="--", zorder=4)
ax.set_xlim(lim); ax.set_ylim(lim)
allD = pd.concat(R.values())
ax.text(0.04, 0.96, f"Spearman ρ = {spearmanr(allD.D, allD.P)[0]:.2f}\n(per setting 0.73–0.94)", transform=ax.transAxes,
        va="top", fontsize=6.6, fontweight="bold")
ax.set_xlabel(r"Predicted $D_b = r\,\sigma_Y\,\varphi(z)$ (NSE*)"); ax.set_ylabel(r"Observed $D_b$ (NSE*)")
ax.set_title("(b) Catchment-level losses", fontsize=8.5, fontweight="bold")

# ---------------- (c) regional means across settings (log-log)
ax = fig.add_subplot(gs[0, 2]); boxgrid(ax)
for k in ORDER:
    d = R[k]; n = len(d)
    bD, bP = [], []
    for _ in range(2000):
        i = rng.integers(0, n, n); bD.append(d.D.values[i].mean()); bP.append(d.P.values[i].mean())
    x, y = d.P.mean(), d.D.mean()
    ax.errorbar(x, y, xerr=[[x - np.percentile(bP, 2.5)], [np.percentile(bP, 97.5) - x]],
                yerr=[[y - np.percentile(bD, 2.5)], [np.percentile(bD, 97.5) - y]],
                fmt=MK[k], ms=6, color=COL[k], mec=INK, mew=0.6, elinewidth=0.9, capsize=1.8, zorder=4)
lim = [0.0012, 0.08]; ax.plot(lim, lim, color=INK, lw=0.9, ls="--")
ax.fill_between(lim, [l / 1.5 for l in lim], [l * 1.5 for l in lim], color="#EDEDED", zorder=0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xticks([0.002, 0.005, 0.01, 0.02, 0.05]); ax.set_xticklabels(["0.002", "0.005", "0.01", "0.02", "0.05"])
ax.set_yticks([0.002, 0.005, 0.01, 0.02, 0.05]); ax.set_yticklabels(["0.002", "0.005", "0.01", "0.02", "0.05"])
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
ax.text(0.97, 0.05, "shaded: within ×1.5", transform=ax.transAxes, ha="right", fontsize=6.2, color=GREY, style="italic")
ax.set_xlabel("Predicted regional mean D (NSE*)"); ax.set_ylabel("Observed regional mean D (NSE*)")
ax.set_title("(c) Regional loss, six settings", fontsize=8.5, fontweight="bold")

# ---------------- (d) mechanism landscape
ax = fig.add_subplot(gs[1, 0])
Z, Rg = np.meshgrid(np.linspace(-2.4, 2.4, 300), np.linspace(-0.3, 1.0, 300))
M = Rg * norm.pdf(Z)
cmap = LinearSegmentedColormap.from_list("signed", [(0.0,"#5691BF"),(.5,"#FFFFFF"),(.74,"#F4AA82"),(1.0,"#8E1B1B")])
scale = TwoSlopeNorm(vmin=-.12,vcenter=0,vmax=.4)
cf = ax.contourf(Z, Rg, M, levels=np.linspace(-.12,.40,27), cmap=cmap, norm=scale, zorder=0)
cs = ax.contour(Z, Rg, M, levels=[0.05, 0.1, 0.2, 0.3], colors="#7F7F7F", linewidths=0.5, zorder=1)
ax.clabel(cs, fmt="%.2f", fontsize=5.5)
for k in ORDER:
    d = R[k]; d = d[np.isfinite(d.z)]
    zq = np.percentile(d.z, [25, 50, 75]); rq = np.percentile(d.r, [25, 50, 75])
    ax.plot([zq[0], zq[2]], [rq[1], rq[1]], color=COL[k], lw=1.1, zorder=3)
    ax.plot([zq[1], zq[1]], [rq[0], rq[2]], color=COL[k], lw=1.1, zorder=3)
    ax.plot(zq[1], rq[1], MK[k], ms=6.5, color=COL[k], mec=INK, mew=0.7, zorder=4)
ax.annotate("", xy=(-0.33, 0.22), xytext=(-0.2, 0.69), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=INK,
            connectionstyle="arc3,rad=0.3"), zorder=5)
ax.text(0.35, 0.47, "held-out → training\n→ external", fontsize=6.2, fontweight="bold", style="italic")
for s in ["top", "right", "bottom", "left"]: ax.spines[s].set_linewidth(1.0)
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-0.3, 1.0)
ax.set_xlabel(r"Architecture tie $z=\mu_X/\sigma_X$"); ax.set_ylabel(r"Persistence $r$")
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=scale), ax=ax, ticks=[-.1,0,.1,.2,.3,.4], fraction=0.05, pad=0.02)
cb.set_label(r"$r\,\varphi(z)$", fontsize=6.8); cb.ax.tick_params(labelsize=6)
ax.set_title("(d) Where catchment-wise choice loses", fontsize=8.5, fontweight="bold")

# ---------------- (e) ridgelines of persistence
ax = fig.add_subplot(gs[1, 1]); boxgrid(ax, "x")
xs = np.linspace(-1, 1, 300)
for i, k in enumerate(ORDER[::-1]):
    v = R[k].r.values; v = v[np.isfinite(v)]
    den = gaussian_kde(v, bw_method=0.3)(xs); den = den / den.max() * 0.85
    ax.fill_between(xs, i, i + den, color=COL[k], alpha=0.55, lw=0, zorder=2 + i)
    ax.plot(xs, i + den, color=COL[k], lw=0.9, zorder=2 + i)
    md = np.median(v); ax.plot([md, md], [i, i + 0.85 * gaussian_kde(v, bw_method=0.3)(md)[0] / gaussian_kde(v, bw_method=0.3)(xs).max()],
                               color=INK, lw=1.0, zorder=10)
    ax.text(-0.97, i + 0.52, f"{LAB[k]}", fontsize=5.7, fontweight="bold", color=INK, zorder=11,
            bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.75))
    ax.text(md + 0.03, i + 0.06, f"{md:.2f}", fontsize=6.0, fontweight="bold", ha="left", zorder=11)
ax.axvline(0, color=GREY, lw=0.6, ls=":")
ax.set_yticks([]); ax.set_xlim(-1, 1); ax.set_ylim(-0.1, len(ORDER) + 0.05)
ax.set_xlabel(r"Persistence $r$ (vertical line: median)")
ax.set_title("(e) Persistence of instance advantages", fontsize=8.5, fontweight="bold")

# ---------------- (f) fold-change decomposition relative to local held-out
ax = fig.add_subplot(gs[1, 2]); boxgrid(ax, "x")
pairs = [(("Training", "M1"), ("Held-out", "M1"), "M1 train."), (("Training", "EA"), ("Held-out", "EA"), "EA train."),
         (("External NSE loss", "EA"), ("Held-out", "EA"), "Ext. NSE"), (("External MSE loss", "EA"), ("Held-out", "EA"), "Ext. MSE")]
comp = [("r", "Persistence r", "#6A51A3"), ("sy", r"Seed spread $\sigma_Y$", "#41AB5D"), ("ph", r"Tie $\varphi(z)$", "#FD8D3C")]
for yi, (k, ref, lab) in enumerate(pairs):
    left = 0
    parts = np.array([np.log2(np.median(R[k][key]) / np.median(R[ref][key])) for key, _, _ in comp])
    total = np.log2(R[k].P.mean() / R[ref].P.mean())          # predicted regional ratio
    parts = parts / parts.sum() * total                         # shares from median ratios, scaled to the predicted ratio
    for f, (key, cl, col) in zip(parts, comp):
        ax.barh(yi, f, 0.5, left=left, color=col, ec=INK, lw=0.4, zorder=3); left += f
    obs = np.log2(R[k].D.mean() / R[ref].D.mean())
    ax.plot(left, yi, "|", ms=11, mew=1.6, color=INK, zorder=4)
    ax.plot(obs, yi, "*", ms=9, color=COL[k], mec=INK, mew=0.6, zorder=5)
    ax.text(min(left, obs) - 0.28, yi, f"×{2 ** obs:.2f}", ha="right", va="center", fontsize=6.3, fontweight="bold")
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[2] for p in pairs], fontsize=6.6, fontweight="bold"); ax.set_ylim(len(pairs) + 1.55, -0.6)
ax.set_xlim(-4.4, 0.4); ax.set_xticks([-4, -3, -2, -1, 0]); ax.set_xticklabels(["1/16", "1/8", "1/4", "1/2", "1"])
ax.set_xlabel("Factor relative to local held-out")
ax.legend([Patch(fc=c[2], ec=INK, lw=0.4) for c in comp] + [Line2D([], [], marker="|", ls="none", color=INK, ms=9, mew=1.5),
           Line2D([], [], marker="*", ls="none", color=GREY, mec=INK, ms=8)],
          [c[1] for c in comp] + ["Sum = predicted", "Observed D"], loc="lower center", prop={"size": 5.8, "weight": "bold"},
          ncol=2, columnspacing=0.8, handlelength=1.2, borderpad=0.4, labelspacing=0.3)
ax.set_title("(f) Median log-ratio allocation", fontsize=8.5, fontweight="bold")

top_legend(fig, [Line2D([], [], marker=MK[k], ls="none", color=COL[k], mec=INK, mew=0.5, ms=5.5) for k in ORDER] +
           [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, ms=5)],
           [LAB[k] for k in ORDER] + ["All settings pooled (95% CI)"], y=0.995, ncol=4)
for ext, kw in [("png", dict(dpi=600)), ("pdf", {}), ("svg", {}), ("tif", dict(dpi=600, pil_kwargs={"compression": "tiff_lzw"}))]:
    fig.savefig(HERE / f"fig9.{ext}", **kw)
print("ok")
