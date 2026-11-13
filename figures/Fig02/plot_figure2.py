"""Render Figure 2 from the verified catchment strata and early baseline scores.
Run prepare_figure2_data.py to rebuild the display table from frozen records.
"""
from pathlib import Path as FilePath
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection
from PIL import Image

HERE = FilePath(__file__).resolve().parent
DATA = HERE / "data"
# ------------------------------------------------------------------ style
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titleweight": "bold", "axes.labelweight": "bold",
    "axes.linewidth": 0.8, "xtick.major.size": 3, "ytick.major.size": 3, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
INK, GR, LG, HI = "#1A1A1A", "#8C8C8C", "#D9D9D9", "#B2182B"
ROLE = {"train": dict(lab="Training", marker="^", s=7, fc="#99A3AB", ec="none", lw=0, z=2),
        "val": dict(lab="Validation", marker="D", s=9, fc="#E6AA4F", ec="#614620", lw=0.5, z=3),
        "test": dict(lab="Held-out", marker="o", s=10, fc="#264C69", ec="none", lw=0, z=4)}


def read_geojson(path):
    with path.open(encoding="utf-8") as handle:
        features = json.load(handle)["features"]
    parts = []
    for feature in features:
        geo = feature["geometry"]
        polygons = [geo["coordinates"]] if geo["type"] == "Polygon" else geo["coordinates"]
        for rings in polygons:
            for ring in rings:
                parts.append(np.asarray(ring, float)[:, :2])
    return parts

STATES = read_geojson(DATA / "census_states_5m_2024.geojson")
HUC2 = read_geojson(DATA / "usgs_huc2_regions.geojson")
def basemap(ax):
    if STATES:
        verts, codes = [], []
        for p in STATES:
            if len(p) < 3: continue
            verts += list(p); codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1)
        ax.add_patch(PathPatch(Path(verts, codes), fc="#F2F2F2", ec="#A6A6A6", lw=0.35, zorder=0))
    if HUC2:
        ax.add_collection(LineCollection(HUC2, colors="#7F7F7F", linewidths=0.45, linestyles=(0, (3, 1.5)), zorder=1))
    ax.set_xlim(-125.5, -66.0); ax.set_ylim(24.0, 50.0); ax.set_aspect(1.25)
    ax.set_xticks([-120, -105, -90, -75]); ax.set_xticklabels(["120°W", "105°W", "90°W", "75°W"])
    ax.set_yticks([30, 40, 50]); ax.set_yticklabels(["30°N", "40°N", "50°N"])
    ax.tick_params(labelsize=7)

# ------------------------------------------------------------------ data

meta = pd.read_csv(DATA / "figure2_strata_skill.csv", dtype={"basin_id":str}, float_precision="round_trip")
report = json.loads((HERE / "data_verification.json").read_text(encoding="utf-8"))
assert hashlib.sha256((DATA / "figure2_strata_skill.csv").read_bytes()).hexdigest() == report["data_file_sha256"]
NROLE = meta.role.value_counts().to_dict()
assert NROLE == {"train":370, "val":50, "test":107} and meta.basin_id.is_unique
t1, t2 = report["precipitation_terciles_mm_per_year"]
medians = np.asarray(report["cold_day_medians_within_precipitation_terciles"])
tercile = np.searchsorted([t1, t2], meta.P, side="left")
np.testing.assert_array_equal(2*tercile + (meta.F.to_numpy() > medians[tercile]), meta.climate_stratum)
share = pd.crosstab(meta.climate_stratum, meta.role, normalize="columns") * 100
full = meta.climate_stratum.value_counts(normalize=True).sort_index() * 100
dev = share["test"] - full
# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(190 * MM, 138 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.22, wspace=0.24,
                      left=0.065, right=0.905, top=0.965, bottom=0.075)

# (a) roles
ax = fig.add_subplot(gs[0, 0]); basemap(ax)
for r in ["train", "val", "test"]:
    d = meta[meta.role == r]; s = ROLE[r]
    ax.scatter(d.gauge_lon, d.gauge_lat, marker=s["marker"], s=s["s"], fc=s["fc"], ec=s["ec"], lw=s["lw"],
               zorder=s["z"] + 2, label=f'{s["lab"]} ({len(d)})')
ax.legend(loc="lower left", prop={"weight": "bold", "size": 6.8}, frameon=False, handletextpad=0.2, borderaxespad=0.3)
ax.set_title("(a) Catchment roles")

# (b) early skill map
ax = fig.add_subplot(gs[0, 1]); basemap(ax)
o = meta.sort_values("NSEs", ascending=False)
h = ax.scatter(o.gauge_lon, o.gauge_lat, c=o.NSEs, cmap="YlOrRd_r", vmin=-1, vmax=1, s=9, ec="none", zorder=4)
cax = ax.inset_axes([1.03, 0.06, 0.035, 0.88]); cb = fig.colorbar(h, cax=cax, ticks=[-1, -0.5, 0, 0.5, 1])
cb.set_label("Early B0 NSE* (transformed mean)", fontweight="bold", fontsize=7); cb.ax.tick_params(labelsize=6.5)
ax.set_title("(b) Early-period skill of B0")

# (c) stratification space
ax = fig.add_subplot(gs[1, 0])
for r in ["train", "val", "test"]:
    d = meta[meta.role == r]; s = ROLE[r]
    ax.scatter(d.P, d.F, marker=s["marker"], s=s["s"] * 1.15, fc=s["fc"], ec=s["ec"], lw=s["lw"], zorder=s["z"])
xlo, xhi = meta.P.min() / 1.15, meta.P.max() * 1.15
ax.set_xscale("log"); ax.set_xlim(xlo, xhi)
ax.set_yscale("function", functions=(lambda v: np.sqrt(np.clip(v, 0, None)), np.square))
yhi = max(0.9, meta.F.max() * 1.05); ax.set_ylim(0, yhi)
yt = [t for t in [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8] if t <= yhi]; ax.set_yticks(yt); ax.set_yticklabels([f"{t:g}" for t in yt])
xt = [t for t in [200, 300, 500, 1000, 2000, 4000] if xlo <= t <= xhi]; ax.set_xticks(xt); ax.set_xticklabels([str(t) for t in xt])
ax.minorticks_off()
for v in (t1, t2): ax.axvline(v, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
edges = [xlo, t1, t2, xhi]
for j, fm in enumerate(medians):
    ax.hlines(fm, edges[j], edges[j+1], color=INK, lw=0.7, linestyle=(0, (4, 2)), zorder=1)
xc = [np.sqrt(xlo * t1), np.sqrt(t1 * t2), np.sqrt(t2 * xhi)]
big = dev.abs().idxmax()
for s in range(6):
    xi, hi = divmod(s, 2)
    fm = medians[xi]
    y = yhi * 0.96 if hi else fm * 0.93
    txt = (f"T {share.loc[s, 'train']:.1f}\nV {share.loc[s, 'val']:.0f}\nH {share.loc[s, 'test']:.1f}\n({dev[s]:+.2f})")
    hl = s == big
    ax.text(xc[xi], y, txt, ha="center", va="top", fontsize=6.1, linespacing=1.15, zorder=6,
            color=HI if hl else INK, fontweight="bold" if hl else "normal",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=HI if hl else LG, lw=0.6, alpha=0.9))
ax.set_xlabel("Mean annual precipitation (mm yr$^{-1}$, log scale)")
ax.set_ylabel("Cold-day precipitation fraction (√ scale)")
ax.set_title("(c) Stratified split in climate space")

# (d) ECDFs
ax = fig.add_subplot(gs[1, 1])
LS = {"train": dict(color="#838F98", lw=1.8, ls="-"), "val": dict(color="#A16C19", lw=1.3, ls=(0, (4, 1.6))),
      "test": dict(color="#264C69", lw=1.4, ls="-")}
for r in ["train", "val", "test"]:
    v = np.sort(meta.loc[meta.role == r, "NSEs"].values); y = np.arange(1, len(v) + 1) / len(v); med = np.median(v)
    ax.step(np.r_[-1, v, 1], np.r_[0, y, 1], where="post", **LS[r], zorder=3 if r == "test" else 2,
            label=f'{ROLE[r]["lab"]} ({len(v)}), median {med:.2f}')
    ax.plot([med], [0.5], marker=ROLE[r]["marker"], ms=4.5, mfc=ROLE[r]["fc"], mec=INK if r != "train" else "#9E9E9E",
            mew=0.7, zorder=5)
ax.axhline(0.5, color=LG, lw=0.6, zorder=0); ax.axvline(0, color=LG, lw=0.6, zorder=0)
ax.set_xlim(-1, 1); ax.set_ylim(0, 1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_xlabel("Early B0 NSE* (WY2000–2003)"); ax.set_ylabel("Cumulative fraction of catchments")
ax.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=2.2)
neg = meta.assign(neg=meta.NSEs < 0).groupby("role").neg.sum()
ax.text(-0.96, 0.33, "NSE* < 0\n" + "\n".join(f'{ROLE[r]["lab"]}: {int(neg[r])} of {NROLE[r]}' for r in ["train", "val", "test"]),
        fontsize=6.4, va="top")
ax.grid(True, lw=0.3, color="#EEEEEE"); ax.set_axisbelow(True)
ax.set_title("(d) Early baseline skill by catchment role")


for ext in ("png", "tif", "svg", "pdf"):
    opts = {"pil_kwargs":{"compression":"tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(HERE / f"fig2.{ext}", dpi=600, **opts)
plt.close(fig)
qa = HERE / "qa"
qa.mkdir(exist_ok=True)
with Image.open(HERE / "fig2.png") as image:
    image.thumbnail((1300,1000))
    image.save(qa / "Figure_2_preview.png")
