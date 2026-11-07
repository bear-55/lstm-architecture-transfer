"""Draw verified multi-metric catchment distributions in the supplied Fig. 7 style."""
from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image
import scipy
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
DATA, QA = HERE / "data", HERE / "qa"
WIDTH_MM, HEIGHT_MM, DPI = 190, 120, 600
INK = "#1A1A1A"
COLOURS = {"M1/B0": ("#D62728", "#F2A7A3"), "EA/B0": ("#1F5FA8", "#9DB9DE")}
PANELS = [
    ("NSE_star", "NSE*", "Gain (ΔNSE*)"),
    ("KGE", "KGE", "Gain (ΔKGE)"),
    ("abs_FHV", "|FHV|", "Reduction in |FHV| (pp)"),
    ("nMAE_high", r"nMAE$_{high}$", r"Reduction in nMAE$_{high}$"),
    ("nMAE_low", r"nMAE$_{low}$", r"Reduction in nMAE$_{low}$"),
]
FONT = font_manager.FontProperties(family=["Times New Roman", "Liberation Serif", "DejaVu Serif"]).get_name()
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.2, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.frameon": False,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.edgecolor": INK,
    "axes.linewidth": .8, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": .8, "ytick.major.width": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": True,
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": DPI,
})


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    QA.mkdir(exist_ok=True)
    audit = json.loads((HERE / "data_verification.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    for filename, digest in audit["data_hashes"].items():
        assert sha(DATA / filename) == digest
    values = pd.read_csv(DATA / "per_catchment_gains.csv", dtype={"basin_id": str, "huc2": str}, float_precision="round_trip")
    summaries = pd.read_csv(DATA / "metric_summary.csv", float_precision="round_trip").set_index(["comparison", "metric"])
    ranges = pd.read_csv(DATA / "distribution_statistics.csv", float_precision="round_trip")
    fig=plt.figure(figsize=(190/25.4,110/25.4))
    rects=[[.085,.565,.245,.325],[.410,.565,.245,.325],[.735,.565,.245,.325],
           [.245,.105,.245,.325],[.575,.105,.245,.325]]
    axs=np.array([fig.add_axes(r) for r in rects])
    density_rows, half_metadata, loss_annotations, bar_labels = [], [], [], []

    def half_violin(ax, data, x, side, colour, comparison, metric, endpoint):
        assert len(data) == 107 and np.isfinite(data).all()
        kde = gaussian_kde(data, bw_method="scott")
        lo, hi = np.quantile(data, [.01, .99])
        yy = np.linspace(lo, hi, 300)
        density = kde(yy)
        width = density / density.max() * .38
        xx = x + (-width if side == "left" else width)
        ax.fill_betweenx(yy, x, xx, facecolor=colour, edgecolor=INK, lw=.65, zorder=3)
        q25, med, q75 = np.quantile(data, [.25, .5, .75])
        offset = -.07 if side == "left" else .07
        ax.plot([x + offset] * 2, [q25, q75], color=INK, lw=1.6, solid_capstyle="butt", zorder=4)
        ax.plot(x + offset, med, "o", ms=3.5, mfc="white", mec=INK, mew=.75, zorder=5)
        half_metadata.append(dict(comparison=comparison, metric=metric, endpoint=endpoint, side=side,
                                  n_catchments=107, q01=float(lo), q99=float(hi),
                                  q25=float(q25), median=float(med), q75=float(q75),
                                  bandwidth_factor=float(kde.factor)))
        for y, d, w in zip(yy, density, width):
            density_rows.append(dict(comparison=comparison, metric=metric, endpoint=endpoint,
                                     y=y, density=d, half_width=w))

    for pi, (metric, title, ylabel) in enumerate(PANELS):
        ax = axs.flat[pi]
        # Include the complete displayed 1st--99th percentile range for every
        # half; pooled percentile limits in the template can clip a second tail.
        rr = ranges[ranges.metric == metric]
        low, high = float(rr.q01.min()), float(rr.q99.max())
        span = high - low
        ax.set_ylim(low - .06 * span, high + .23 * span)
        ax.set_xlim(-.6, 1.6)
        ax.set_xticks([0, 1], ["M1/B0", "EA/B0"], fontweight="bold")
        ax.set_title(f"({'abcde'[pi]}) {title}", pad=6)
        ax.set_ylabel(ylabel, labelpad=5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(axis="y", color="#B5B5B5", lw=.6, ls=(0, (4, 3)), zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK, lw=.9, zorder=2)
        for xi, (comparison, colours) in enumerate(COLOURS.items()):
            part = values[(values.metric == metric) & (values.comparison == comparison)]
            for endpoint, side, colour in zip(["G_same", "G_new"], ["left", "right"], colours):
                half_violin(ax, part[endpoint].to_numpy(), xi, side, colour, comparison, metric, endpoint)
            summary = summaries.loc[(comparison, metric)]
            label = f"mean $D$ = {summary.D_mean:.3g}\n$D_b > 0$: {summary.share_D_positive:.0%}"
            item = ax.text(.245 if xi == 0 else .755, .979, label, transform=ax.transAxes,
                           ha="center", va="top", fontsize=6.5, linespacing=1.1,
                           bbox=dict(facecolor="white", edgecolor="none", pad=.8), zorder=6)
            loss_annotations.append(item)

    legend = fig.legend([Patch(facecolor="#808080", edgecolor=INK, lw=.65),
                         Patch(facecolor="#D0D0D0", edgecolor=INK, lw=.65)],
                        [r"Left: retained weights ($G_{same,b}$)", r"Right: new weights ($G_{new,b}$)"],
                        loc="upper center", bbox_to_anchor=(.54, .997), ncol=2,
                        fontsize=7.5, handlelength=1.7, columnspacing=2.5, handletextpad=.6)

    import sys
    sys.path.insert(0,str(HERE.parent))
    from plot_v6_figures import finish
    pd.DataFrame(density_rows).to_csv(DATA/'density_coordinates.csv',index=False)
    finish(fig,6,dict(distributions=half_metadata,positive_loss_labels=True))

if __name__=='__main__': main()
