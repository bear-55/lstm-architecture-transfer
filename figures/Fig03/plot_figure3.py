"""Render Figure 3 in the supplied v5 style using verified real data only."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
QA = HERE / "qa"
QA.mkdir(exist_ok=True)
DPI, MM = 600, 1 / 25.4
INK, GREY = "#1A1A1A", "#7F7F7F"
PAIR = {"M1/B0": "#D62728", "EA/B0": "#1F5FA8"}
CMAP = LinearSegmentedColormap.from_list("historical_gain", ["#2C5DA6", "#8FB3D9", "#FFF2BE", "#F28E5C", "#D62728"])
NORM = TwoSlopeNorm(vmin=-.10, vcenter=0, vmax=.10)
DISPLAY_LIMIT = .22
STAGES = ["historical", "late_retained", "late_retrained"]
FONT = font_manager.FontProperties(family=["Times New Roman", "Liberation Serif", "DejaVu Serif"]).get_name()
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.1, "axes.titleweight": "bold",
    "axes.labelsize": 8, "axes.labelweight": "bold", "axes.edgecolor": INK,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.linewidth": .8,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
    "ytick.major.size": 3, "xtick.major.width": .8, "ytick.major.width": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": DPI, "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.unicode_minus": True,
})


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rho_label(value):
    return f"{value:.2f}".replace("-", "−")


def main():
    audit = json.loads((HERE / "data_verification.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    for name, digest in audit["data_hashes"].items():
        assert sha(DATA / name) == digest
    traj = pd.read_csv(DATA / "trajectories.csv", dtype={"basin_id": str})
    quintiles = pd.read_csv(DATA / "quintile_summary.csv")
    correlations = pd.read_csv(DATA / "correlation_summary.csv").set_index("panel")
    meta = {r["panel"]: r for r in audit["panels"]}
    fig, axs = plt.subplots(2, 3, figsize=(190 * MM, 133 * MM),
                            gridspec_kw=dict(left=.100, right=.874, top=.9167118644, bottom=.145,
                                             wspace=.43, hspace=.60))
    panel_axes = {"a": axs[0, 0], "b": axs[0, 1], "c": axs[0, 2],
                  "d": axs[1, 0], "e": axs[1, 1], "f": axs[1, 2]}
    artists_count = {}
    for panel in ["a", "b", "d", "e", "f"]:
        ax, info, corr = panel_axes[panel], meta[panel], correlations.loc[panel]
        sample = traj[(traj.panel == panel) & (traj.source_seed == info["example_seed"])].sort_values("basin_id")
        assert len(sample) == info["n_catchments"]
        values = sample[STAGES].to_numpy()
        segments = np.stack([np.tile(np.arange(3), (len(values), 1)), values], axis=2)
        # Every catchment remains in the plot. Keep each line's historical
        # colour across all three axes; adjust alpha only for panel density.
        opacity = .18 * (107 / len(values)) ** .65
        colours = CMAP(NORM(values[:, 0]))
        colours[:, 3] = opacity
        paint_order = np.random.default_rng(903).permutation(len(values))
        background = LineCollection(segments[paint_order], colors=colours[paint_order],
                                    linewidths=.34, zorder=1)
        ax.add_collection(background)
        # Individual trajectories are kept, with explicit edge marks where
        # their values exceed the shared display; all foreground medians fit.
        for stage_i in range(3):
            if np.any(values[:, stage_i] > DISPLAY_LIMIT):
                ax.plot(stage_i, DISPLAY_LIMIT - .0035, marker="^", ms=2.8, color="#A0A0A0", mec="white", mew=.25, ls="none", zorder=5)
            if np.any(values[:, stage_i] < -DISPLAY_LIMIT):
                ax.plot(stage_i, -DISPLAY_LIMIT + .0035, marker="v", ms=2.8, color="#A0A0A0", mec="white", mew=.25, ls="none", zorder=5)
        for q in range(1, 6):
            s = quintiles[(quintiles.panel == panel) & (quintiles.quintile == q)].set_index("stage").loc[STAGES]
            med = s.mean_median.to_numpy()
            assert np.abs(med).max() < DISPLAY_LIMIT
            colour = CMAP(NORM(med[0]))
            line, = ax.plot(np.arange(3), med, color=colour, lw=1.25, zorder=4)
            line.set_path_effects([pe.Stroke(linewidth=2.1, foreground="white", alpha=.85), pe.Normal()])
            ax.plot(np.arange(3), med, "o", color=colour, ms=2.5, mec="#555555", mew=.35, zorder=5)
        for xx in [0, 1, 2]:
            ax.axvline(xx, color="#777777", lw=.6, zorder=2)
        ax.axhline(0, color="#9A9A9A", lw=.55, ls="--", zorder=1)
        ax.set_xlim(-.16, 2.16)
        ax.set_ylim(-DISPLAY_LIMIT, DISPLAY_LIMIT)
        ax.set_yticks([-.20, -.10, 0, .10, .20])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.set_xticks([0, 1, 2], ["Historical", f"Late retained\nρ = {rho_label(corr.rho_retained)}", f"Late independent\nρ = {rho_label(corr.rho_retrained)}"])
        ax.tick_params(axis="x", pad=3)
        for lab in ax.get_xticklabels():
            lab.set_fontweight("bold")
            lab.set_fontsize(6.1)
            lab.set_linespacing(1.08)
        if panel in ["a", "d"]:
            ax.set_ylabel("ΔNSE* (candidate − B0)", labelpad=3)
        title = f"({panel}) {info['comparison']} · {info['group']} (n={info['n_catchments']})"
        ax.set_title(title, fontsize=7.4, pad=5)
        artists_count[panel] = dict(background_catchment_lines=len(values), foreground_quintiles=5,
                                   quartile_bars=0, individual_line_opacity=float(opacity), example_seed=info["example_seed"])

    # Variance shares use both scales and the full period × seed × catchment decomposition.
    ax=panel_axes['c']; pos=ax.get_position(); ax.remove()
    variance=pd.read_csv(DATA/'variance_shares.csv')
    from matplotlib.patches import Patch
    components=['Catchment','Seed×Catchment','Period×Seed×Catchment','Other']
    colours=['#1F5FA8','#D62728','#F3B2B0','#CFCFCF']
    groups=[('Held-out','M1/B0'),('Held-out','EA/B0'),('Training','M1/B0'),('Training','EA/B0')]
    for j,metric in enumerate(['NSE_star','NSE']):
        aa=fig.add_axes([pos.x0+.009,pos.y0+(.54 if j==0 else .02)*pos.height,pos.width-.009,pos.height*.39])
        for yi,(group,comp) in enumerate(groups):
            row=variance[(variance.group==group)&(variance.comparison==comp)&(variance.metric==metric)].set_index('component').share_pct
            vals=[row[c] for c in components[:3]]; vals+=[100-sum(vals)]
            left=0
            for ii,(v,col) in enumerate(zip(vals,colours)):
                aa.barh(3-yi,v,left=left,height=.68,color=col,edgecolor='white',linewidth=.25,hatch='////' if ii==2 else None)
                if v>=11: aa.text(left+v/2,3-yi,f'{v:.0f}',ha='center',va='center',fontsize=5.5,color=INK if ii>1 else 'white',fontweight='bold')
                left+=v
        aa.set_xlim(0,100); aa.set_ylim(-.6,3.6); aa.set_xticks([0,50,100]); aa.tick_params(length=2,pad=1,labelsize=5.7)
        aa.set_yticks([3,2,1,0],['M1 held-out','EA held-out','M1 training','EA training'],fontsize=5.2)
        aa.set_title(('(c) Variance shares · NSE*' if j==0 else 'NSE'),fontsize=6.6,pad=4)
        if j==0: aa.tick_params(labelbottom=False)
        if j==1: aa.set_xlabel('Share (%)',fontsize=6,labelpad=1)
    fig.legend([Patch(facecolor=c,edgecolor='white',hatch='////' if i==2 else None) for i,c in enumerate(colours)],
               ['Catchment','Seed × catchment','Period × seed × catchment','Other'],
               loc='lower center',bbox_to_anchor=(.52,.008),ncol=4,fontsize=6.3,columnspacing=1.2,handlelength=1.4)

    for row, name in enumerate(["M1/B0", "EA/B0"]):
        box = axs[row, 0].get_position()
        fig.text(.018, (box.y0 + box.y1) / 2, name, rotation=90,
                 ha="center", va="center", fontsize=9, fontweight="bold")
    cax = fig.add_axes([.914, .2151694915, .014, .5659322034])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, extend="both",
                      ticks=[-.10, -.05, 0, .05, .10])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    cb.ax.tick_params(labelsize=6.4, length=2, pad=2)
    cb.set_label("Historical ΔNSE* (line colour)", fontsize=7, fontweight="bold", labelpad=4)
    cb.solids.set_rasterized(False)

    handles = [Line2D([], [], color="#8496AE", lw=.45, alpha=.55),
               Line2D([], [], color="#8496AE", lw=1.3, marker="o", ms=2.7, mec="#555555", mew=.35)]
    fig.legend(handles, ["Individual catchment", "Historical-quintile median (mean over seeds)"],
               loc="lower center", bbox_to_anchor=(.49, .047), ncol=2, handlelength=2.5,
               columnspacing=2, fontsize=6.5)

    import sys
    sys.path.insert(0,str(HERE.parent))
    from plot_v6_figures import finish
    finish(fig,3,dict(trajectories=artists_count,variance_scales=['NSE_star','NSE']))

if __name__=='__main__': main()
