"""Four-panel Figure 8 from the verified local point estimates and resampling draws."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
COLOR = {"M1": "#D62728", "EA": "#1F5FA8"}
LIGHT = {"M1": "#F2A7A3", "EA": "#9DB9DE"}
GOLD, GREY, INK = "#E8A317", "#7F7F7F", "#1A1A1A"
MARK = {"M1": "o", "EA": "s"}
DASH = (0, (4, 3))
GROUPS = (
    ("Main estimate", (("main", "Seed + HUC2 bootstrap"), ("seed_only", "Seed-only bootstrap"))),
    ("Catchment influence", (("median", "Catchment median"), ("omit3", "Omit 3 largest losses"),
                             ("omit7", "Omit 7 largest losses"), ("omitlib", "Omit 7 largest library gains"))),
    ("Selection period", (("prespec", "WY1990–1999 (alternative)"), ("windows", "Seven four-year windows"))),
    ("Seed pairing", (("perm", "Permuted B0 seed labels"),)),
    ("Selection budget", (("k5", "Five source instances"),)),
    ("Evaluation setting", (("train_mean", "Training catchments: mean"), ("train_median", "Training catchments: median"),
                            ("ext_nse", "External archive: NSE loss"), ("ext_mse", "External archive: MSE loss"))),
)
SHORT = {"seed_only": "Seed-only", "median": "Median", "omit3": "Omit 3", "omit7": "Omit 7",
         "omitlib": "Omit lib.", "prespec": "WY1990–99", "windows": "Windows", "k5": "k = 5",
         "train_mean": "Train mean", "train_median": "Train med."}


def grid(ax, axis):
    ax.grid(axis=axis, color="#B5B5B5", linewidth=.6, linestyle=DASH)
    ax.set_axisbelow(True)


def number(value):
    precision = 4 if abs(value) >= .001 else 5
    return f"{value:.{precision}f}".replace("-", "−")


def forest(ax, table, model, letter, values, distributions):
    headers, rows, y = [], [], 0.
    for group, entries in GROUPS:
        headers.append((group, y))
        y += .8
        for key, label in entries:
            rows.append((key, label, y))
            y += 1.
        y += .35
    for j, (_, start) in enumerate(headers):
        stop = headers[j+1][1] if j+1 < len(headers) else y
        if j % 2 == 0:
            for a in (ax, table):
                a.axhspan(start-.35, stop-.35, color="#F4F4F4", zorder=0, linewidth=0)

    main = values.at["main", "estimate"]
    ax.axvline(0, color=INK, linewidth=.8, zorder=1)
    ax.axvline(main, color=COLOR[model], linewidth=.8, linestyle=DASH, alpha=.75, zorder=1)
    for key, _, pos in rows:
        if key not in values.index:
            table.text(.04, pos, "—", va="center", fontsize=6.4, color=GREY)
            continue
        row = values.loc[key]
        samples = distributions[f"{model}__{key}"]
        if key == "windows":
            ax.plot([row.low, row.high], [pos, pos], color=LIGHT[model], linewidth=5,
                    solid_capstyle="round", zorder=2)
            ax.scatter(samples, np.full(len(samples), pos), s=10, color=COLOR[model],
                       edgecolors="white", linewidth=.4, zorder=3)
            table.text(.04, pos, f"{number(row.low)}–{number(row.high)}", va="center", fontsize=6.4)
            continue
        external = key.startswith("ext_")
        if not external:
            xs = np.linspace(*np.quantile(samples, [.005, .995]), 201)
            density = gaussian_kde(samples)(xs)
            crest = pos - .62 * density / density.max()
            ax.fill_between(xs, pos, crest, color=LIGHT[model], alpha=.75, linewidth=0, zorder=2)
            ax.plot(xs, crest, color=COLOR[model], linewidth=.5, zorder=2)
        ax.plot([row.low, row.high], [pos, pos], color=COLOR[model], linewidth=1.3,
                solid_capstyle="butt", zorder=3)
        marker = "*" if key == "main" else "D" if key == "perm" else "s" if external else "o"
        face = GOLD if key == "main" else "white" if external or key == "perm" else COLOR[model]
        ax.plot(row.estimate, pos, marker=marker, linestyle="none", markersize=9 if key == "main" else 4.6,
                markerfacecolor=face, markeredgecolor=COLOR[model] if key == "perm" else INK,
                markeredgewidth=1.0 if key == "perm" else .6, zorder=4)
        text = ("range" if key == "perm" else number(row.estimate)) + f" [{number(row.low)}, {number(row.high)}]"
        table.text(.04, pos, text, va="center", fontsize=6.4, fontweight="bold" if row.low > 0 else "normal",
                   color=INK if row.low > 0 else GREY)

    ax.set_ylim(y-.2, -.6)
    table.set_ylim(y-.2, -.6)
    # Use the same scale for both contrasts and include all the plotted density tails.
    ax.set_xlim(-.043, .20)
    ax.set_xticks([0, .05, .10, .15, .20])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    grid(ax, "x")
    ax.set_xlabel(r"Retraining loss $D$ (NSE)")
    ax.set_title(f"({letter}) {model}/B0", fontsize=9, fontweight="bold", pad=22, color=COLOR[model])
    secondary = ax.secondary_xaxis("top", functions=(lambda x: x / main * 100, lambda p: p / 100 * main))
    secondary.set_xticks([0, 50, 100, 150, 200])
    secondary.tick_params(labelsize=6, length=2, pad=1)
    secondary.set_xlabel("% of main estimate", fontsize=6.2, labelpad=2)
    table.axis("off")
    table.set_xlim(0, 1)
    table.text(.04, -.45, "Estimate [95% interval]", fontsize=6.4, fontweight="bold", style="italic", va="center")
    if letter == "a":
        ax.set_yticks([pos for _, _, pos in rows], [label for _, label, _ in rows], fontsize=6.8)
        for name, pos in headers:
            ax.text(-.035, pos+.05, name, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=7.2, fontweight="bold", style="italic")
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)


def matching(ax, estimates):
    schemes = ("unmatched", "across_HUC2", "within_HUC2")
    grid(ax, "x")
    for model, offset in (("M1", -.17), ("EA", .17)):
        results = estimates.loc[model].reindex(schemes)
        for y, row in enumerate(results.itertuples()):
            ax.plot([row.low, row.high], [y+offset, y+offset], color=COLOR[model], linewidth=1.3)
            ax.plot(row.point, y+offset, MARK[model], markersize=4.6, color=COLOR[model],
                    markeredgecolor=INK, markeredgewidth=.5, zorder=4)
        first, last = results.at["unmatched", "point"], results.at["within_HUC2", "point"]
        ax.annotate("", xy=(last, 2+offset), xytext=(first, offset),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR[model], linewidth=.8, linestyle=(0,(2,2)),
                                    alpha=.8, connectionstyle="arc3,rad=0.25"))
        change = f"{(last/first-1)*100:+.0f}%".replace("-", "−")
        ax.text(.0315, 2+offset, change, va="center", color=COLOR[model], fontsize=6.6, fontweight="bold")
    count = estimates.loc["M1"].n_heldout
    labels = ["Unmatched", f"Across HUC2 ({count['across_HUC2']} pairs)", f"Within HUC2 ({count['within_HUC2']} pairs)"]
    ax.axvline(0, color=INK, linewidth=.8)
    ax.set_yticks(range(3), labels, fontsize=6.8)
    ax.set_ylim(2.55, -.55)
    ax.set_xlim(-.01, .042)
    ax.set_xlabel(r"Held-out − training $D$ (NSE*)")
    ax.set_title("(c) Held-out minus training, by matching", fontsize=8.5, fontweight="bold", pad=4)


def retained_shares(ax, shares):
    grid(ax, "y")
    for i, key in enumerate(SHORT):
        for model, offset in (("M1", -.18), ("EA", .18)):
            row = shares.loc[(model, key)]
            x = i+offset
            color = COLOR[model]
            ax.plot([x, x], [0, row.percent], color=color, linewidth=.9 if row.bounds_include_zero else 1.6,
                    alpha=.5 if row.bounds_include_zero else 1)
            ax.plot(x, row.percent, MARK[model], markersize=4, markeredgecolor=color, markeredgewidth=.8,
                    markerfacecolor="white" if row.bounds_include_zero else color)
    ax.axhline(100, color=INK, linewidth=.7, linestyle=DASH)
    ax.axhline(0, color=INK, linewidth=.8)
    ax.set_xticks(range(len(SHORT)), list(SHORT.values()), rotation=35, ha="right", fontsize=6.4)
    ax.set_ylim(0,125)
    ax.set_ylabel("% of main estimate", fontsize=7)
    ax.set_title("(d) Share of the main estimate retained", fontsize=8.5, fontweight="bold", pad=4)


def render_figure8(directory, finish):
    import json
    record = json.loads((directory / "enhanced_data_verification.json").read_text(encoding="utf-8"))
    if record["status"] != "PASS":
        raise ValueError("Figure 8 requires verified local data")
    endpoints = pd.DataFrame(record["upper_panel"]).set_index(["model", "key"])
    matched = pd.DataFrame(record["matching"]).set_index(["model", "scheme"])
    shares = pd.read_csv(directory/"data/retained_shares.csv", float_precision="round_trip").set_index(["model", "check"])
    style = {"font.serif": ["Times New Roman", "Liberation Serif"], "font.family": "serif",
             "mathtext.fontset": "stix", "axes.labelweight": "bold", "axes.titleweight": "bold",
             "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": .8, "ytick.major.width": .8,
             "axes.linewidth": .8, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42}
    with plt.rc_context(style), np.load(directory/"data/forest_distributions.npz", allow_pickle=False) as draws:
        fig = plt.figure(figsize=(190/25.4, 168/25.4))
        top = fig.add_gridspec(1,4, width_ratios=[1,.52,1,.52], wspace=0, left=.205, right=.995, top=.865, bottom=.355)
        lower = fig.add_gridspec(1,2, wspace=.30, left=.205, right=.995, top=.245, bottom=.07)
        ax_a = fig.add_subplot(top[0,0])
        ax_at = fig.add_subplot(top[0,1], sharey=ax_a)
        ax_b = fig.add_subplot(top[0,2], sharey=ax_a)
        ax_bt = fig.add_subplot(top[0,3], sharey=ax_a)
        forest(ax_a, ax_at, "M1", "a", endpoints.loc["M1"], draws)
        forest(ax_b, ax_bt, "EA", "b", endpoints.loc["EA"], draws)
        matching(fig.add_subplot(lower[0,0]), matched)
        retained_shares(fig.add_subplot(lower[0,1]), shares)
        handles = [
            Line2D([],[], marker="*", linestyle="none", color=GOLD, markeredgecolor=INK, markersize=9),
            Line2D([],[], marker="o", linestyle="none", color=GREY, markeredgecolor=INK, markersize=4.5),
            Patch(facecolor="#DDDDDD", edgecolor=GREY, linewidth=.5),
            Line2D([],[], color=GREY, linewidth=5, alpha=.6),
            Line2D([],[], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=GREY, markersize=4.5),
            Line2D([],[], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=GREY, markersize=4.5),
            Line2D([],[], color=GREY, linewidth=.8, linestyle=DASH),
        ]
        labels = ["Main estimate", "Estimate, 95% interval", "Bootstrap distribution", "Seven windows",
                  "Observed pairing (permutation range)", "External archive", "Main-estimate level"]
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.998), ncol=4,
                   handlelength=1.6, columnspacing=1.6, prop={"weight":"bold","size":7.5})
        finish(fig, 8, dict(revision="Enhanced four-panel forest, 2026-11-06",
                            windows="Seven observed point estimates and their range (a,b); arithmetic mean (d).",
                            permutation="Diamond is the observed pairing, interval is the central 95% of permutations.",
                            local_data_checks="enhanced_data_verification.json"))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    from plot_v6_figures import finish
    render_figure8(HERE, finish)
