"""Figure 2: real-data maps, adapted from the user-supplied v5 template.

Run with Python + numpy, pandas, matplotlib and Pillow. All map and experiment
inputs are local. This script does not change the manuscript or raw results.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
QA = HERE / "qa"
QA.mkdir(exist_ok=True)
MM = 1 / 25.4
DPI = 600
INK, BLUE = "#1A1A1A", "#1F5FA8"
FONT = font_manager.FontProperties(family=["Times New Roman", "Liberation Serif", "DejaVu Serif"]).get_name()
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.titlesize": 8.5, "axes.titleweight": "bold",
    "axes.labelsize": 8, "axes.labelweight": "bold",
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": .8, "axes.edgecolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": .8, "ytick.major.width": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "savefig.dpi": DPI, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.unicode_minus": True,
})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_parts(feature):
    g = feature["geometry"]
    assert g and g["type"] in {"Polygon", "MultiPolygon"}
    return [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]


def make_path(features):
    vertices, codes = [], []
    for feat in features:
        for polygon in polygon_parts(feat):
            for ri, ring in enumerate(polygon):
                a = np.asarray(ring, dtype=float)[:, :2]
                assert len(a) >= 4 and np.isfinite(a).all()
                if not np.array_equal(a[0], a[-1]):
                    a = np.vstack([a, a[0]])
                signed_area = np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1])
                # Consistent winding preserves holes in PDF, SVG and Agg.
                if (ri == 0 and signed_area < 0) or (ri > 0 and signed_area > 0):
                    a = a[::-1]
                vertices.extend(a)
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(a) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(np.asarray(vertices), codes)


def size_for_loss(values):
    return 8 + 250 * np.minimum(np.abs(values), .15)


def add_map(ax, state_features, huc_features):
    land_path = make_path(state_features)
    # A true cartographic boundary replaces the hand-drawn template outline.
    land = PathPatch(land_path, facecolor="#F3F3F0", edgecolor="#BEBEBA", linewidth=.34, zorder=0)
    ax.add_patch(land)
    lines = [np.asarray(ring)[:, :2] for f in huc_features for p in polygon_parts(f) for ring in p]
    regions = LineCollection(lines, colors="#8C8C88", linewidths=.38,
                             linestyles=(0, (2.4, 1.7)), zorder=1)
    regions.set_clip_path(land_path, ax.transData)
    ax.add_collection(regions)
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50.5)
    # Preserve the template's geographic display/aspect; no distance/area
    # calculations are performed in longitude/latitude coordinates.
    ax.set_aspect(1.25)
    ax.set_xticks([-120, -105, -90, -75], ["120°W", "105°W", "90°W", "75°W"])
    ax.set_yticks([30, 40, 50], ["30°N", "40°N", "50°N"])
    ax.tick_params(pad=2)


def title(ax, label, text):
    ax.set_title(f"({label}) {text}", loc="center", pad=5)


def main():
    audit = json.loads((HERE / "data_verification.json").read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert sha(DATA / "figure2_catchments.csv") == audit["data_file_sha256"]
    frame = pd.read_csv(DATA / "figure2_catchments.csv", dtype={"basin_id": str, "huc2": str})
    assert len(frame) == 527 and frame.basin_id.nunique() == 527
    states = json.loads((DATA / "census_states_5m_2024.geojson").read_text(encoding="utf-8"))["features"]
    hucs = json.loads((DATA / "usgs_huc2_regions.geojson").read_text(encoding="utf-8"))["features"]
    assert len(states) == 49 and len(hucs) == 18
    assert len({f["properties"]["STATE"] for f in states}) == 49
    assert {f["properties"]["huc2"] for f in hucs} == {f"{i:02d}" for i in range(1, 19)}

    fig, axs = plt.subplots(2, 2, figsize=(190 * MM, 120 * MM),
                           gridspec_kw=dict(left=.06, right=.905, top=.95, bottom=.125,
                                            wspace=.14, hspace=.29))
    for ax in axs.flat:
        add_map(ax, states, hucs)

    # (a) Original role counts and positions, not generated sample locations.
    ax = axs[0, 0]
    style = {"train": ("Training", "^", "#9E9E9E", 7),
             "val": ("Validation", "D", "white", 7),
             "test": ("Held-out", "o", INK, 10)}
    for role, (label, marker, colour, size) in style.items():
        d = frame[frame.role == role]
        ax.scatter(d.gauge_lon, d.gauge_lat, marker=marker, s=size, c=colour,
                   edgecolors=INK if role != "train" else "none", linewidths=.4,
                   label=f"{label} ({len(d)})", zorder=3)
    ax.legend(loc="lower left", bbox_to_anchor=(.015, .025), borderpad=.1,
              labelspacing=.35, handletextpad=.3, prop={"weight": "bold", "size": 6.5})
    title(ax, "a", "Catchment roles")

    # (b) Full bounded range: negative skills are not clipped to zero.
    ax = axs[0, 1]
    sc = ax.scatter(frame.gauge_lon, frame.gauge_lat,
                    c=frame.early_B0_transformed_mean_NSE, cmap="YlOrRd_r",
                    norm=Normalize(-1, 1), s=9, edgecolors="none", zorder=3)
    risk = frame[frame.early_risk]
    assert len(risk) == 7 and (risk.early_EA_B0_pooled_NSE < 0).all()
    ax.scatter(risk.gauge_lon, risk.gauge_lat, s=36, facecolors="none",
               edgecolors=BLUE, linewidths=1, zorder=5, label="Early risk catchments (7)")
    ax.legend(loc="lower left", bbox_to_anchor=(.025, .015), borderpad=.15,
              handletextpad=.6, prop={"weight": "bold", "size": 6.5})
    cax = ax.inset_axes([1.025, .1, .031, .8])
    cb = fig.colorbar(sc, cax=cax, extend="neither", ticks=[-1, -.5, 0, .5, 1])
    cb.ax.tick_params(labelsize=6.7, length=2, pad=2)
    cb.set_label("Early B0 NSE* (transformed mean)", fontsize=7, fontweight="bold", labelpad=3)
    cb.solids.set_rasterized(False)
    title(ax, "b", "Early-period skill of B0")

    heldout = frame[frame.role == "test"]
    other = frame[frame.role != "test"]
    # Keep the reference's common colour normalization and size scale in c/d.
    norm = TwoSlopeNorm(vmin=-.06, vcenter=0, vmax=.12)
    for j, model in enumerate(["M1", "EA"]):
        ax = axs[1, j]
        ax.scatter(other.gauge_lon, other.gauge_lat, s=3, c="#CFCFCF", edgecolors="none", zorder=2)
        loss = heldout[f"D_{model}_NSEstar"]
        # Largest symbols first, allowing small nearby catchments to remain visible.
        d = heldout.assign(size=size_for_loss(loss)).sort_values("size", ascending=False)
        sc = ax.scatter(d.gauge_lon, d.gauge_lat, c=d[f"D_{model}_NSEstar"],
                        cmap="RdBu_r", norm=norm, s=d["size"], edgecolors=INK,
                        linewidths=.3, zorder=3)
        top = heldout[heldout.top3_library_gain]
        assert len(top) == 3
        ax.scatter(top.gauge_lon, top.gauge_lat, s=90, facecolors="none", edgecolors=INK,
                   linewidths=1.1, zorder=5)
        stats = audit["loss_statistics"][model]
        ax.text(.98, .035, f"positive: {stats['positive_percent']:.0f}%\nmedian $D_b$ = {stats['median']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, fontweight="bold")
        title(ax, "cd"[j], f"Retraining loss $D_b$ (NSE*), {model}/B0")
    cax = axs[1, 1].inset_axes([1.025, .1, .031, .8])
    loss_ticks = [-.06, -.03, 0, .03, .06, .09, .12]
    cb = fig.colorbar(sc, cax=cax, extend="neither", ticks=loss_ticks)
    # Flat ends match panel b. The upper tick also identifies observations above the colour limit.
    cb.set_ticks(loss_ticks, labels=["−0.06", "−0.03", "0.00", "0.03", "0.06", "0.09", "≥0.12"])
    cb.ax.tick_params(labelsize=6.7, length=2, pad=2)
    cb.set_label("$D_b$ (NSE*)", fontsize=7, fontweight="bold", labelpad=3)
    cb.solids.set_rasterized(False)

    # Common keys replace the template/proxy footnote.
    common = [Line2D([], [], marker="o", ls="none", ms=2.5, color="#CFCFCF"),
              Line2D([], [], marker="o", ls="none", ms=6, mfc="none", mec=INK, mew=1.0),
              Line2D([], [], color="#BEBEBA", lw=.6),
              Line2D([], [], color="#8C8C88", lw=.6, ls=(0, (2.4, 1.7)))]
    fig.legend(common, ["Other catchments (c, d)", "Largest library gains (3)", "State boundary", "HUC2 boundary"],
               loc="lower center", bbox_to_anchor=(.49, .052), ncol=4,
               handlelength=1.6, handletextpad=.45, columnspacing=1.3, fontsize=6.6)
    sizes = [Line2D([], [], marker="o", ls="none", ms=np.sqrt(size_for_loss(v)),
                    mfc="#EFEFEF", mec=INK, mew=.4) for v in [.02, .08, .15]]
    fig.legend(sizes, ["0.02", "0.08", "≥0.15"], loc="lower center", bbox_to_anchor=(.60, .011),
               ncol=3, fontsize=6.6, handlelength=1.6, handletextpad=.5, columnspacing=2)
    fig.text(.18, .027, "Marker size in (c, d): $|D_b|$", ha="left", va="center", fontsize=6.6)

    fig.canvas.draw()
    # Save a small QC preview independently of the four publication formats.
    fig.savefig(QA / "Figure_2_preview.png", dpi=180)
    basename = HERE / "fig2"
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(basename.with_suffix("." + ext), dpi=DPI)
    # TIFF stores RGB pixels and explicit 600-dpi metadata, with lossless LZW.
    with Image.open(basename.with_suffix(".png")) as im:
        im.convert("RGB").save(basename.with_suffix(".tif"), dpi=(DPI, DPI), compression="tiff_lzw")

    renderer = fig.canvas.get_renderer()
    text_outside = []
    from matplotlib.text import Text
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.width > 0 and box.height > 0 and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
                text_outside.append(artist.get_text())
    assert not text_outside, text_outside
    manifest = dict(figure="2", template="figure_templates_v5_all.zip / figures_v5.py: fig02",
                    width_mm=190, height_mm=120, raster_dpi=DPI, font=FONT,
                    map_crs="EPSG:4326", display="longitude/latitude axes; aspect 1.25 from supplied template",
                    source_data_sha256=sha(DATA / "figure2_catchments.csv"),
                    colour_range_B0=[-1, 1], colour_range_D=[-.06, .12], colour_center_D=0,
                    colour_extension_B0="neither", colour_extension_D="neither", upper_tick_D="≥0.12",
                    marker_area_pt2="8 + 250 * min(abs(D_b), 0.15)",
                    risk_definition="EA/B0 pooled-ensemble NSE < 0 during WY2000-2003",
                    corrected_template_proxy=True, no_synthetic_data=True,
                    outside_canvas_text=text_outside,
                    outputs={ext: dict(filename=basename.with_suffix('.' + ext).name,
                                      bytes=basename.with_suffix('.' + ext).stat().st_size,
                                      sha256=sha(basename.with_suffix('.' + ext))) for ext in ['png', 'tif', 'svg', 'pdf']},
                    versions=dict(python=__import__('sys').version, matplotlib=matplotlib.__version__,
                                  numpy=np.__version__, pandas=pd.__version__))
    (HERE / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    plt.close(fig)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
