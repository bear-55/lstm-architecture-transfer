"""Catchment-level gain decomposition for the six supplementary examples."""
from hashlib import sha256
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
DATA,QA=HERE/"data",HERE/"qa"
WIDTH_MM,HEIGHT_MM,DPI=190,245,600
INK,RED,BLUE="#1A1A1A","#D62728","#1F5FA8"
COMPARISONS=["M1/B0","EA/B0"]
COLOURS={"M1/B0":RED,"EA/B0":BLUE}
METRICS=["NSE_star","KGE","nMAE_high"]
HEADINGS=["NSE*","KGE",r"nMAE$_{high}$"]
ENDPOINTS=["G_same","G_new","D_b"]
FONT=font_manager.FontProperties(family=["Times New Roman", "Liberation Serif", "DejaVu Serif"]).get_name()
plt.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman", "Liberation Serif", "DejaVu Serif"],"mathtext.fontset":"stix",
    "font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,"xtick.labelsize":7,"ytick.labelsize":6.4,
    "legend.frameon":False,"axes.labelweight":"bold","axes.titleweight":"bold","axes.edgecolor":INK,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.linewidth":.7,"xtick.direction":"out","ytick.direction":"out",
    "xtick.major.size":2.5,"ytick.major.size":2.5,"xtick.major.width":.65,"ytick.major.width":.65,
    "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":True,
    "figure.facecolor":"white","savefig.facecolor":"white","savefig.dpi":DPI,
})


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    QA.mkdir(exist_ok=True)
    audit=json.loads((HERE/"data_verification.json").read_text(encoding="utf-8"))
    assert audit["status"]=="PASS"
    for name,digest in audit["data_hashes"].items():
        assert sha(DATA/name)==digest
    cases=pd.read_csv(DATA/"catchment_selection.csv",dtype={"basin_id":str},float_precision="round_trip")
    source=pd.read_csv(DATA/"source_contributions.csv",dtype={"basin_id":str},float_precision="round_trip")
    means=pd.read_csv(DATA/"mean_endpoints.csv",dtype={"basin_id":str},float_precision="round_trip")
    fig,axes=plt.subplots(6,3,figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),
                          gridspec_kw=dict(left=.106,right=.987,top=.918,bottom=.052,wspace=.28,hspace=.40))
    artists,panels,bar_records=[],[],[]
    for ri,case in enumerate(cases.itertuples()):
        for ci,metric in enumerate(METRICS):
            ax=axes[ri,ci]
            sub=source[(source.basin_id==case.basin_id)&(source.metric==metric)]
            avg=means[(means.basin_id==case.basin_id)&(means.metric==metric)].set_index(["comparison","endpoint"])
            assert len(sub)==60 and len(avg)==6
            # Mean endpoints have their own figure; full source contributions are shown in Figure 11D.
            vmin,vmax=min(0,float(avg.estimate.min())),max(0,float(avg.estimate.max()))
            span=max(vmax-vmin,1e-4)
            low,high=vmin-.10*span,vmax+.43*span
            ax.set_ylim(low,high)
            ticks=MaxNLocator(nbins=3,min_n_ticks=3).tick_values(low,high)
            ticks=ticks[(ticks>=low)&(ticks<=high)]
            ax.set_yticks(np.unique(np.r_[ticks,0]))
            ax.ticklabel_format(axis="y",style="plain",useOffset=False)
            ax.axvspan(1.52,2.52,color="#F1F1F1",lw=0,zorder=0)
            ax.set_axisbelow(True)
            ax.grid(axis="y",color="#D8D8D8",lw=.4,ls=(0,(2,3)))
            ax.axhline(0,color=INK,lw=.65,zorder=2)
            for ei,endpoint in enumerate(ENDPOINTS):
                for mi,comparison in enumerate(COMPARISONS):
                    x=ei+(-.18 if mi==0 else .18)
                    value=float(avg.loc[(comparison,endpoint),"estimate"])
                    face=COLOURS[comparison]
                    ax.bar(x,value,width=.30,color=face,alpha=.95 if endpoint=="D_b" else .82,
                           edgecolor=INK,linewidth=.75 if endpoint=="D_b" else .45,zorder=3)
                    ss=sub[(sub.endpoint==endpoint)&(sub.comparison==comparison)].sort_values("source_index")
                    np.testing.assert_allclose(ss.value.mean(),value,atol=1e-13)
                    bar_records.append(dict(basin_id=case.basin_id,metric=metric,comparison=comparison,
                                            endpoint=endpoint,mean=value,n_source_values=len(ss)))
            label=chr(ord('a')+ri*3+ci)
            artists.append(ax.text(.025,.967,f"({label})",transform=ax.transAxes,fontsize=6.8,fontweight="bold",va="top"))
            ds=[]
            for mi,comparison in enumerate(COMPARISONS):
                d=float(avg.loc[(comparison,"D_b"),"estimate"])
                number=f"{d:.4f}".replace("-","−")
                artists.append(ax.text(.968,.968-.135*mi,rf"$D_b = {number}$",transform=ax.transAxes,
                                       fontsize=6.7,color=COLOURS[comparison],fontweight="bold",ha="right",va="top"))
                ds.append(d)
            ax.set_xlim(-.52,2.52)
            ax.set_xticks([0,1,2],[r"$G_{\mathrm{same}}$",r"$G_{\mathrm{new}}$",r"$D_b$"],fontsize=7.3)
            ax.tick_params(axis="x",pad=2)
            ax.tick_params(axis="y",pad=2)
            if ci==0:
                ax.set_ylabel("Gain / loss",fontsize=6.8,labelpad=4)
            if ri==0:
                bbox=ax.get_position()
                fig.text((bbox.x0+bbox.x1)/2,.947,HEADINGS[ci],ha="center",va="center",fontsize=9.5,fontweight="bold")
            panels.append(dict(panel=label,basin_id=case.basin_id,metric=metric,D_M1=ds[0],D_EA=ds[1],
                               ylimits=[low,high],mean_min=float(avg.estimate.min()),mean_max=float(avg.estimate.max()),
                               source_values_averaged=60,mean_bars=6))
        box=axes[ri,0].get_position()
        fig.text(.020,(box.y0+box.y1)/2,case.basin_id,rotation=90,ha="center",va="center",fontsize=8.2,fontweight="bold")
    legend=fig.legend([Patch(facecolor=RED,edgecolor=INK,lw=.5),Patch(facecolor=BLUE,edgecolor=INK,lw=.5)],
                      ["M1/B0","EA/B0"],loc="upper center",
                      bbox_to_anchor=(.55,.993),ncol=2,prop={"weight":"bold","size":8.5},
                      handlelength=1.5,handletextpad=.6,columnspacing=2.2)
    # Panel labels and numerical estimates are part of the data display; there is no explanatory footer.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    outside=[]
    for text in fig.findobj(match=Text):
        if text.get_visible() and text.get_text():
            box=text.get_window_extent(renderer)
            if box.width>0 and box.height>0 and (box.x0 < -1 or box.y0 < -1 or
                    box.x1>fig.bbox.width+1 or box.y1>fig.bbox.height+1):
                outside.append(text.get_text())
    assert not outside,outside
    for ax in axes.flat:
        labels=ax.get_xticklabels()
        assert all(not a.get_window_extent(renderer).overlaps(b.get_window_extent(renderer)) for a,b in zip(labels[:-1],labels[1:]))
    for i,first in enumerate(artists):
        for second in artists[i+1:]:
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)),(first.get_text(),second.get_text())
    assert len(panels)==18 and len(bar_records)==108 and sum(r["n_source_values"] for r in bar_records)==1080
    assert all(p["ylimits"][0]<=p["mean_min"]<=p["mean_max"]<=p["ylimits"][1] for p in panels)
    fig.savefig(QA/"preview.png",dpi=200)
    base=HERE/"figS2"
    for ext in ["png","svg","pdf"]:
        fig.savefig(base.with_suffix("."+ext),dpi=DPI)
    with Image.open(base.with_suffix(".png")) as im:
        pixels=list(im.size)
        im.convert("RGB").save(base.with_suffix(".tif"),compression="tiff_lzw",dpi=(DPI,DPI))
    manifest=dict(figure="S2",width_mm=WIDTH_MM,height_mm=HEIGHT_MM,raster_dpi=DPI,pixels=pixels,font=FONT,
                  no_synthetic_data=True,layout="6 rows x 3 columns",panels=panels,mean_bars=bar_records,
                  n_panels=18,n_mean_bars=108,n_source_values_averaged=1080,
                  source_contribution_display="data/source_contributions.csv",
                  y_scale="Limits vary by panel; both model contrasts and all endpoints share each panel's scale",
                  data_clipping=False,smoothing=False,confidence_intervals=False,explanatory_footnotes=False,
                  outside_canvas_text=outside,data_hashes=audit["data_hashes"],
                  outputs={ext:dict(filename=base.with_suffix("."+ext).name,bytes=base.with_suffix("."+ext).stat().st_size,
                                   sha256=sha(base.with_suffix("."+ext))) for ext in ["png","tif","svg","pdf"]},
                  versions=dict(matplotlib=matplotlib.__version__,numpy=np.__version__,pandas=pd.__version__))
    (HERE/"figure_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    plt.close(fig)
    print(json.dumps({k:manifest[k] for k in ["figure","pixels","n_panels","n_mean_bars","outputs"]},indent=2))


if __name__=="__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
