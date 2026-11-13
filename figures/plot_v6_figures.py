"""Publication figures from verified, local V6 display tables (no synthetic data)."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, LogLocator, NullFormatter, LogFormatterMathtext
import matplotlib.dates as mdates
from scipy.interpolate import PchipInterpolator
from PIL import Image

ROOT=Path(__file__).resolve().parent
INK,RED,BLUE,GOLD,GREY='#1A1A1A','#D62728','#1F5FA8','#D99912','#808080'
COL={'M1/B0':RED,'EA/B0':BLUE}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],'mathtext.fontset':'stix',
 'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'axes.titleweight':'bold','axes.labelweight':'bold',
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.frameon':False,'axes.linewidth':.8,
 'xtick.direction':'out','ytick.direction':'out','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white','figure.facecolor':'white','savefig.dpi':600})
def folder(n): return ROOT/(f'Fig{n:02}' if isinstance(n,int) else f'Fig{n}')
def data(n,file,**kw): return pd.read_csv(folder(n)/'data'/file,float_precision='round_trip',**kw)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finish(fig,n,extra=None):
    p=folder(n); (p/'qa').mkdir(exist_ok=True)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[]
    visible_text=list(fig.texts)
    for ax in fig.axes:
        visible_text+=list(ax.texts)+[ax.title]
        if ax.axison:
            visible_text += [ax.xaxis.label,ax.yaxis.label]
            visible_text += [t for t in ax.get_xticklabels() if min(ax.get_xlim())<=t.get_position()[0]<=max(ax.get_xlim())]
            visible_text += [t for t in ax.get_yticklabels() if min(ax.get_ylim())<=t.get_position()[1]<=max(ax.get_ylim())]
    for t in visible_text:
        if not t.get_visible() or not t.get_text(): continue
        if t.axes is not None and t in t.axes.get_yticklabels() and not min(t.axes.get_ylim())<=t.get_position()[1]<=max(t.axes.get_ylim()): continue
        bb=t.get_window_extent(renderer)
        if bb.width and bb.height and (bb.x0< -2 or bb.y0< -2 or bb.x1>fig.bbox.width+2 or bb.y1>fig.bbox.height+2): outside.append(t.get_text())
    if outside: raise AssertionError((n,'Text outside canvas',outside))
    base=p/f'fig{n}'; fig.savefig(p/'qa/preview.png',dpi=180)
    for ext in ['png','svg','pdf']: fig.savefig(base.with_suffix('.'+ext),dpi=600)
    with Image.open(base.with_suffix('.png')) as im:
        pixels=im.size; im.convert('RGB').save(base.with_suffix('.tif'),compression='tiff_lzw',dpi=(600,600))
    manifest=dict(figure=str(n),revision='V6',raster_dpi=600,dimensions_mm=(fig.get_size_inches()*25.4).tolist(),
                  pixels=pixels,vector_formats=['pdf','svg'],no_explanatory_footer=True,outside_canvas_text=outside,
                  source_data_hashes={f.name:sha(f) for f in sorted((p/'data').glob('*')) if f.is_file()},
                  outputs={ext:dict(filename=base.with_suffix('.'+ext).name,sha256=sha(base.with_suffix('.'+ext))) for ext in ['png','tif','svg','pdf']})
    if extra: manifest.update(extra)
    (p/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig); print('Rendered',n,flush=True)
def grid(ax,axis='y'):
    ax.grid(axis=axis,color='#D0D0D0',lw=.55,ls=(0,(4,3))); ax.set_axisbelow(True)

def fig4():
    table=data(4,'endpoints.csv'); ratios=data(4,'transfer_ratios.csv')
    fig,axs=plt.subplots(1,2,figsize=(190/25.4,87/25.4),gridspec_kw=dict(left=.085,right=.984,bottom=.23,top=.87,wspace=.24))
    for pi,(comp,col) in enumerate(COL.items()):
        ax=axs[pi]; t=table[table.comparison==comp.replace('/','-')].set_index('endpoint')
        names=['same','D','other','global_other','incremental_other']; xs=np.arange(5)
        for i,key in enumerate(names):
            r=t.loc[key]; value=r.estimate
            if key=='D':
                ax.bar(i,-value,bottom=t.loc['same','estimate'],color='#DEDEDE',edgecolor=INK,lw=.6,hatch='///',width=.62,zorder=3)
                # This floating segment displays the paired difference only; its CI is shown in Fig. 7.
                ax.text(i,t.loc['same','estimate']+.005,f'−{value:.3f}',ha='center',va='bottom',fontsize=7.6,fontweight='bold')
            else:
                ax.bar(i,value,color=col if key in ['same','other'] else GOLD if key=='global_other' else '#A7A7A7',
                       edgecolor=INK,lw=.6,width=.62,alpha=1 if key!='same' else .80,zorder=3)
                ax.errorbar(i,value,yerr=[[value-r.low95],[r.high95-value]],fmt='none',color=INK,lw=.8,capsize=2,zorder=5)
                y=r.high95+.006
                ax.text(i,y,f'{value:.3f}'.replace('-','−'),ha='center',va='bottom',fontsize=7.5,fontweight='bold')
        for x,y in [(0,t.loc['same','estimate']),(1,t.loc['other','estimate'])]: ax.plot([x+.31,x+1-.31],[y,y],color=INK,lw=.6)
        ax.axhline(0,color=INK,lw=.8); grid(ax)
        ax.set_xticks(xs,['Retained\ngain','Retraining\nloss','Transferred\ngain','Global-rule\ngain','Local\nincrement'],fontweight='bold',fontsize=7)
        ax.set_xlim(-.6,4.6); ax.set_ylim(-.112,.188)
        ax.set_ylabel('Late-period NSE gain' if pi==0 else '')
        ax.set_title(f'({"ab"[pi]}) {comp}',pad=9)
        ratio=t.loc['other','estimate']/t.loc['same','estimate']
        ax.text(.97,.96,f'Transfer ratio = {ratio:.2f}'.replace('-','−'),transform=ax.transAxes,ha='right',va='top',fontsize=8,fontweight='bold')
    finish(fig,4,dict(floating_loss_uncertainty='Shown in Fig. 7; other four bars have original synchronized seed/HUC2 95% intervals'))

def fig5():
    t=data(5,'curve_endpoints.csv'); changes=data(5,'paired_changes_k1_k5.csv')
    fig=plt.figure(figsize=(190/25.4,137/25.4))
    gs=fig.add_gridspec(2,2,left=.088,right=.98,top=.90,bottom=.09,wspace=.27,hspace=.52,height_ratios=[1.25,.72])
    legend_handles = {}
    interval_label = '95% CI limits for transferred gain'
    for pi,(comp,col) in enumerate(COL.items()):
        ax=fig.add_subplot(gs[0,pi]); s=t[t.comparison==comp]
        y={e:s[s.endpoint==e].sort_values('k') for e in ['G_same','G_new','G_global_new','D']}
        x=np.arange(1,6)
        legend_handles['Retraining loss'] = ax.fill_between(x,y['G_same'].estimate,y['G_new'].estimate,color='#E3E3E3',hatch='///',edgecolor='#B3B3B3',lw=.0,zorder=0)
        legend_handles['Local increment'] = ax.fill_between(x,y['G_new'].estimate,y['G_global_new'].estimate,color=GOLD,alpha=.22,lw=0,zorder=1)
        for e,colour,marker,label in [('G_same',col,'o',f'Retained gain ({comp})'),('G_new',INK,'s','Transferred gain'),('G_global_new',GOLD,'^','Global-rule gain')]:
            line, = ax.plot(x,y[e].estimate,marker=marker,ms=4.3,lw=1.5,color=colour,mec='white',mew=.35,label=label,zorder=4)
            legend_handles[label] = line
        lower, = ax.plot(x,y['G_new'].low95,ls=':',color=INK,lw=.8)
        ax.plot(x,y['G_new'].high95,ls=':',color=INK,lw=.8)
        legend_handles[interval_label] = lower
        for k,ha,dx in [(1,'left',.10),(5,'right',-.10)]:
            yy=float(y['G_same'].iloc[k-1].estimate)+.007
            ax.text(k+dx,yy,f'$D$ = {y["D"].iloc[k-1].estimate:.3f}',ha=ha,va='bottom',fontsize=7.4,fontweight='bold')
        ax.axhline(0,color='#888888',lw=.65); grid(ax)
        ax.set_xlim(.8,5.2); ax.set_ylim(-.055,.105); ax.set_xticks(x)
        ax.set_xlabel('Source instances, k'); ax.set_ylabel('Late-period NSE gain' if pi==0 else '')
        ax.set_title(f'({"ab"[pi]}) {comp}',pad=8)
    ax=fig.add_subplot(gs[1,:]); pos=ax.get_position(); ax.set_position([.24,pos.y0,.74,pos.height]); labs=[]
    for i,(comp,col) in enumerate(COL.items()):
        for j,e in enumerate(['G_same','G_new']):
            y=3-(i*2+j); r=changes[(changes.comparison==comp)&(changes.endpoint==e)].iloc[0]
            ax.barh(y,r.estimate,color=col,alpha=.95 if j==0 else .45,height=.6,edgecolor=INK,lw=.5)
            ax.errorbar(r.estimate,y,xerr=[[r.estimate-r.low95],[r.high95-r.estimate]],fmt='none',color=INK,lw=.8,capsize=2)
            ax.text(r.high95+.0018,y,f'{r.estimate:+.3f}'.replace('-','−'),va='center',fontsize=7.5,fontweight='bold')
            labs.append(f'{comp} · {"Retained" if j==0 else "Transferred"}')
    ax.set_yticks([3,2,1,0],labs,fontsize=7.5); ax.axvline(0,color=INK,lw=.8); grid(ax,'x')
    ax.set_xlim(-.078,.070); ax.set_ylim(-.6,3.6); ax.set_xlabel('Change in gain from k = 1 to k = 5 (NSE)')
    ax.set_title('(c) Why the retraining loss shrinks',pad=7)
    labels = ['Retained gain (M1/B0)', 'Retained gain (EA/B0)',
              'Transferred gain', interval_label, 'Global-rule gain', 'Local increment', 'Retraining loss']
    fig.legend([legend_handles[label] for label in labels], labels, loc='upper center',
               bbox_to_anchor=(.5,.993), ncol=4, fontsize=7, handlelength=1.8,
               columnspacing=1.5, labelspacing=.35, borderpad=0, borderaxespad=0)
    finish(fig,5,dict(intervals='Original 20,000 HUC2 resamples, conditional on the fitted library; dotted bounds for transferred gain'))

def fig7():
    cases=data(7,'case_selection.csv',dtype={'basin_id':str}); daily=data(7,'daily_nine_instances.csv',dtype={'basin_id':str},parse_dates=['date']); fdc=data(7,'fdc_nine_instances.csv',dtype={'basin_id':str})
    styles={'observed':('Observed',INK,1.0,'-'),'B0_source':('B0',GREY,.7,'-'),'EA_retained':('EA retained',RED,.9,'-'),
            'EA_new':('EA independent: mean + IQR',BLUE,1,'-'),'pooled_ensemble':('Pooled ensemble',GOLD,.9,'--')}
    fig=plt.figure(figsize=(190/25.4,151/25.4)); gs=fig.add_gridspec(3,2,left=.10,right=.98,bottom=.065,top=.90,wspace=.28,hspace=.48,width_ratios=[2.5,1])
    for i,c in enumerate(cases.itertuples()):
        g=daily[(daily.basin_id==c.basin_id)&daily.in_hydrograph]; f=fdc[fdc.basin_id==c.basin_id]
        ax=fig.add_subplot(gs[i,0]); ax.fill_between(g.date,g.EA_new_q25,g.EA_new_q75,color=BLUE,alpha=.23,lw=0,zorder=1)
        for series,(label,col,lw,ls) in styles.items(): ax.plot(g.date,g[series],color=col,lw=lw,ls=ls,zorder=6 if series=='observed' else 3)
        ax.set_xlim(g.date.min(),g.date.max()); ax.set_ylim(0,1.06*g[list(styles)+['EA_new_q75']].max().max()); ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10,1,4,7])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_ylabel(r'Q (mm d$^{-1}$)'); ax.set_title(f'({"ace"[i]}) {c.basin_id} · WY2004',pad=6)
        ax=fig.add_subplot(gs[i,1]); new=f[f.series=='EA_new'].sort_values('rank'); ax.fill_between(new.exceedance_pct,new.q25,new.q75,color=BLUE,alpha=.23,lw=0)
        for series,(label,col,lw,ls) in styles.items():
            s=f[f.series==series].sort_values('rank'); ax.plot(s.exceedance_pct,s.q_mm_day,color=col,lw=lw,ls=ls,zorder=6 if series=='observed' else 3)
        ax.axvspan(.05,2,color='#FBE3C8',alpha=.5,lw=0,zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(.05,100)
        ax.set_ylim(min(f.q_mm_day.min(),new.q25.min())/1.2,max(f.q_mm_day.max(),new.q75.max())*1.18)
        ax.set_xticks([.1,1,10,100],['0.1','1','10','100']); ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.set_ylabel(r'Q (mm d$^{-1}$)'); ax.set_xlabel('Exceedance (%)' if i==2 else ''); ax.set_title(f'({"bdf"[i]}) Flow duration\nWY2004–2008',pad=6,fontsize=8,linespacing=1.05)
    fig.legend([Line2D([],[],color=col,lw=lw+.2,ls=ls) for _,col,lw,ls in styles.values()],[s[0] for s in styles.values()],
               loc='upper center',bbox_to_anchor=(.53,.992),ncol=3,fontsize=8,columnspacing=1.8,handlelength=2)
    finish(fig,7,dict(independent_EA_instances=9,band='Interquartile range across independent instances; source excluded'))

def fig8():
    from Fig08.plot_figure8 import render_figure8
    render_figure8(folder(8), finish)

def fig9():
    import runpy
    runpy.run_path(str(folder(9) / 'plot_figure9.py'), run_name='__main__')

def figS4():
    t=data('S4','reference_radar_profiles.csv'); gain=data('S4','gain_concentration.csv',dtype={'basin_id':str})
    failures=data('S4','failure_matrix.csv',dtype={'basin_id':str}).set_index('basin_id')
    metrics=['nse','kge','r','alpha_closeness','beta_closeness','abs_fhv','mae_high','mae_low']
    names=['NSE','KGE','r','1−|α−1|','1−|β−1|','|FHV|','High MAE','Low MAE']
    groups=[('typical_heldout','Typical held-out (100)'),('early_risk','Early risk (7)'),('training','Training (370)'),('external','External (531)')]
    fig=plt.figure(figsize=(190/25.4,134/25.4)); ang=np.pi/2-np.arange(8)*np.pi/4
    cx,cy=np.cos(ang),np.sin(ang)
    for i,(group,title) in enumerate(groups):
        ax=fig.add_axes([.005+i*.248,.512,.247,.355]); g=t[t.group==group]
        for radius in [.25,.5,.75,1]:
            ax.plot(np.r_[cx,cx[0]]*radius,np.r_[cy,cy[0]]*radius,color=INK if radius==1 else '#D3D3D3',lw=.9 if radius==1 else .55)
        for x,y in zip(cx,cy): ax.plot([0,x],[0,y],color='#D0D0D0',lw=.5)
        ax.plot(np.r_[cx,cx[0]]*.5,np.r_[cy,cy[0]]*.5,color=INK,ls='--',lw=1.1,zorder=3)
        for policy,col,marker in [('library_local',RED,'o'),('select_arch_then_ensemble',GOLD,'s')]:
            r=g[g.policy==policy].set_index('metric').loc[metrics].radius.to_numpy()
            assert np.all((r>=0)&(r<=1))
            ax.fill(np.r_[cx*r,cx[0]*r[0]],np.r_[cy*r,cy[0]*r[0]],color=col,alpha=.13,zorder=2)
            ax.plot(np.r_[cx*r,cx[0]*r[0]],np.r_[cy*r,cy[0]*r[0]],color=col,marker=marker,lw=1.3,ms=3.4,mec=INK,mew=.4,zorder=4)
        rg=g[g.policy=='library_local'].set_index('metric')
        for x,y,m,name in zip(cx,cy,metrics,names):
            ax.text(x*1.13,y*1.13,f'{name}\n(±{rg.loc[m,"axis_half_range"]:g})',fontsize=5.9,fontweight='bold',
                    ha='center' if abs(x)<.3 else ('left' if x>0 else 'right'),va='center' if abs(y)<.3 else ('bottom' if y>0 else 'top'),linespacing=1.05)
        ax.set_xlim(-1.95,1.95); ax.set_ylim(-1.48,1.48); ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'({"abcd"[i]}) {title}',fontsize=8,pad=11)
    fig.legend([Line2D([],[],color=RED,marker='o',mec=INK,mew=.5,ms=5),Line2D([],[],color=GOLD,marker='s',mec=INK,mew=.5,ms=5),Line2D([],[],color=INK,ls='--',lw=1.2)],
               ['One instance per catchment','Architecture choice + ensemble','Pooled ensemble (reference)'],
               loc='upper center',bbox_to_anchor=(.51,.993),ncol=3,prop={'size':7.7,'weight':'bold'},handlelength=1.8,columnspacing=1.7)
    ax=fig.add_axes([.085,.105,.385,.335]); grid(ax)
    ax.plot(gain['rank'],gain.cumulative_gain,color=INK,lw=1.5)
    first=gain.iloc[:3]; total=gain.cumulative_gain.iloc[-1]
    ax.plot(first['rank'],first.cumulative_gain,'o',color=RED,mec=INK,mew=.5,ms=5)
    ax.axhline(total,ls='--',color=GREY,lw=.8)
    ax.text(103,total+.11,f'Net total {total:.2f}',ha='right',va='bottom',color='#606060',fontsize=7)
    ax.annotate(f'Top 3: {first.iloc[-1].cumulative_net_share_pct:.1f}% of net gain',
                xy=(3,first.iloc[-1].cumulative_gain),xytext=(1.12,5.75),fontsize=7,fontweight='bold',
                arrowprops=dict(arrowstyle='-',lw=.7,color=INK))
    ax.set_xscale('log'); ax.set_xlim(.9,120); ax.set_xticks([1,2,3,10,30,100],['1','2','3','10','30','100']); ax.set_ylim(.8,max(gain.cumulative_gain)*1.07)
    ax.set_ylabel('Cumulative ΣΔNSE\n(one instance − pooled)',fontsize=8)
    ax.set_xlabel('Held-out catchments ranked by gain',fontsize=7.7); ax.set_title('(e) Where the gain comes from',pad=8)
    ax=fig.add_axes([.615,.105,.335,.335]); grid(ax); ax.axhline(0,color=INK,ls='--',lw=.8)
    for basin in ['04127997','08324000','07301410']:
        vals=failures.loc[basin,['library_local','select_arch_then_ensemble','mixed_ensemble']].to_numpy(float)
        ax.plot([0,1,2],vals,color=RED,lw=1.4,marker='o',ms=4.6,mec=INK,mew=.5)
        offset={'04127997':.08,'08324000':-.08,'07301410':0}[basin]
        ax.annotate(basin,xy=(2,vals[2]),xytext=(2.09,vals[2]+offset),fontsize=6.7,va='center',fontweight='bold',arrowprops=dict(arrowstyle='-',lw=.5,color=INK))
    ax.set_xlim(-.15,2.9); ax.set_ylim(-1.57,.52); ax.set_yticks([-1.5,-1,-.5,0,.5])
    ax.set_xticks([0,1,2],['One\ninstance','Arch. +\nensemble','Pooled\nensemble'],fontsize=7.5,fontweight='bold')
    ax.set_ylabel('Late-period NSE'); ax.set_title('(f) Three dominant catchments',pad=8)
    finish(fig,'S4',dict(radar_reference='Radius 0.5 = pooled strategy median; radius differences are oriented so outward is better',
                     radar_comparable_ranges='Identical in panels a,c,d; wider in early-risk panel b',radar_clipped_values=0,
                     cumulative_curve='Exact unsmoothed sum of ranked NSE gains',ensemble_size_local=20,ensemble_size_external=16))

def figS1():
    bal=data('S1','covariate_balance.csv'); ecdf=data('S1','score_ecdfs.csv')
    schemes=[('unmatched','Unmatched',GREY,'o'),('across_HUC2','Across HUC2',GOLD,'s'),('within_HUC2','Within HUC2',INK,'D')]
    attrs=bal.attribute.unique().tolist(); names=[{'early_B0_NSEstar':'Early B0 NSE*','aridity':'Aridity','frac_snow':'Snow fraction','log_area':'Log area','baseflow_index':'Baseflow index','p_mean':'Precipitation','elev_mean':'Elevation'}[a] for a in attrs]
    fig,axs=plt.subplots(1,2,figsize=(190/25.4,92/25.4),gridspec_kw=dict(left=.145,right=.98,bottom=.15,top=.83,wspace=.40))
    ax=axs[0]; ax.axvspan(-.1,.1,color='#EEEEEE'); ax.axvline(0,color=INK,lw=.8)
    for i,(scheme,label,col,m) in enumerate(schemes):
        s=bal[bal.matching==scheme].set_index('attribute').loc[attrs]
        ax.plot(s.SMD,np.arange(len(attrs))[::-1]+(.20-i*.20),marker=m,color=col,ls='none',ms=4.6,mec=INK,mew=.4,label=label)
    ax.set_yticks(np.arange(len(attrs))[::-1],names); ax.set_ylim(-.6,6.6); ax.set_xlim(-.7,.65); grid(ax,'x')
    ax.set_xlabel('SMD (held-out − training)'); ax.set_title('(a) Covariate balance',pad=8); ax.legend(fontsize=7,loc='upper right')
    ax=axs[1]
    lines=[('unmatched','heldout','Unmatched held-out (107)',GREY,'-'),('across_HUC2','heldout','Across-HUC2 held-out (101)',GOLD,'-'),
           ('within_HUC2','heldout','Within-HUC2 held-out (77)',INK,'-'),('unmatched','training','Training (370)',INK,'--')]
    for scheme,role,label,col,ls in lines:
        g=ecdf[(ecdf.matching==scheme)&(ecdf.role==role)].sort_values('early_B0_NSEstar')
        x=np.r_[-1,g.early_B0_NSEstar,1]; y=np.r_[0,g.ecdf,1]; xx=np.unique(np.r_[np.linspace(-1,1,4001),x])
        yy=PchipInterpolator(x,y)(xx); assert np.all(np.diff(yy)>=-1e-12)
        ax.plot(xx,yy,color=col,ls=ls,lw=1.35,label=label)
    ax.set_xlim(-1,1); ax.set_ylim(0,1); grid(ax); ax.set_xlabel('Early B0 NSE*'); ax.set_ylabel('Cumulative fraction'); ax.set_title('(b) Baseline-skill distributions',pad=8)
    ax.legend(loc='upper left',fontsize=6.8,handlelength=2,labelspacing=.55)
    finish(fig,'S1',dict(ECDF_curves=4,interpolation='Monotone PCHIP through every empirical point'))

def figS3():
    folder('S3').mkdir(exist_ok=True); (folder('S3')/'data').mkdir(exist_ok=True)
    import shutil
    for f in ['curve_endpoints.csv','power_fits.csv']: shutil.copy2(folder(5)/'data'/f,folder('S3')/'data'/f)
    t=data(5,'curve_endpoints.csv'); fits=data(5,'power_fits.csv')
    fig,axs=plt.subplots(1,2,figsize=(190/25.4,80/25.4),gridspec_kw=dict(left=.085,right=.98,bottom=.18,top=.85,wspace=.25))
    for ax,(comp,col) in zip(axs,COL.items()):
        s=t[(t.comparison==comp)&(t.endpoint=='D')]; r=fits[fits.comparison==comp].iloc[0]; x=np.linspace(1,5,101)
        ax.errorbar(s.k,s.estimate,yerr=[s.estimate-s.low95,s.high95-s.estimate],color=col,marker='o',ls='none',capsize=2,ms=4.5)
        ax.plot(x,r.amplitude*x**(-r.alpha),color=col,lw=1.4,label=f'Power fit: α = {r.alpha:.2f}')
        ax.plot(x,s.estimate.iloc[0]/np.sqrt(x),color=GREY,lw=1,ls='--',label=r'$k^{-1/2}$ reference')
        ax.set_xticks(range(1,6)); ax.set_xlabel('Source instances, k'); ax.set_ylabel('Retraining loss, D (NSE)'); ax.set_title(comp); grid(ax); ax.legend(fontsize=7)
    finish(fig,'S3',dict(fit_scope='Observed k=1..5 only; intervals conditional on fixed library'))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for n in (sys.argv[1:] or ['4','5','7','8','9','S1','S3','S4']): globals()['fig'+n]()
