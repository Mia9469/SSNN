"""New vector figures; retain the author's Figure 1/2 and Figure 3a/4a-d."""
from pathlib import Path
from copy import deepcopy
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject
from strengthening_analysis import local

HERE=Path(__file__).resolve().parent
BASE=HERE/'author_figures'
OUT=HERE/'figures'; PREVIEW=HERE/'output/preview'
INK,TEAL,BLUE,RUST,GRAY='#182630','#29776e','#245e94','#a0422b','#788892'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.7,
    'axes.labelsize':8.5,'axes.titlesize':9.5,'xtick.labelsize':7.4,
    'ytick.labelsize':7.4,'legend.fontsize':6.5,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':INK,'axes.edgecolor':INK,
    'text.color':INK,'axes.titleweight':'semibold','pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,stem):
    OUT.mkdir(exist_ok=True);PREVIEW.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{stem}.pdf',facecolor='white')
    fig.savefig(PREVIEW/f'{stem}.png',dpi=180,facecolor='white')
    plt.close(fig)


def author_scene(target, source, top_height):
    page=deepcopy(PdfReader(target).pages[0]);old=PdfReader(source).pages[0]
    w,h=float(old.mediabox.width),float(old.mediabox.height)
    dh=float(page.mediabox.height)-h
    clip=deepcopy(old)
    clip.mediabox=RectangleObject([0,h-top_height,w,h]);clip.cropbox=clip.mediabox
    page.merge_transformed_page(clip,Transformation().translate(0,dh))
    writer=PdfWriter();writer.add_page(page)
    with target.open('wb') as f: writer.write(f)


def figure3(record):
    r=record['region']; t=np.asarray(r['times']); w,h=497.518,405.
    fig=plt.figure(figsize=(w/72,h/72))
    b=fig.add_axes([.08,.47,.38,.26]);c=fig.add_axes([.60,.47,.37,.26])
    d=fig.add_axes([.08,.10,.38,.23]);e=fig.add_axes([.60,.10,.32,.23])
    ts=np.linspace(0,5.5,400);kernel=(np.exp(-.5*ts)-np.exp(-2*ts))/1.5/np.sqrt(2)
    b.plot(ts,2.34*kernel,c=RUST,lw=1.6,label='Slow output')
    b.plot(ts,.9*kernel,c=BLUE,ls='--',lw=1.5,label='Fast output')
    b.set(title='(b) Mean ranking stays fixed',xlabel='Time after cue',
          ylabel='Mean response to positive cue',xlim=(0,5.5),ylim=(0,.6))
    b.legend(frameon=False,loc='upper right');b.text(1.7,.35,r'Slow signal $=2.6\times$ fast',fontsize=7,color=GRAY)
    for label,col,ls in [('Baseline',INK,'-'),('No source noise',TEAL,'--'),('No local noise',GRAY,':')]:
        c.plot(t,r['ablations'][label]['ratio'],c=col,ls=ls,lw=1.4,label=label)
    c.axvline(1.39930089008,c=GRAY,ls=':',lw=.7)
    for tt,col in [(.5,RUST),(3.,BLUE)]:
        y=np.interp(tt,t,r['ablations']['Baseline']['ratio']);c.plot(tt,y,'o',c=col,ms=3)
    c.set(title='(c) Noise removal and the reversal',xlabel='Readout duration $T$',
          ylabel=r"Cue separation ratio $d'_{\rm slow}/d'_{\rm fast}$",
          xlim=(0,5.5),ylim=(.65,1.36))
    c.legend(frameon=False,loc='upper right',handlelength=2.,labelspacing=.2)
    tt=np.linspace(.002,5.5,400);prices=np.array([local(.5,v)/local(2.,v) for v in tt])
    d.plot(tt,prices,c=INK,lw=1.5,label='Equal-discrimination price')
    d.axhline(6.76,c=TEAL,ls='--',lw=1.4,label='Actual allocation')
    d.axvline(1.39930089008,c=GRAY,ls=':',lw=.8)
    d.plot(1.39930089008,6.76,'o',c=INK,ms=3)
    d.set(title='(d) The budget-price crossing',xlabel='Readout duration $T$',
          ylabel='Slow / fast budget ratio',xlim=(0,5.5),ylim=(3.5,13))
    d.legend(frameon=False,loc='lower right',fontsize=6.2)
    x,y=np.asarray(r['rates']),np.asarray(r['gammas'])
    z=np.array([[np.nan if v is None else v for v in row] for row in r['strength_percent']])
    cmap=LinearSegmentedColormap.from_list('advantage',['#edf5f1','#9bc2b9',TEAL])
    im=e.pcolormesh(x,y,np.ma.masked_invalid(z),cmap=cmap,vmin=0,vmax=12,rasterized=True,shading='auto')
    e.plot(x,np.sqrt(x),c=GRAY,ls='--',lw=.8);e.plot(x,x,c=GRAY,ls='--',lw=.8)
    e.plot(4.,2.6,'o',mec=INK,mfc='white',ms=4)
    e.set(title='(e) Reversal region at fixed cost',xlabel=r'Rate ratio $R=a_f/a_s$',
          ylabel=r'Coupling ratio $\gamma=m_s/m_f$',xlim=(1,8),ylim=(1,8),xticks=[2,4,6,8],yticks=[2,4,6,8])
    e.text(1.4,6.,'Slow wins\nfor all $T$',fontsize=6.5,color=GRAY)
    e.text(5.3,1.35,'Fast wins\nfor all $T$',fontsize=6.3,color=GRAY)
    cb=fig.colorbar(im,cax=fig.add_axes([.941,.10,.014,.23]),ticks=[0,6,12])
    cb.ax.tick_params(labelsize=6.5,pad=2)
    cb.ax.set_title('%',fontsize=7,pad=4)
    save(fig,'fig_prr3_strengthened')
    author_scene(OUT/'fig_prr3_strengthened.pdf',BASE/'fig_prr3_fixed_state_polished.pdf',85.)


def figure4(record):
    rows=record['recurrent']['rows'];w=507.678;rh=137.
    fig=plt.figure(figsize=(w/72,rh/72))
    e=fig.add_axes([.10,.27,.36,.58]);f=fig.add_axes([.64,.27,.32,.58])
    e.plot([0,1],[0,1],c=GRAY,ls=':',lw=1.)
    for t,col,marker in [(.35,RUST,'o'),(1.6,BLUE,'s')]:
        a=[r for r in rows if r['T']==t]
        e.scatter([v for r in a for v in r['predicted_shares']],
                  [v for r in a for v in r['direct_shares']],s=12,alpha=.6,c=col,marker=marker,
                  label=f'$T={t}$',edgecolors='none')
    e.set(title='(e) Predict recurrent allocations',xlabel='Predicted budget share',
          ylabel='Directly optimized share',xlim=(-.02,1.02),ylim=(-.02,1.02),xticks=[0,.5,1],yticks=[0,.5,1])
    e.title.set_fontsize(9.4);e.legend(frameon=False,loc='upper left',fontsize=6.4)
    for k in range(8):
        for mult in (.25,1.,4.):
            pair=[r for r in rows if r['network']==k and r['budget_multiplier']==mult]
            vals=[r['gain_over_equal_percent'] for r in pair]
            f.plot([0,1],vals,c=GRAY,alpha=.25,lw=.5,zorder=1)
            for pos,v,col,m in zip([0,1],vals,[RUST,BLUE],['o','s']):
                f.plot(pos,v,m,c=col,ms=2.8,alpha=.65)
    meds=[np.median([r['gain_over_equal_percent'] for r in rows if r['T']==t]) for t in (.35,1.6)]
    f.plot([-.16,.16],[meds[0]]*2,c=INK,lw=1.7);f.plot([.84,1.16],[meds[1]]*2,c=INK,lw=1.7)
    f.set(title='(f) Gain over equal allocation',ylabel='Joint utility gain (%)',xlim=(-.3,1.3),
          xticks=[0,1],xticklabels=['$T=0.35$','$T=1.6$'])
    f.title.set_fontsize(9.4);f.axhline(0,c=GRAY,lw=.6,ls=':')
    save(fig,'fig4_prediction_row')
    old=PdfReader(BASE/'fig_prr4_allocation_polished.pdf').pages[0]
    page=PageObject.create_blank_page(width=w,height=float(old.mediabox.height)+rh+13.)
    page.merge_transformed_page(deepcopy(old),Transformation().translate(0,rh+13.))
    page.merge_page(PdfReader(OUT/'fig4_prediction_row.pdf').pages[0])
    writer=PdfWriter();writer.add_page(page)
    with (OUT/'fig_prr4_strengthened.pdf').open('wb') as stream:writer.write(stream)


def supplement(record):
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2))
    fig.subplots_adjust(left=.17,right=.98,bottom=.12,top=.91,wspace=.75,hspace=.65)
    a,b,c,d=axes.ravel();base=json.loads((HERE/'source_records/window_crossover_results.json').read_text())
    families=[('Baseline','box',(0.,0.),0.),('Return on both','box',(.1,.1),0.),
        ('Colored drive','box',(0.,0.),1.),('Triangular window','triangle',(0.,0.),0.),
        ('Exponential window','exponential',(0.,0.),0.),('One-sided return','box',(.1,0.),0.),
        ('Return + colored','box',(.1,.1),1.)]
    for i,(name,shape,fb,color) in enumerate(families):
        r=next(z for z in base['tests'] if z['window']==shape and tuple(z['feedback_over_a_slow'])==fb and z['tau_times_a_fast']==color)
        x,y=[s['dprime_ratio'] for s in r['test_points']]
        a.plot([x,y],[i,i],c=GRAY,lw=.8)
        a.plot(x,i,'o',c=RUST,ms=3.5,label='Short $T=0.5$' if i==0 else None)
        a.plot(y,i,'s',mec=BLUE,mfc='white',ms=3.5,label='Long $T=3$' if i==0 else None)
    a.axvline(1,c=GRAY,ls=':',lw=.8);a.set(yticks=range(7),yticklabels=[r[0] for r in families],ylim=(6.5,-1.2),
        xlabel=r"Separation ratio $d'_s/d'_f$",title='(a) Robustness across model changes')
    a.tick_params(axis='y',labelsize=6.8);a.legend(frameon=False,fontsize=5.8,loc='upper center',ncol=2,handlelength=1.2)
    t=np.array(record['region']['times'])
    b.plot(t,record['region']['ablations']['Baseline']['ratio'],c=INK,label='Matched source drive')
    b.plot(t,record['region']['mismatch']['ratio'],c=TEAL,ls='--',label='Source drive unequal')
    b.axhline(1,c=GRAY,ls=':',lw=.8)
    b.set(xlim=(0,5.5),xlabel='Readout duration $T$',ylabel=r"Separation ratio $d'_s/d'_f$",title='(b) Shared-term assumptions matter')
    b.legend(frameon=False,fontsize=6.4,loc='center right')
    rows=record['recurrent']['rows'];c.plot([0,1],[0,1],c=GRAY,ls=':')
    for t,col,mark in [(.35,RUST,'o'),(1.6,BLUE,'s')]:
        subset=[r for r in rows if r['T']==t]
        c.scatter([x for r in subset for x in r['fair_predicted_shares']],
                  [x for r in subset for x in r['fair_direct_shares']],s=10,c=col,marker=mark,alpha=.6,label=f'$T={t}$')
    c.set(xlim=(0,1),ylim=(0,1),xlabel='Predicted budget share',ylabel='Direct max-min share',title='(c) Independently check the second task')
    c.legend(frameon=False,fontsize=6.5,loc='upper left')
    for i,mult in enumerate((.25,1.,4.)):
        for objective,col,mark,offset in [('joint',INK,'o',-.08),('fair',TEAL,'s',.08)]:
            vals=[r[f'{objective}_share_total_variation'] for r in record['recurrent']['shifts'] if r['budget_multiplier']==mult]
            d.scatter(np.full(8,i+offset),vals,s=11,c=col,marker=mark,alpha=.65,label=objective if i==0 else None)
    d.set(xticks=range(3),xticklabels=['0.25','1','4'],xlabel='Budget / unit-gain cost',
          ylabel='Allocation change with $T$',title='(d) Window-dependent budget changes')
    d.legend(frameon=False,loc='upper left',fontsize=6.4)
    for ax in axes.ravel():ax.title.set_fontsize(8.)
    save(fig,'fig_prrS_strengthening')


def main():
    record=json.loads((HERE/'strengthening_results.json').read_text())
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.pdf')}
    figure3(record);figure4(record);supplement(record)
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.pdf')}
    print('New Figure 3/4 and supplementary checks written; author sources unchanged.')


if __name__=='__main__':main()
