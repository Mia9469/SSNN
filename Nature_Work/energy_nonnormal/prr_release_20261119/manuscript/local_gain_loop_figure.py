"""Vector figure for the activity-driven feedback extension."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
INK, TEAL, BLUE, RUST, GRAY = '#182630','#29776e','#245e94','#a0422b','#788892'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':9.5,
    'axes.titleweight':'semibold','axes.spines.top':False,'axes.spines.right':False,
    'axes.labelcolor':INK,'axes.edgecolor':INK,'text.color':INK,'legend.fontsize':7,'pdf.fonttype':42})


def plot(ax, hist, field, color, label, *, idx=None, ls='-', band=False, scale=1., shift=0):
    rows = list(hist.values())[1:]
    x = np.array([r['decisions_per_output'] for r in rows])+shift
    q = np.array([r[field] for r in rows])*scale
    if idx is not None:
        q=q[:,:,idx]
    if band:
        ax.fill_between(x,q[:,0],q[:,2],color=color,alpha=.13,linewidth=0.)
    ax.plot(x,q[:,1],color=color,ls=ls,lw=1.35,label=label)


def main():
    z = json.loads((HERE/'local_gain_loop_results.json').read_text())['runs']
    fig, axs = plt.subplots(2,2,figsize=(7.1,5.0),layout='constrained')
    a,b,c,d = axs.flat
    n=z['2_0.5']['margin']['4096']['decisions_per_output']
    for key, method, shift in [('2_0.5','margin',0),('2_3.0','switch_margin',n)]:
        hist=z[key][method]
        a.hlines(z[key]['target_share'][0],shift,shift+n,color=GRAY,ls=':',lw=1.,
                 label='Max–min target' if shift==0 else None)
        plot(a,hist,'goal_share',TEAL,'Target budget' if shift==0 else None,idx=0,ls='--',shift=shift)
        plot(a,hist,'realized_share_of_B',INK,'Gain-implied cost' if shift==0 else None,idx=0,band=True,shift=shift)
    a.axvline(n,color=GRAY,lw=.8)
    a.text(.04,.93,r'$T=0.5$',transform=a.transAxes,va='top')
    a.text(.56,.93,r'$T=3$',transform=a.transAxes,va='top')
    a.set(title='(a) Two routes: change the readout',ylabel='Slow-route cost / total target',
          xlabel='Decisions per output',xticks=[0,n,2*n],ylim=(.735,.965))
    a.legend(loc='lower right',frameon=False)

    for key,color,ls in [('4_0.35',RUST,'-'),('4_1.6',BLUE,'-'),('8_0.35',RUST,'--'),('8_1.6',BLUE,'--')]:
        number, win = key.split('_')
        plot(b,z[key]['margin'],'allocation_tv',color,f'{number} routes, $T={win}$',ls=ls)
    b.set(title='(b) Error relative to optimal shares',ylabel='Total variation',ylim=(0,.81))
    b.legend(frameon=False,loc='upper right')

    for key,method,color,label,ls in [('4_0.35','margin',RUST,'4 routes, nominal','-'),
        ('4_0.35','slow_actuator',GRAY,'4 routes, slow gain','--'),('8_0.35','margin',BLUE,'8 routes, nominal','-')]:
        plot(c,z[key][method],'abs_cost_error',color,label,ls=ls,scale=100.)
    c.set(title='(c) Cost is tracked, not conserved',ylabel='Absolute total cost error (%)')
    c.legend(frameon=False,loc='upper left')

    for key,color,nm in [('4_0.35',TEAL,'4 routes'),('8_0.35',INK,'8 routes')]:
        for method,ls in [('binary','--'),('margin','-')]:
            plot(d,z[key][method],'frozen_gain_benefit_pp',color,nm+', '+method,ls=ls)
    d.axhline(0,color=GRAY,lw=.7)
    d.set(title=r'(d) Feedback versus benefit, $T=0.35$',ylabel='Worst-output gain (percentage points)')
    d.legend(frameon=False,loc='lower right')

    for ax in (b,c,d):
        ax.set(xscale='log',xlim=(230,150000),xticks=[256,2048,16384,131072],xlabel='Decisions per output')
    for ax in axs.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{int(x):,}'))
        ax.tick_params(axis='both',labelsize=7.3)
    fig.savefig(HERE/'figures'/'fig_prrS_local_gain_loop.pdf',facecolor='white')


if __name__=='__main__':
    main()
