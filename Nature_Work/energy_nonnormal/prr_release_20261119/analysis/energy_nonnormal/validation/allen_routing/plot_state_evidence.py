"""Export an independent, auditable figure; never replaces manuscript artwork."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from state_inference import ROOT, matching


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--results',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args()
    results=a.results or a.root/'state_inference_final'
    out=a.out or Path(__file__).parent/'outputs_20261002'
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((results/'cohort_summary.json').read_text())
    ids=summary['local_300s']['sessions']
    rows=[json.loads((results/(s+'_inference.json')).read_text()) for s in ids]
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(7.1,4.8),layout='constrained')
    ax=axs[0,0]
    example=ids[0]  # Outcome-independent: lowest session ID in local cohort.
    with np.load(a.root/'analysis'/(example+'_trials.npz')) as f:raw=f['rates']
    tab=pd.read_csv(a.root/'analysis'/(example+'_trials.csv'));paired,_=matching(tab,300)
    time=.11+np.arange(raw.shape[1])*.02
    for state,label,color,style in [(0,'Stationary','#2269B3','-'),(1,'Locomotion','#D14A39','--')]:
        rate=raw[paired[:,state]].mean(axis=2)
        mean=rate.mean(axis=0);sem=rate.std(axis=0,ddof=1)/np.sqrt(len(rate))
        ax.plot(time,mean,color=color,linestyle=style,label=label,lw=1.3)
        ax.fill_between(time,mean-sem,mean+sem,color=color,alpha=.15,lw=0)
    ax.set(title='(a) Same recorded population',xlabel='Time from grating onset (s)',ylabel='Mean firing rate (spikes/s)')
    ax.legend(frameon=False,loc='upper right')
    ax.text(.02,.03,f'Example: {raw.shape[2]} units, {len(paired)} pairs',transform=ax.transAxes,fontsize=7)
    ax=axs[0,1]
    values=np.array([[r['local_300s']['delta_R2'][c]*100 for c in ('pooled','scalar','diagonal')] for r in rows])
    for v in values:ax.plot(range(3),v,color='.78',lw=.6,zorder=1)
    rng=np.random.default_rng(34)
    colors=['#3D6EAD','#B54831','#7B5C9B']
    for i,c in enumerate(('pooled','scalar','diagonal')):
        ax.scatter(i+rng.uniform(-.06,.06,len(values)),values[:,i],s=15,color=colors[i],edgecolors='white',lw=.3,zorder=2)
        ax.plot([i-.13,i+.13],[np.median(values[:,i])]*2,color='black',lw=1.5,zorder=3)
    ax.axhline(0,color='.4',lw=.8,ls=':')
    ax.set_xticks(range(3),['Common\noperator','Scalar\ngain','Diagonal\ngain'])
    ax.set(title='(b) State models improve held-out prediction',ylabel=r'$\Delta R^2$ at 100 ms (percentage points)')
    ax.set_ylim(min(-.24,float(values.min())-.1),float(values.max())+.38)
    ax.text(.02,.96,f'{len(rows)} mice; dots = mice; bars = medians',va='top',transform=ax.transAxes,fontsize=6.8)
    ax=axs[1,0]
    for i,r in enumerate(rows):
        null=np.array(r['refitted_nulls']['normalized_map_distance'])
        lo,hi=np.quantile(null,[.025,.975]);median=np.median(null)
        ax.plot([lo,hi],[i,i],color='.6',lw=2,solid_capstyle='round')
        ax.plot(median,i,'o',color='.55',ms=2.5)
        ax.plot(r['local_300s']['geometry']['normalized_map_distance'],i,'D',color='#BB532F',ms=3.5)
    ax.plot([],[],'-',color='.6',lw=2,label='Refitted null, 95% interval')
    ax.plot([],[],'D',color='#BB532F',ms=3.5,label='Observed state difference')
    ax.set_yticks(range(len(rows)),[str(i+1) for i in range(len(rows))],fontsize=6.5)
    ax.set(title='(c) Finite-time maps differ beyond scalar gain',xlabel='Gain-normalized map distance',ylabel='Mouse')
    ax.set_xlim(0,1.65);ax.set_ylim(-1,len(rows)+2.2)
    ax.legend(frameon=False,loc='upper left',fontsize=6.3)
    ax=axs[1,1]
    names=['8D','12D','16D','Time-block\nheld out','Mean rate\nremoved']
    stats=[summary['dimensions']['8']['scalar'],summary['local_300s']['scalar'],
           summary['dimensions']['16']['scalar'],summary['temporal_folds']['scalar'],
           summary['population_mean_removed']['scalar']]
    for i,s in enumerate(stats):
        m=s['median']*100;lo,hi=np.array(s['median_bootstrap_CI95'])*100
        ax.errorbar(i,m,yerr=np.array([[m-lo],[hi-m]]),color='#B54831',marker='o',ms=4,lw=1,capsize=2)
    ax.axhline(0,color='.4',lw=.8,ls=':')
    ax.set_xticks(range(len(names)),names,fontsize=7)
    ax.set(title='(d) Advantage over scalar gain is robust',ylabel=r'Median $\Delta R^2$ (percentage points)')
    ax.text(.03,.98,'Animal-bootstrap 95% intervals',transform=ax.transAxes,va='top',fontsize=6.8)
    fig.savefig(out/'fig_allen_state_routing.pdf')
    fig.savefig(out/'fig_allen_state_routing.png',dpi=220)
    plt.close(fig)
    # Only derived summaries are exported to the repository, not raw recordings.
    (out/'cohort_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    pupil=a.root/'pupil_robustness'/'pupil_summary.json'
    if pupil.exists():
        (out/'pupil_summary.json').write_text(pupil.read_text())
    table=[]
    for i,r in enumerate(rows):
        g=r['local_300s']['geometry'];stat=r['local_300s']['delta_R2']
        table.append({'figure_mouse':i+1,'session':r['session'],'locally_matched_pairs':r['local_pair_count'],
                      **{f'delta_R2_{k}':v for k,v in stat.items()},
                      'map_distance':g['normalized_map_distance'],
                      'independent_half_cosine':g['split_half_difference_cosine'],
                      'rho_stationary':g['rho'][0],'rho_locomotion':g['rho'][1]})
    pd.DataFrame(table).to_csv(out/'mouse_level_results.csv',index=False)
    print(out.resolve())


if __name__=='__main__':main()
