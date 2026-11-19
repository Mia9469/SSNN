"""Secondary pupil matching and train-only additive pupil adjustment.

Added after discovering eye traces in later NWBs; report as exploratory.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from state_inference import ROOT, SEEDS, matching, matched_fit, split_pairs, summary_stats


def pupil_data(nwb, tab):
    with h5py.File(nwb,'r') as f:
        name='processing/filtered_gaze_mapping/pupil_area'
        if name not in f:return None
        g=f[name];timestamps=g['timestamps'][:];values=g['data'][:]
    mean=[];coverage=[];trace=[]
    for r in tab.itertuples():
        a,b=np.searchsorted(timestamps,[r.start_time,r.stop_time])
        good=np.isfinite(values[a:b]) & (values[a:b]>0)
        coverage.append(float(np.mean(good)) if b>a else 0)
        mean.append(float(np.nanmean(np.where(good,values[a:b],np.nan))) if np.any(good) else np.nan)
        bins=[]
        for j in range(95):
            a,b=np.searchsorted(timestamps,[r.start_time+.1+j*.02,r.start_time+.12+j*.02])
            v=values[a:b];ok=np.isfinite(v)&(v>0)
            bins.append(float(np.mean(v[ok])) if np.any(ok) else np.nan)
        trace.append(bins)
    return np.array(mean),np.array(coverage),np.array(trace)


def pupil_matching(tab,mean,coverage):
    t=tab.copy();t['block']=((t.start_time-t.start_time.min())//300).astype(int)
    valid=np.isfinite(mean)&(coverage>=.8)
    scale=np.std(mean[valid])
    if scale<=0:return np.empty((0,2),int),t,None
    z=(mean-np.mean(mean[valid]))/scale
    rows=[]
    for _,g in t.groupby(['condition','block']):
        ids=[g.index[(g.state==s)&valid[g.index]].to_numpy() for s in (0,1)]
        if not all(map(len,ids)):continue
        distance=np.abs(z[ids[0],None]-z[ids[1]][None,:])
        dt=np.abs(t.start_time.iloc[ids[0]].to_numpy()[:,None]-t.start_time.iloc[ids[1]].to_numpy()[None,:])/300
        cost=np.where(distance<=.5,distance+dt*.05,1e6)
        a,b=linear_sum_assignment(cost)
        rows.extend((ids[0][i],ids[1][j]) for i,j in zip(a,b) if distance[i,j]<=.5)
    paired=np.asarray(rows,int).reshape(-1,2)
    if len(paired):
        counts=t.condition.iloc[paired[:,0]].value_counts()
        paired=paired[np.isin(t.condition.iloc[paired[:,0]],counts.index[counts>=3])]
    return paired,t,z


def additive_adjust(raw,tab,ids,trpair,trace):
    train=ids[trpair].reshape(-1)
    trace=trace.copy()
    mean=np.nanmean(trace[train]);sd=np.nanstd(trace[train])
    if not np.isfinite(sd) or sd<=0:raise ValueError('Pupil has no training variance')
    # Missing individual bins use that trial's measured mean, with a missingness flag.
    missing=~np.isfinite(trace)
    trialmean=np.nanmean(trace,axis=1)
    trace[missing]=np.broadcast_to(trialmean[:,None],trace.shape)[missing]
    z=(trace-mean)/sd
    covariates=np.stack([np.ones_like(z),z,z*z,missing.astype(float)],axis=-1)
    residual=raw.copy()
    for _,g in tab.groupby(['condition','state']):
        ix=np.intersect1d(g.index,train)
        if not len(ix):continue
        residual[g.index]-=raw[ix].mean(axis=0)
    X=covariates[train].reshape(-1,4);Y=residual[train].reshape(-1,raw.shape[-1])
    beta=np.linalg.lstsq(X,Y,rcond=None)[0]
    adjusted=raw-covariates@beta
    if not np.isfinite(adjusted).all():raise FloatingPointError('Nonfinite pupil adjustment')
    return adjusted


def main():
    out=ROOT/'pupil_robustness';out.mkdir(exist_ok=True)
    rows=[]
    with np.errstate(all='ignore'):
        for file in sorted((ROOT/'analysis').glob('session_*_trials.csv')):
            tab=pd.read_csv(file);stem=file.stem.replace('_trials','')
            info=pupil_data(ROOT/'nwb'/(stem+'.nwb'),tab)
            r={'session':stem,'scope':'Exploratory pupil checks; no causal state manipulation or complete arousal adjustment'}
            if info is None:
                r['status']='no_pupil_trace';rows.append(r);continue
            mean,coverage,trace=info
            r['pupil_valid_trials']=int(sum(np.isfinite(mean)&(coverage>=.8)))
            with np.load(file.with_suffix('.npz')) as f:raw=f['rates'].astype(float)
            paired,t,z=pupil_matching(tab,mean,coverage)
            r['pupil_matched_pairs']=len(paired)
            if len(paired)>=20 and t.condition.iloc[paired[:,0]].nunique()>=2:
                tr,te=split_pairs(paired,t,SEEDS[0])
                r['pupil_matched']=matched_fit(raw,t,paired,tr,te,12,SEEDS[0],False)
                r['pupil_z_mean_absolute_pair_difference']=float(np.mean(abs(z[paired[:,0]]-z[paired[:,1]])))
            # Keep the original local matching, then remove only invalid pupil trials.
            clean=tab[np.isfinite(mean)&(coverage>=.8)].index.to_numpy()
            reduced=tab.iloc[clean].reset_index(drop=True)
            ids,t2=matching(reduced,300)
            r['additive_adjustment_pairs']=len(ids)
            if len(ids)>=20 and t2.condition.iloc[ids[:,0]].nunique()>=2:
                data=raw[clean];pupil=trace[clean]
                tr,te=split_pairs(ids,t2,SEEDS[0])
                r['pupil_available_unadjusted']=matched_fit(data,t2,ids,tr,te,12,SEEDS[0],False)
                adjusted=additive_adjust(data,t2,ids,tr,pupil)
                r['additive_pupil_adjusted']=matched_fit(adjusted,t2,ids,tr,te,12,SEEDS[0],False)
            r['status']='checked';rows.append(r)
            print(stem,'pupil matched',len(paired),'adjustment pairs',len(ids),flush=True)
    summary={'sessions_with_pupil':sum(r['status']=='checked' for r in rows),
             'missing_pupil_sessions':[r['session'] for r in rows if r['status']=='no_pupil_trace'],
             'matching_caliper':'0.5 SD of session trial-mean pupil area, >=80% coverage, same stimulus/300-s block',
             'status':'Secondary exploratory robustness, not independent confirmation',
             'pupil_matched':{},'pupil_available_unadjusted':{},'additive_pupil_adjusted':{},'sessions':rows}
    for name in ('pupil_matched','pupil_available_unadjusted','additive_pupil_adjusted'):
        for c in ('pooled','scalar','diagonal'):
            summary[name][c]=summary_stats([r[name]['delta_R2'][c] for r in rows if name in r])
    (out/'pupil_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in summary.items() if k!='sessions'},indent=2),flush=True)


if __name__=='__main__':main()
