"""Frozen feasibility/pilot analysis; no causal or optimality inference.

Direct HDF5 reader for official Visual Coding NWB; no AllenSDK dependency.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from scipy.linalg import solve_discrete_lyapunov

AREAS = ('VISp','VISl','VISrl','VISal','VISpm','VISam')

def array(g, name):
    a = g[name][:]
    if a.dtype.kind in ('S','O'):
        a = np.asarray([x.decode() if isinstance(x,bytes) else x for x in a])
    return a

def read_trials(f):
    if 'intervals/drifting_gratings_presentations' not in f:
        return None, {'excluded':'no_drifting_gratings'}
    g = f['intervals/drifting_gratings_presentations']
    cols=('start_time','stop_time','orientation','temporal_frequency')
    tab=pd.DataFrame({k:array(g,k) for k in cols})
    for k in cols: tab[k]=pd.to_numeric(tab[k],errors='coerce')
    tab=tab.dropna().reset_index(drop=True)
    if 'intervals/invalid_times' in f:
        inv=f['intervals/invalid_times']
        for a,b in zip(array(inv,'start_time'),array(inv,'stop_time')):
            tab=tab[~((tab.start_time<b)&(tab.stop_time>a))]
    run=f['processing/running/running_speed']
    t, v=run['timestamps'][:],run['data'][:].reshape(-1)
    speeds=[]
    for trial in tab.itertuples():
        a,b=np.searchsorted(t,[trial.start_time,trial.stop_time])
        speeds.append(float(np.nanmean(v[a:b])) if b>a else np.nan)
    tab['speed']=speeds
    tab=tab[np.isfinite(tab.speed)].copy()
    tab['state']=(tab.speed>1).astype(int)
    tab['condition']=[f'{o:g}_{tf:g}' for o,tf in zip(tab.orientation,tab.temporal_frequency)]
    report={'trials_before_matching':len(tab),'counts':tab.groupby('state').size().to_dict()}
    counts=tab.groupby(['condition','state']).size().unstack(fill_value=0).reindex(columns=[0,1],fill_value=0)
    keep=counts.index[(counts>=5).all(axis=1)]
    tab=tab[tab.condition.isin(keep)].reset_index(drop=True)
    report['matched_condition_count']=len(keep)
    report['eligible_counts']=tab.groupby('state').size().to_dict()
    return tab, report

def unit_selection(f):
    u=f['units']; e=f['general/extracellular_ephys/electrodes']
    locations=dict(zip(array(e,'id'),array(e,'location')))
    ids=array(u,'id')
    regions=np.array([locations.get(x,'unknown') for x in array(u,'peak_channel_id')])
    good=np.isin(regions,AREAS)
    for field, threshold, greater in [('presence_ratio',.95,True),('amplitude_cutoff',.1,False),('isi_violations',.5,False)]:
        values=array(u,field)
        good &= np.isfinite(values)&((values>=threshold) if greater else (values<=threshold))
    if 'quality' in u: good &= array(u,'quality')=='good'
    return np.where(good)[0],ids[good],regions[good]

def bin_trials(f,tab,indices,dt=.02):
    # Uniform 2-s gratings only; no concatenation of adjacent trials.
    duration=2.0
    if not np.all(np.abs((tab.stop_time-tab.start_time)-duration)<.06):
        raise ValueError('Unexpected grating duration; requires loader review')
    edges=np.arange(101)*dt
    out=np.empty((len(tab),100,len(indices)),dtype=np.float32)
    ends=f['units/spike_times_index'][:]
    spikes=f['units/spike_times']
    for j,i in enumerate(indices):
        s=spikes[int(ends[i-1]) if i else 0:int(ends[i])]
        for k,start in enumerate(tab.start_time):
            a,b=np.searchsorted(s,[start,start+duration])
            out[k,:,j]=np.histogram(s[a:b]-start,bins=edges)[0]/dt
    return out[:,5:,:]

def pairs(z):
    return z[:,:-1].reshape(-1,z.shape[-1]),z[:,1:].reshape(-1,z.shape[-1])

def fit(z,lam):
    x,y=pairs(z)
    c=x.T@x/len(x)
    # Row-vector fit returned as column-vector propagation A.
    return np.linalg.solve(c+lam*np.eye(c.shape[0]), x.T@y/len(x)).T

def mse(z,A,k=5):
    pred=z[:,:-k]@np.linalg.matrix_power(A,k).T
    return float(np.mean((z[:,k:]-pred)**2))

def select_ridge(z):
    n=len(z); nval=max(1,n//4)
    tr,va=z[:-nval],z[-nval:]
    grid=np.logspace(-5,2,10)
    scores=[mse(va,fit(tr,l),1) for l in grid]
    return float(grid[np.argmin(scores)])

def geometry(A,Q):
    P=np.linalg.matrix_power(A,5)
    u,s,vh=np.linalg.svd(P)
    rho=float(np.max(np.abs(np.linalg.eigvals(A))))
    d={'rho':rho,'squared_gain_100ms':float(s[0]**2),
       'singular_energy_participation':float(np.sum(s*s)**2/np.sum(s**4)),
       's1_over_s2':float(s[0]/s[1]),'stable':rho<1}
    if rho<1:
        sigma=solve_discrete_lyapunov(A,Q)
        residual=np.linalg.norm(sigma-A@sigma@A.T-Q)/max(np.linalg.norm(Q),1e-12)
        d.update(common_drive_stationary_cost=float(np.trace(sigma)),lyapunov_relative_residual=float(residual))
    return d,u[:,0],vh[0]

def analyze_arrays(raw,tab,dim=12,seed=20260929):
    rng=np.random.default_rng(seed)
    trains,tests=[],[]
    for cond,g in tab.groupby('condition',sort=True):
        groups=[g.index[g.state==s].to_numpy() for s in (0,1)]
        n=min(map(len,groups)); nt=max(2,n//3)
        for group in groups:
            chosen=rng.permutation(group)[:n]
            tests.extend(chosen[:nt]); trains.extend(chosen[nt:])
    trains=np.array(trains); tests=np.array(tests)
    resid=raw.copy()
    for (cond,state),g in tab.groupby(['condition','state']):
        tr=np.intersect1d(g.index,trains)
        resid[g.index] -= raw[tr].mean(axis=0)
    scale=resid[trains].reshape(-1,raw.shape[-1]).std(axis=0)
    active=scale>1e-6
    if active.sum()<max(20,dim+1):return {'excluded':'too_few_variable_units'}
    resid=resid[:,:,active]/scale[active]
    x=resid[trains].reshape(-1,active.sum())
    _,_,vh=np.linalg.svd(x,full_matrices=False)
    z=resid@vh[:dim].T
    # Trial ordering randomized independently of states/conditions before inner CV.
    rng.shuffle(trains)
    lp=select_ridge(z[trains]); pooled=fit(z[trains],lp)
    operators=[]; noises=[]; outputs=[]
    for state in (0,1):
        tr=trains[tab.state.iloc[trains].to_numpy()==state]
        te=tests[tab.state.iloc[tests].to_numpy()==state]
        lam=select_ridge(z[tr]); A=fit(z[tr],lam)
        x,y=pairs(z[tr]); yp=x@pooled.T
        scalar=max(0,float(np.sum(yp*y)/np.sum(yp*yp)))
        diagonal=np.maximum(0,np.sum(yp*y,axis=0)/np.maximum(np.sum(yp*yp,axis=0),1e-12))
        q=np.cov((y-x@A.T),rowvar=False)
        operators.append(A); noises.append(q)
        baseline=float(np.mean(z[te,5:]**2))
        vals={name: mse(z[te],mat) for name,mat in
              [('state',A),('pooled',pooled),('scalar',scalar*pooled),('diagonal',diagonal[:,None]*pooled)]}
        outputs.append({'state':'stationary' if state==0 else 'locomotion',
            'train_trials':len(tr),'test_trials':len(te),'ridge':lam,'scalar_gain':scalar,
            'mse_100ms':vals,'zero_predictor_mse':baseline,
            'prediction_R2_vs_zero':{k:1-v/baseline for k,v in vals.items()},
            'activity_covariance_trace':float(np.trace(np.cov(z[tr].reshape(-1,dim),rowvar=False))),
            'innovation_covariance_trace':float(np.trace(q))})
    Qcommon=(noises[0]+noises[1])/2
    gs=[geometry(a,Qcommon) for a in operators]
    for d,g in zip(outputs,gs):d['geometry']=g[0]
    angles={name:float(np.degrees(np.arccos(np.clip(abs(gs[0][i]@gs[1][i]),0,1))))
            for name,i in [('output_angle_deg',1),('input_angle_deg',2)]}
    return {'status':'pilot_only','latent_dim':dim,'states':outputs,**angles,
            'direction_warning':'Descriptive angles only; split-half reliability and refitted shuffle null required.',
            'causal_scope':'Residual dynamics, not measured causal stimulus routing or budget optimality.',
            'unit_count':int(active.sum()),'seed':seed}

def process(path,out):
    with h5py.File(path,'r') as f:
        tab,rep=read_trials(f)
        idx,ids,regions=unit_selection(f)
        rep.update(session=path.stem,units=len(idx),areas={str(a):int((regions==a).sum()) for a in np.unique(regions)})
        if tab is None or len(idx)<40 or len(np.unique(regions))<2 or min(rep.get('eligible_counts',{}).get(s,0) for s in (0,1))<20:
            rep['status']='excluded_by_feasibility_gate'
        else:
            raw=bin_trials(f,tab,idx)
            np.savez_compressed(out/(path.stem+'_trials.npz'),rates=raw,unit_ids=ids,regions=regions,
                start_time=tab.start_time.to_numpy(),state=tab.state.to_numpy(),condition=tab.condition.to_numpy(dtype=str))
            tab.to_csv(out/(path.stem+'_trials.csv'),index=False)
            rep['analysis']=analyze_arrays(raw,tab)
    (out/(path.stem+'_report.json')).write_text(json.dumps(rep,indent=2,allow_nan=False))
    print(json.dumps(rep),flush=True)

def selftest():
    rng=np.random.default_rng(4); d=24
    mats=[.75*np.eye(d),.75*np.eye(d)]
    mats[0][0,1]=.7; mats[1][2,3]=.7
    trials=[]; rows=[]
    for state in (0,1):
        for n in range(100):
            z=np.zeros((100,d))
            for t in range(1,100):z[t]=mats[state]@z[t-1]+rng.normal(size=d)*.15
            trials.append(z); rows.append({'condition':'test','state':state})
    result=analyze_arrays(np.array(trials),pd.DataFrame(rows))
    json.dumps(result,allow_nan=False)
    for s in result['states']:
        assert s['mse_100ms']['state']<s['mse_100ms']['pooled']
        assert s['geometry']['lyapunov_relative_residual']<1e-8
    assert result['input_angle_deg']>50
    print('PASS: known state-dependent channels, held-out advantage, Lyapunov residual; synthetic only')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('nwb',nargs='?',type=Path)
    p.add_argument('--out',type=Path,default=Path('/Volumes/MimiHDD/energy_nonnormal_data/allen_visual_coding/analysis'))
    p.add_argument('--selftest',action='store_true');a=p.parse_args()
    if a.selftest:selftest()
    else:
        a.out.mkdir(parents=True,exist_ok=True);process(a.nwb,a.out)
