"""Held-out state prediction and locally matched, fully refitted nulls.

See INFERENCE_PLAN.md. Observational effective maps, not causal routing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon

from analyze_session import analyze_arrays, fit, mse, pairs, select_ridge

ROOT = Path('/Volumes/MimiHDD/energy_nonnormal_data/allen_visual_coding')
SEEDS = [20260929, 20260930, 20261001, 20261002, 20261003]


def scalar_cos(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denom) if denom > 1e-20 else 0.0


def prediction_delta(result, control):
    return float(np.mean([(s['mse_100ms'][control] - s['mse_100ms']['state']) /
                          s['zero_predictor_mse'] for s in result['states']]))


def matching(tab, window):
    """Nearest-time matching within stimulus and fixed acquisition block."""
    table = tab.copy()
    table['block'] = ((table.start_time - table.start_time.min()) // window).astype(int)
    matched = []
    for (_, _), group in table.groupby(['condition', 'block'], sort=True):
        ix = [group.index[group.state == s].to_numpy() for s in (0, 1)]
        if not all(len(x) for x in ix):
            continue
        times = [table.start_time.loc[x].to_numpy() for x in ix]
        i, j = linear_sum_assignment(np.abs(times[0][:, None] - times[1][None, :]))
        matched.extend(zip(ix[0][i], ix[1][j]))
    ids = np.asarray(matched, dtype=int).reshape(-1, 2)
    if not len(ids):
        return ids, table
    counts = table.condition.iloc[ids[:, 0]].value_counts()
    keep = counts.index[counts >= 3]
    ids = ids[np.isin(table.condition.iloc[ids[:, 0]], keep)]
    return ids, table


def split_pairs(ids, tab, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    cond = tab.condition.iloc[ids[:, 0]].to_numpy()
    for c in sorted(set(cond)):
        ix = rng.permutation(np.flatnonzero(cond == c))
        nte = max(1, len(ix) // 3)
        te.extend(ix[:nte]); tr.extend(ix[nte:])
    return np.asarray(tr), np.asarray(te)


def representation(raw, tab, train, dim):
    """Common scale and PCA fitted exclusively to training residuals."""
    resid = np.asarray(raw, dtype=np.float64).copy()
    for (_, _), group in tab.groupby(['condition', 'state']):
        ix = np.intersect1d(group.index.to_numpy(), train)
        if not len(ix):
            raise ValueError('A condition/state has no training template')
        resid[group.index] -= raw[ix].mean(axis=0)
    x = resid[train].reshape(-1, raw.shape[-1])
    scale = x.std(axis=0)
    active = scale > 1e-6
    if active.sum() < max(20, dim):
        raise ValueError('Too few variable units')
    resid = resid[:, :, active] / scale[active]
    x = resid[train].reshape(-1, active.sum())
    covariance = x.T @ x / len(x)
    eig, vec = np.linalg.eigh(covariance)
    basis = vec[:, -dim:][:, ::-1].T
    z = resid @ basis.T
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite PCA coordinates')
    return z, basis, active, scale


def half_difference(raw, tab, ids, train_pair, basis, active, scale, lambdas, seed):
    rng = np.random.default_rng(seed)
    cond = tab.condition.iloc[ids[train_pair, 0]].to_numpy()
    halves = [[], []]
    for c in sorted(set(cond)):
        ix = rng.permutation(train_pair[cond == c])
        # A single trial's own full PSTH would remove that trial identically.
        # Use only conditions supporting >=2 independent trials/state/half.
        if len(ix) < 4:
            continue
        for half in (0, 1):
            halves[half].extend(ix[half::2])
    if min(map(len,halves)) < 5:
        return None
    # Independent centering of the halves, in common training-defined coordinates.
    projected = (raw[:, :, active] / scale[active]) @ basis.T
    differences = []
    for hp in halves:
        hids = ids[np.asarray(hp)]
        Ps = []
        for s in (0, 1):
            selected = hids[:, s]
            z = projected[selected].copy()
            conditions = tab.condition.iloc[selected].to_numpy()
            for c in sorted(set(conditions)):
                j = np.flatnonzero(conditions == c)
                z[j] -= z[j].mean(axis=0)
            P = np.linalg.matrix_power(fit(z, lambdas[s]), 5)
            Ps.append(P / max(np.linalg.norm(P), 1e-20))
        differences.append(Ps[1] - Ps[0])
    return scalar_cos(*differences)


def matched_fit(raw, tab, ids, train_pair, test_pair, dim, seed,
                geometry=True, regions=None):
    # Subset contains exactly matched pairs; each pair stays in one split.
    original = ids.reshape(-1)
    data = raw[original]
    t = tab.iloc[original].reset_index(drop=True).copy()
    local_ids = np.arange(len(original)).reshape(-1, 2)
    train = local_ids[train_pair].reshape(-1)
    test = local_ids[test_pair].reshape(-1)
    z, basis, active, scale = representation(data, t, train, dim)
    rng = np.random.default_rng(seed)
    shuffled_train = rng.permutation(train)
    pooled = fit(z[shuffled_train], select_ridge(z[shuffled_train]))
    ops, lambdas, states = [], [], []
    for state in (0, 1):
        tr = train[t.state.iloc[train].to_numpy() == state]
        te = test[t.state.iloc[test].to_numpy() == state]
        tr = rng.permutation(tr)
        lam = select_ridge(z[tr]); A = fit(z[tr], lam)
        x, y = pairs(z[tr]); yp = x @ pooled.T
        scalar = max(0.0, float(np.sum(yp * y) / max(np.sum(yp * yp), 1e-20)))
        diag = np.maximum(0, np.sum(yp * y, axis=0) /
                          np.maximum(np.sum(yp * yp, axis=0), 1e-20))
        mats = {'state': A, 'pooled': pooled, 'scalar': scalar * pooled,
                'diagonal': diag[:, None] * pooled}
        zero = float(np.mean(z[te, 5:] ** 2))
        states.append({'state': state, 'train_trials': len(tr), 'test_trials': len(te),
                       'ridge': lam, 'zero_mse': zero,
                       'mse': {k: {name: mse(z[te], m, int(k)) for name, m in mats.items()}
                               for k in ('1', '5')}})
        ops.append(A); lambdas.append(lam)
    result = {'delta_R2': {c: float(np.mean([(s['mse']['5'][c] - s['mse']['5']['state']) /
                             s['zero_mse'] for s in states])) for c in ('pooled','scalar','diagonal')},
              'states': states, 'train_pairs': len(train_pair), 'test_pairs': len(test_pair)}
    if geometry:
        Ps = [np.linalg.matrix_power(A, 5) for A in ops]
        svd = [np.linalg.svd(P) for P in Ps]
        norm = [P / max(np.linalg.norm(P), 1e-20) for P in Ps]
        geom = {'map_cosine': scalar_cos(*Ps),
                'normalized_map_distance': float(np.linalg.norm(norm[1] - norm[0])),
                'input_angle_deg': float(np.degrees(np.arccos(np.clip(abs(svd[0][2][0] @ svd[1][2][0]),0,1)))),
                'output_angle_deg': float(np.degrees(np.arccos(np.clip(abs(svd[0][0][:,0] @ svd[1][0][:,0]),0,1)))),
                'rank3_input_angles_deg': np.degrees(subspace_angles(svd[0][2][:3].T,svd[1][2][:3].T)).tolist(),
                'rank3_output_angles_deg': np.degrees(subspace_angles(svd[0][0][:,:3],svd[1][0][:,:3])).tolist(),
                'split_half_difference_cosine': half_difference(data,t,local_ids,train_pair,basis,active,scale,lambdas,seed+11),
                'rho': [float(max(abs(np.linalg.eigvals(A)))) for A in ops],
                'gain_squared_100ms': [float(v[1][0]**2) for v in svd],
                's1_over_s2': [float(v[1][0]/max(v[1][1],1e-20)) for v in svd],
                'singular_energy_participation': [float(sum(v[1]**2)**2/sum(v[1]**4)) for v in svd]}
        if regions is not None:
            reg = regions[active]
            if 'VISp' in reg:
                B = basis[:, reg == 'VISp']
                area_energy = {}
                for area in sorted(set(reg) - {'VISp'}):
                    C = basis[:, reg == area].T
                    area_energy[area] = [float(np.sum((C @ P @ B)**2) / len(C)) for P in Ps]
                geom['V1_to_area_model_energy_per_output_unit'] = area_energy
                geom['area_map_scope'] = 'Model impulse maps in standardized common coordinates; not measured causal transfer'
        result['geometry'] = geom
    json.dumps(result, allow_nan=False)
    return result


def temporal_folds(raw, tab, ids, dim):
    blocks = tab.block.iloc[ids[:, 0]].to_numpy()
    conditions = tab.condition.iloc[ids[:,0]].to_numpy()
    out = []
    for fold in range(3):
        train = np.flatnonzero(blocks % 3 != fold)
        test = np.flatnonzero(blocks % 3 == fold)
        counts = pd.Series(conditions[train]).value_counts()
        supported = counts.index[counts >= 2]
        keep = np.flatnonzero(np.isin(conditions, supported))
        selected = ids[keep]
        newtrain = np.flatnonzero(np.isin(keep, train))
        newtest = np.flatnonzero(np.isin(keep, test))
        if len(newtrain) >= 10 and len(newtest) >= 5:
            r = matched_fit(raw, tab, selected, newtrain, newtest, dim, SEEDS[0]+fold, False)
            out.append(r)
    return out


def p_plus_one(obs, null):
    return float((1 + np.sum(np.asarray(null) >= obs)) / (len(null) + 1))


def run_session(path, args):
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    with np.load(path, allow_pickle=False) as f:
        raw = f['rates'].astype(np.float64)
        regions = f['regions'].astype(str)
    tab = pd.read_csv(path.with_suffix('.csv'))
    out = {'session': path.stem.replace('_trials',''), 'unit_count': raw.shape[2],
           'scope': 'Exploratory observational state dependence; no budget optimization or causal gain identification'}
    # Accelerate on this machine raises spurious matmul flags; reject nonfinite outputs.
    with np.errstate(all='ignore'):
        repeats = []
        for seed in SEEDS:
            r = analyze_arrays(raw, tab, 12, seed)
            repeats.append({'seed': seed, 'delta_R2': {c: prediction_delta(r,c) for c in ('pooled','scalar','diagonal')},
                            'states':r['states'], 'input_angle_deg':r['input_angle_deg'],'output_angle_deg':r['output_angle_deg']})
        out['original_protocol_repeated'] = repeats
        ids,t = matching(tab,300)
        out['local_pair_count'] = len(ids)
        out['local_condition_count'] = int(t.condition.iloc[ids[:,0]].nunique()) if len(ids) else 0
        if len(ids) >= 20 and out['local_condition_count'] >= 2:
            tr,te = split_pairs(ids,t,SEEDS[0])
            out['local_300s'] = matched_fit(raw,t,ids,tr,te,12,SEEDS[0],True,regions)
            out['local_300s_repeats'] = [matched_fit(raw,t,ids,*split_pairs(ids,t,seed),12,seed,True)
                                       for seed in SEEDS[1:]]
            out['dimensions'] = {str(d): matched_fit(raw,t,ids,tr,te,d,SEEDS[0],False) for d in (8,16)}
            out['population_mean_removed'] = matched_fit(raw-raw.mean(axis=2,keepdims=True),t,ids,tr,te,12,SEEDS[0],False)
            out['temporal_block_folds'] = temporal_folds(raw,t,ids,12)
            null = {'delta_R2':{c:[] for c in ('pooled','scalar','diagonal')},
                    'normalized_map_distance':[], 'split_half_difference_cosine':[]}
            rng = np.random.default_rng(int(out['session'].split('_')[-1]) + SEEDS[0])
            for i in range(args['permutations']):
                perm = t.copy()
                swapped = ids[rng.random(len(ids)) < .5]
                for a,b in swapped:
                    perm.loc[a,'state'],perm.loc[b,'state'] = perm.loc[b,'state'],perm.loc[a,'state']
                # local_ids columns track original pairs, not new state labels.
                ordered = ids.copy()
                flipped = perm.state.iloc[ordered[:,0]].to_numpy() == 1
                ordered[flipped] = ordered[flipped,::-1]
                r = matched_fit(raw,perm,ordered,tr,te,12,SEEDS[0],True)
                for c in null['delta_R2']:null['delta_R2'][c].append(r['delta_R2'][c])
                for c in ('normalized_map_distance','split_half_difference_cosine'):null[c].append(r['geometry'][c])
                if (i+1)%25 == 0: print(out['session'],'null',i+1,flush=True)
            out['refitted_nulls'] = null
            obs = out['local_300s']
            out['permutation_p'] = {c:p_plus_one(obs['delta_R2'][c],null['delta_R2'][c]) for c in null['delta_R2']}
            out['geometry_permutation_p'] = {c:p_plus_one(obs['geometry'][c],null[c])
                                           for c in ('normalized_map_distance','split_half_difference_cosine')
                                           if obs['geometry'][c] is not None}
        else:
            out['local_300s_exclusion'] = 'Fewer than 20 locally matched pairs or two conditions after training-template support'
        ids600,t600 = matching(tab,600)
        out['local_600s_pair_count'] = len(ids600)
        if len(ids600)>=20 and t600.condition.iloc[ids600[:,0]].nunique()>=2:
            out['local_600s'] = matched_fit(raw,t600,ids600,*split_pairs(ids600,t600,SEEDS[0]),12,SEEDS[0],False)
    out['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    out['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out['null_count'] = args['permutations']
    json.dumps(out,allow_nan=False)
    return out


def summary_stats(values):
    x=np.asarray(values,dtype=float)
    if not len(x):return {'n':0}
    rng=np.random.default_rng(20261002)
    boot=np.median(rng.choice(x,(10000,len(x))),axis=1)
    return {'n':len(x),'positive':int(sum(x>0)),'median':float(np.median(x)),
            'median_bootstrap_CI95':np.quantile(boot,[.025,.975]).tolist(),
            'wilcoxon_two_sided':float(wilcoxon(x).pvalue) if np.any(x!=0) else 1.0}


def aggregate(outdir,root):
    reports=[json.loads(p.read_text()) for p in sorted(outdir.glob('session_*_inference.json')) if not p.name.startswith('.')]
    meta=pd.read_csv(root/'sessions.csv').set_index('id')
    for r in reports:r['specimen_id']=int(meta.loc[int(r['session'].split('_')[-1]),'specimen_id'])
    assert len(set(r['specimen_id'] for r in reports))==len(reports),'Animal aggregation needed'
    summary={'sessions_analyzed':len(reports),'distinct_mice':len(reports),'null_per_session':sorted(set(r['null_count'] for r in reports)),
             'scope':'Associational effective residual geometry; not causal routing, matched spectra/costs, or budget optimization',
             'original':{},'local_300s':{},'local_600s':{},'dimensions':{},'population_mean_removed':{},'temporal_folds':{}}
    controls=('pooled','scalar','diagonal')
    matched=[r for r in reports if 'local_300s' in r]
    for c in controls:
        summary['original'][c]=summary_stats([np.mean([q['delta_R2'][c] for q in r['original_protocol_repeated']]) for r in reports])
        summary['local_300s'][c]=summary_stats([r['local_300s']['delta_R2'][c] for r in matched])
        summary['local_600s'][c]=summary_stats([r['local_600s']['delta_R2'][c] for r in reports if 'local_600s' in r])
        summary['population_mean_removed'][c]=summary_stats([r['population_mean_removed']['delta_R2'][c] for r in matched])
        summary['temporal_folds'][c]=summary_stats([np.mean([q['delta_R2'][c] for q in r['temporal_block_folds']])
                                                 for r in matched if r['temporal_block_folds']])
        for d in ('8','16'):
            summary['dimensions'].setdefault(d,{})[c]=summary_stats([r['dimensions'][d]['delta_R2'][c] for r in matched])
        if matched and all(r['null_count']>0 for r in matched):
            # Independent within-mouse swaps; pooled group statistic gives each mouse one vote.
            observed=np.median([r['local_300s']['delta_R2'][c] for r in matched])
            null=np.median([r['refitted_nulls']['delta_R2'][c] for r in matched],axis=0)
            summary['local_300s'][c].update(group_permutation_p=p_plus_one(observed,null),
                                          group_null_median=float(np.median(null)))
    summary['geometry']={k:summary_stats([r['local_300s']['geometry'][k] for r in matched
                                        if r['local_300s']['geometry'][k] is not None])
                         for k in ('normalized_map_distance','split_half_difference_cosine','input_angle_deg','output_angle_deg')}
    # Distances/angles are nonnegative by definition; testing them against zero
    # is not evidence of a state effect. Their refitted null supplies the test.
    for k in ('normalized_map_distance','input_angle_deg','output_angle_deg'):
        summary['geometry'][k].pop('wilcoxon_two_sided',None)
        summary['geometry'][k].pop('positive',None)
    if matched and all(r['null_count']>0 for r in matched):
        for k in ('normalized_map_distance','split_half_difference_cosine'):
            available=[r for r in matched if r['local_300s']['geometry'][k] is not None]
            if available:
                obs=np.median([r['local_300s']['geometry'][k] for r in available]);null=np.median([r['refitted_nulls'][k] for r in available],axis=0)
                summary['geometry'][k]['group_permutation_p']=p_plus_one(obs,null)
        summary['animal_FDR']={}
        for c in controls:
            ps=np.array([r['permutation_p'][c] for r in matched]);order=np.argsort(ps)
            q=np.minimum.accumulate((ps[order]*len(ps)/np.arange(1,len(ps)+1))[::-1])[::-1]
            qs=np.empty_like(q);qs[order]=np.minimum(q,1)
            summary['animal_FDR'][c]={r['session']:{'p':float(p),'q':float(q)} for r,p,q in zip(matched,ps,qs)}
    summary['local_300s']['sessions']=[r['session'] for r in matched]
    primary_p=np.array([summary['local_300s'][c].get('group_permutation_p',np.nan)
                        for c in ('pooled','scalar')])
    if np.isfinite(primary_p).all():
        order=np.argsort(primary_p)
        corrected=np.maximum.accumulate(primary_p[order]*np.array([2,1]))
        for i,rank in enumerate(order):
            summary['local_300s'][('pooled','scalar')[rank]]['group_permutation_Holm_two_controls']=float(min(1,corrected[i]))
    summary['cohort']=[{'session':r['session'],'mouse':r['specimen_id'],'unit_count':r['unit_count'],
                        'local_pairs':r['local_pair_count'],'local_eligible':'local_300s' in r} for r in reports]
    exclusions=[]
    for p in sorted((root/'analysis').glob('*_report.json')):
        if p.name.startswith('.'):continue
        r=json.loads(p.read_text())
        if 'analysis' not in r:exclusions.append(r)
    summary['original_feasibility_exclusions']=exclusions
    (outdir/'cohort_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary


def selftest():
    rng=np.random.default_rng(12);raw=[];rows=[]
    for i in range(60):
        for state in (0,1):
            A=.8*np.eye(24)
            for channel in range(12):
                a,b=2*channel,2*channel+1
                A[a,b] = .6 if state==0 else 0
                A[b,a] = .6 if state==1 else 0
            x=np.zeros((95,24))
            for j in range(1,95):x[j]=A@x[j-1]+rng.normal(size=24)*.15
            raw.append(x);rows.append({'condition':str(i%3),'state':state,'start_time':float(6*i+state)})
    raw=np.array(raw);tab=pd.DataFrame(rows);ids,t=matching(tab,300);tr,te=split_pairs(ids,t,SEEDS[0])
    r=matched_fit(raw,t,ids,tr,te,24,SEEDS[0],True)
    assert r['delta_R2']['scalar']>.01,r
    assert r['geometry']['split_half_difference_cosine']>0,r
    r2=matched_fit(raw,t,ids,tr,te,24,SEEDS[0],True)
    assert r==r2
    assert len(set(ids[tr].reshape(-1)) & set(ids[te].reshape(-1)))==0
    identical=raw.copy();identical[1::2]=identical[0::2]
    same=matched_fit(identical,t,ids,tr,te,24,SEEDS[0],True)
    assert abs(same['delta_R2']['pooled'])<1e-10,same
    assert same['geometry']['normalized_map_distance']<1e-8,same
    print('PASS synthetic distinct channels, held-out gain, independent difference reliability, determinism, trial isolation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--permutations',type=int,default=999)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--selftest',action='store_true')
    p.add_argument('--aggregate-only',action='store_true');a=p.parse_args()
    warnings.filterwarnings('ignore',message='.*encountered in matmul',category=RuntimeWarning)
    if a.selftest:selftest();return
    out=a.out or a.root/'state_inference';out.mkdir(parents=True,exist_ok=True)
    if not a.aggregate_only:
        files=[x for x in sorted((a.root/'analysis').glob('session_*_trials.npz')) if not x.name.startswith('.')]
        if a.limit:files=files[:a.limit]
        pending=[]
        for path in files:
            dest=out/(path.stem.replace('_trials','')+'_inference.json')
            if dest.exists() and json.loads(dest.read_text()).get('null_count')==a.permutations:continue
            pending.append(path)
        print('pending',len(pending),'nulls',a.permutations,flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(run_session,f,{'permutations':a.permutations}):f for f in pending}
            for future in as_completed(jobs):
                r=future.result();dest=out/(r['session']+'_inference.json')
                dest.write_text(json.dumps(r,indent=2,allow_nan=False))
                print('COMPLETED',r['session'],'local pairs',r['local_pair_count'],flush=True)
    s=aggregate(out,a.root)
    print(json.dumps({k:s[k] for k in ('sessions_analyzed','original','local_300s','geometry')},indent=2),flush=True)


if __name__=='__main__':main()
