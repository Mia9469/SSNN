"""Sparse budget exchange, online activity cost estimates, and actual gain dynamics.

Fixed protocol: Fig. 3 pair and first 4-/8-pathway networks, both original
windows, 48 replicates, 4096 batches of 32 decisions per route. Continuous
linear Gaussian dynamics are sampled exactly across windows and blank gaps.
No neural state is reinitialized when a gain or readout window changes.
"""
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.linalg import block_diag, expm
from scipy.special import ndtr

from feedback_allocation import B as PAIR_B, X0, env
from strengthening_analysis import covariance

HERE = Path(__file__).resolve().parent
SEED, REP, STEPS, BATCH, WARM = 20261113, 48, 4096, 32, 4096
GAP, TAU, GAMMA = 1., 256., .5
CP = [0, 8, 32, 64, 128, 256, 512, 1024, 2048, STEPS]


def kernel(a, q, t):
    """Van Loan covariance over an interval, including integrated observations."""
    n = len(a)
    e = expm(t*np.block([[a, q], [np.zeros_like(a), -a.T]]))
    p = e[:n, :n]
    cov = e[:n, n:]@p.T
    val, vec = np.linalg.eigh((cov+cov.T)/2)
    assert val.min() >= -1e-9*max(1., val.max())
    return np.ascontiguousarray(p.T), np.ascontiguousarray((vec*np.sqrt(np.maximum(val, 0))).T), cov


def matrices(route, t):
    a, d, w = (np.asarray(route[k]) for k in ('A','D','B'))
    s, o = len(a), len(d)
    j = np.block([[a, np.zeros((s, 2*o))],
                  [w, d, np.zeros_like(d)],
                  [np.zeros((o, s)), np.zeros_like(d), d]])
    q = block_diag(route['Qs'], np.zeros_like(d), route['Qo'])
    p = covariance(j, q)
    c = np.asarray(route['c'])[:o]
    # Integrate c*r and c*v_local over the window (not point samples).
    obs = np.zeros((2, s+2*o)); obs[0, s:s+o] = c/t; obs[1, s+o:] = c/t
    win, lw, covw = kernel(np.block([[j, np.zeros((len(j), 2))],
                                 [obs, np.zeros((2, 2))]]), block_diag(q, np.zeros((2, 2))), t)
    gap, lg, covgap = kernel(j, q, GAP)
    np.testing.assert_allclose(gap.T@p@gap+covgap, p, atol=2e-9)
    edw = expm(d*t)
    dc = np.linalg.solve(d, edw-np.eye(o))
    h = np.asarray(route['cue'])[o:]@win[:s, -2]
    from strengthening_analysis import generator, moments
    rr = {key: np.asarray(val) for key, val in route.items()}
    h1, v1, _ = moments(generator(rr, 1.), block_diag(rr['Qo'], rr['Qs']), rr['cue'], rr['c'], t)
    avg = win[:len(j), -2:].sum(axis=1)
    np.testing.assert_allclose([h, avg@p@avg+covw[-2:, -2:].sum()], [h1,v1], rtol=1e-8, atol=1e-11)
    assert abs(h) > 0
    return (gap, lg, win, lw, np.ascontiguousarray(expm(d*GAP).T),
            np.ascontiguousarray(edw.T), np.ascontiguousarray(dc.T@c/t), h, s, o,
            float(np.trace(p[s:s+o, s:s+o])))


@njit
def batch(x, delta, m, zg, zw, cue, gap, lg, win, lw, edg, edw, c_int, h, s, o):
    n = x.shape[1]
    augmented = np.zeros((len(m), n+2))
    scores = np.zeros((len(m), len(cue)))
    variance = np.zeros(len(m))
    for j in range(len(cue)):
        x = x@gap+zg[j]@lg
        delta = delta@edg
        for k in range(s, s+o):
            variance += x[:, k]**2
        augmented[:, :n] = x
        y = augmented@win+zw[j]@lw
        raw = m*y[:, n]+y[:, n+1]+delta@c_int
        scores[:, j] = np.sign(h)*cue[j]*raw+m*abs(h)
        x = np.ascontiguousarray(y[:, :n])
        delta = delta@edw
    return x, delta, scores, variance/len(cue)


def warm(kernels, rng):
    states, ks = [], []
    for a in kernels:
        gap, lg, *_, s, o, _ = a
        x = np.zeros((REP, len(gap)))
        stat = np.zeros(REP)
        for j in range(WARM):
            x = x@gap+rng.normal(size=x.shape)@lg
            if j >= WARM//4:
                stat += np.sum(x[:, s:s+o]**2, axis=1)
        ks.append(stat/(3*WARM//4)); states.append(x)
    return states, np.array(ks).T


def exchange(b, p, step):
    # Open chain: flow along an edge is positive from i to i+1.
    flow = step*(p[:, :-1]-p[:, 1:])
    debit = np.pad(np.maximum(flow, 0.), ((0,0),(0,1)))
    debit += np.pad(np.maximum(-flow, 0.), ((0,0),(1,0)))
    scale = np.divide(b, debit, out=np.ones_like(b), where=debit>b)
    flow *= np.where(flow >= 0., scale[:, :-1], scale[:, 1:])
    return np.maximum(b+np.pad(flow, ((0,0),(1,0)))-np.pad(flow, ((0,0),(0,1))), 0.)


def actuator(m, b, k, speed):
    root = np.sqrt(b/k)
    th = np.tanh(speed*np.sqrt(b*k))
    div = root+m*th
    return np.where(b > 0, root*(m+root*th)/np.where(div>0,div,1.), m/(1+speed*k*m))


def run(kernels, budget, target, correct, start, m0, seed, signal='binary', speed=GAMMA, prior=None):
    rng = np.random.default_rng(seed)
    n = len(kernels)
    ktrue = np.array([a[-1] for a in kernels])  # diagnostics only
    if prior is None:
        states, k = warm(kernels, rng)
        m = np.broadcast_to(m0, (REP,n)).copy()
        b = np.broadcast_to(start, (REP,n)).copy()
        delta = [np.zeros((REP,a[-2])) for a in kernels]  # prior source-driven output minus m*r
    else:
        states, k, m, b, delta = prior
    initial = correct(ktrue*m*m).min(axis=1)
    history = {}
    alpha = -np.expm1(-1/TAU)
    sums = np.zeros_like(b); bsums = np.zeros_like(b); prefix = {}

    def save(j, actual, goal):
        q = lambda a: np.quantile(a, [.1,.5,.9], axis=0).tolist()
        norm = actual.sum(axis=1)/budget
        tv = np.abs(actual/np.maximum(actual.sum(axis=1, keepdims=True), 1e-300)-target).sum(axis=1)/2
        return dict(decisions_per_output=j*BATCH, goal_share=q(goal/budget),
                    realized_share_of_B=q(actual/budget), cost_ratio=q(norm),
                    abs_cost_error=q(abs(norm-1)), allocation_tv=q(tv) if j or np.all(norm) else None,
                    frozen_gain_benefit_pp=q(100*(correct(actual).min(axis=1)-initial)),
                    relative_cost_estimation_error=q(abs(k/ktrue-1)),
                    instantaneous_abs_cost_error=q(abs((ktrue*m*m).sum(axis=1)/budget-1)))

    history[0] = save(0, ktrue*m*m, b)
    for j in range(1,STEPS+1):
        ps = []; vs = []
        labels = rng.choice(np.array([-1.,1.]), BATCH)
        for i,a in enumerate(kernels):
            z1 = rng.normal(size=(BATCH,REP,len(a[0])))
            z2 = rng.normal(size=(BATCH,REP,len(a[2])))
            x, dd, score, var = batch(states[i],delta[i],m[:,i],z1,z2,labels,*a[:-1])
            states[i] = x; delta[i] = dd; vs.append(var)
            if signal == 'binary':
                ps.append((score>0).mean(axis=1))
            else:
                # Same labelled samples; use the cue-aligned readout margin.
                ps.append(ndtr(score.mean(axis=1)/np.maximum(score.std(axis=1,ddof=1),1e-12)))
        k = (1-alpha)*k+alpha*np.array(vs).T
        b = exchange(b,np.array(ps).T,.20*budget/(1+(j-1)/50)**.65)
        new = actuator(m,b,np.maximum(k,1e-12),speed)
        for i,a in enumerate(kernels):
            s,o = a[-3:-1]
            delta[i] += (m[:,i]-new[:,i])[:,None]*states[i][:,s:s+o]
        m = new
        sums += ktrue*m*m; bsums += b
        if j in (x//2 for x in CP[1:]):
            prefix[j] = (sums.copy(), bsums.copy())
        if j in CP[1:]:
            p,q = prefix.pop(j//2)
            history[j] = save(j,(sums-p)/(j//2),(bsums-q)/(j//2))
    assert np.max(abs(b.sum(axis=1)/budget-1))<1e-12 and np.min(m)>=0
    return history, (states,k,m,b,delta)


def pair_routes():
    return [dict(A=[[-b]], D=[[-a]], B=[[1.]], Qs=[[1.]], Qo=[[1.]],
                 c=[1.,0.],cue=[0.,1/np.sqrt(2)]) for a,b in ((.5,2.),(2.,.5))]


def main():
    from recurrent_feedback import environment
    previous = json.loads((HERE/'strengthening_results.json').read_text())['recurrent']
    result = {'protocol': dict(seed=SEED,replicates=REP,batches=STEPS,batch_size=BATCH,
              blank_time=GAP, warmup_blank_samples=WARM, tau_batches=TAU, actuator_rate=GAMMA,
              checkpoints=CP, topology='open unit-weight chain',
              evaluation='second-half mean frozen-gain costs, not covariance during adaptation'), 'runs':{}}
    for case in (-1,1,3):
        net = None if case<0 else next(x for x in previous['networks'] if x['case']==case)
        budget = PAIR_B if net is None else net['unit_gain_cost']
        routes = pair_routes() if net is None else net['routes']
        n=len(routes)
        goal = np.array([X0, budget-X0]) if net is None else np.full(n,budget/n)
        m0 = np.array([2.34,.90]) if net is None else np.ones(n)
        short_state = None
        for wi,t in enumerate((.5,3.) if net is None else (.35,1.6)):
            kernels = [matrices(r,t) for r in routes]
            if net is None:
                _,x = env(t); target = np.array([x,budget-x])/budget
                # Pair evaluator at arbitrary (not necessarily budget-conserving) gains.
                from strengthening_analysis import moments, generator
                coefs=[]
                for r in routes:
                    rr={key:np.asarray(v) for key,v in r.items()}
                    j0,j1=generator(rr,0.),generator(rr,1.)
                    _,v0,_=moments(j0,np.eye(2),rr['cue'],rr['c'],t)
                    h,v1,_=moments(j1,np.eye(2),rr['cue'],rr['c'],t)
                    coefs.append((h,v0,v1-v0))
                h,l,s=np.array(coefs).T
                correct=lambda x: ndtr(h*np.sqrt((x/.2)/(l+(x/.2)*s)))
            else:
                correct,_=environment(net,t)
                target=np.array(next(r['fair_predicted_shares'] for r in previous['rows'] if
                    r['network']==case and r['T']==t and r['budget_multiplier']==1.))
            np.testing.assert_allclose(correct(budget*target), correct(budget*target).mean(),atol=1e-10)
            key=f'{n}_{t}'
            seed=SEED+20*n+wi
            kinds = [('binary',m0,'binary',GAMMA),('margin',m0,'margin',GAMMA)]
            if case in (1,3):
                kinds += [('zero_start',np.zeros(n),'margin',GAMMA)]
            if case==1 and wi==0:
                kinds += [('slow_actuator',m0,'margin',GAMMA/64)]
            result['runs'][key] = dict(target_share=target.tolist(),budget=budget)
            for name, start, sig, speed in kinds:
                z, state = run(kernels,budget,target,correct,goal,start,seed,signal=sig,speed=speed)
                if case<0 and wi==0 and name=='margin':
                    short_state=state
                result['runs'][key][name]=z
                print(key,name,[(j,round(z[j]['allocation_tv'][1],4),round(z[j]['cost_ratio'][1],4),
                                 round(z[j]['frozen_gain_benefit_pp'][1],4)) for j in (64,512,4096)],flush=True)
            if case<0 and wi==1:
                z,_ = run(kernels,budget,target,correct,goal,m0,seed,signal='margin',prior=short_state)
                result['runs'][key]['switch_margin']=z
                print(key,'switch',z[0]['realized_share_of_B'][1],z[4096]['realized_share_of_B'][1],flush=True)
    (HERE/'local_gain_loop_results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
