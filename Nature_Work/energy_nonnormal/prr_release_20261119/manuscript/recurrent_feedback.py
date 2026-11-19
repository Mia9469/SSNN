"""Extend S9 to the first four-pathway recurrent network (case 1).

Keep both original windows, the unit-gain total budget, the S9 batch
size, schedule and duration, and all four vertices plus equal shares.
Only the simulated environment and evaluator know the model moments.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from feedback_allocation import BATCH, REPEATS, STEPS
from strengthening_analysis import generator, moments

HERE = Path(__file__).resolve().parent
CASE = 1
WINDOWS = (.35, 1.6)
SEED = 20261112
CHECKPOINTS = np.array([2**k for k in range(2, 13)])


def project_budget(v, budget):
    """Euclidean projection onto nonnegative allocations of a fixed sum."""
    u = np.sort(v, axis=-1)[:, ::-1]
    t = (np.cumsum(u, axis=-1)-budget)/np.arange(1, v.shape[-1]+1)
    return np.maximum(v-np.max(t, axis=-1, keepdims=True), 0.)


def environment(network, window):
    coefs = []
    for route in network['routes']:
        r = {k: np.asarray(v) for k, v in route.items()}
        def direct(m):
            return moments(generator(r, m), r['Q'], r['cue'], r['c'], window)
        _, l, s0 = direct(0.)
        h, total, s = direct(1.)
        kap = np.trace(s-s0)
        coefs.append((abs(h), l, total-l, kap))
        for m in (.23, 1.7):
            m_h, m_var, m_s = direct(m)
            np.testing.assert_allclose([m_h, m_var, np.trace(m_s-s0)],
                                       [m*h, l+m*m*(total-l), kap*m*m], atol=1e-12)
    h, l, s, kap = np.array(coefs).T
    assert np.min(coefs) > 0.

    def correct(x):
        m2 = x/kap
        return ndtr(h*np.sqrt(m2/(l+m2*s)))

    return correct, kap


def simulate(correct, budget, target, start, rng, *, chance=False):
    x = np.broadcast_to(start, (REPEATS, len(start))).copy()
    running = np.zeros_like(x)
    snapshots = {}
    base = correct(start).min()
    best = correct(target).min()
    np.testing.assert_allclose(correct(target), best, atol=1e-10)
    assert best >= base

    def record(k, averaged):
        share = averaged/budget
        tv = np.abs(averaged-target).sum(axis=1)/(2*budget)
        gain = 100*(correct(averaged).min(axis=1)-base)
        return {'decisions_per_output': k*BATCH,
                'share_q10_q50_q90': np.quantile(share, [.1, .5, .9], axis=0).tolist(),
                'tv_q10_q50_q90': np.quantile(tv, [.1, .5, .9]).tolist(),
                'gain_pp_q10_q50_q90': np.quantile(gain, [.1, .5, .9]).tolist(),
                'fraction_improved': float(np.mean(gain>0)),
                'per_run_share': share.tolist()}

    snapshots[0] = record(0, x)
    half = {int(k)//2 for k in CHECKPOINTS}
    prefix = {}
    for k in range(1, STEPS+1):
        p = np.full_like(x, .5) if chance else correct(x)
        obs = rng.binomial(BATCH, p)/BATCH
        step = .20*budget/(1+(k-1)/50)**.65
        x = project_budget(x+step*(obs.mean(axis=1, keepdims=True)-obs), budget)
        np.testing.assert_allclose(x.sum(axis=1), budget, rtol=0., atol=2e-13*budget)
        assert np.min(x) >= 0.
        running += x
        if k in half:
            prefix[k] = running.copy()
        if k in CHECKPOINTS:
            snapshots[k] = record(k, (running-prefix.pop(k//2))/(k//2))
    return {'initial_shares': (start/budget).tolist(),
            'oracle_gain_pp': 100*(best-base), 'checkpoints': snapshots}


def main():
    prev = json.loads((HERE/'strengthening_results.json').read_text())['recurrent']
    network = next(n for n in prev['networks'] if n['case'] == CASE)
    n = network['pathways']
    assert n == 4 and CHECKPOINTS[-1] == STEPS
    rng = np.random.SeedSequence(SEED)
    out = {'protocol': {'case': CASE, 'seed': SEED, 'repeats': REPEATS,
                        'batch_per_output': BATCH, 'batches': STEPS,
                        'budget': network['unit_gain_cost'],
                        'starts': ['equal', 'vertex1', 'vertex2', 'vertex3', 'vertex4'],
                        'summary': 'evaluate average allocation over second half of each prefix'},
           'windows': {}}
    for t in WINDOWS:
        correct, kap = environment(network, t)
        budget = out['protocol']['budget']
        np.testing.assert_allclose(budget, kap.sum(), atol=1e-12)
        row = next(r for r in prev['rows'] if r['network']==CASE and r['T']==t
                   and r['budget_multiplier']==1.)
        target = budget*np.asarray(row['fair_predicted_shares'])
        np.testing.assert_allclose(target/budget, row['fair_direct_shares'], atol=2e-7)
        starts = [np.ones(n)/n, *np.eye(n)]
        runs = {label: simulate(correct, budget, target, budget*start,
                                np.random.default_rng(rng.spawn(1)[0]))
                for label, start in zip(out['protocol']['starts'], starts)}
        chance = simulate(correct, budget, target, budget*starts[0],
                          np.random.default_rng(SEED+1), chance=True)
        out['windows'][str(t)] = {'target_shares': (target/budget).tolist(),
                                  'active': runs, 'chance': chance}
        print('T', t, 'target', np.round(target/budget, 4),
              'oracle gain (pp)', runs['equal']['oracle_gain_pp'])
        for name, run in [*runs.items(), ('chance', chance)]:
            checks = run['checkpoints']
            print(name, 'final TV', np.round(checks[STEPS]['tv_q10_q50_q90'], 4),
                  'median gains (64, 1024, 4096 batches)',
                  [round(checks[k]['gain_pp_q10_q50_q90'][1], 4) for k in (64, 1024, STEPS)],
                  'improved', checks[STEPS]['fraction_improved'])
    a, b = (out['windows'][str(t)]['chance']['checkpoints'][STEPS] for t in WINDOWS)
    np.testing.assert_array_equal(a['per_run_share'], b['per_run_share'])
    (HERE/'recurrent_feedback_results.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    main()
