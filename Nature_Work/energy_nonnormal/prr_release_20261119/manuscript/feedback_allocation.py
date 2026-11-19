"""Finite-feedback test on the unchanged Fig. 3 pair.

Fixed protocol: 512 repeats per condition, 4096 batches of 32 labelled
trials per output, three starts, two windows, and a short-to-long switch.
The learner only sees binary decisions and the size of the budget pool.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from strengthening_analysis import moments

B = 1.25712
X0 = 1.09512
KAPPA = 0.2
REPEATS = 512
STEPS = 4096
BATCH = 32
SEED = 20261111
CHECKPOINTS = (64, 256, 1024, STEPS)


def env(window):
    """Compute the exact Gaussian readout via full-state OU moments."""
    cue, readout = np.array([0., 1/np.sqrt(2)]), np.array([1., 0.])
    coefs = []
    for a, b in ((.5, 2.), (2., .5)):
        def direct(m):
            return moments(np.array([[-a, m], [0., -b]]), np.eye(2),
                           cue, readout, window)
        _, v0, s0 = direct(0.)
        h, v, s = direct(1.)
        np.testing.assert_allclose(np.trace(s-s0), KAPPA, atol=1e-14)
        coefs.append((h, v0, v-v0))
        for x in (0., B/2, B):
            m = np.sqrt(x/KAPPA)
            h2, v2, s2 = direct(m)
            np.testing.assert_allclose((h2, v2, np.trace(s2-s0)),
                                       (m*h, v0+m*m*(v-v0), x), atol=1e-13)
    h, v0, v1 = np.array(coefs).T

    def probability(x):
        x = np.array([x, B-x])
        m = np.sqrt(x/KAPPA)
        scale = (slice(None),) + (None,)*(m.ndim-1)
        return ndtr(m*h[scale]/np.sqrt(v0[scale]+m*m*v1[scale]))

    target = brentq(lambda x: np.diff(probability(x))[0], 0., B)
    expected = B*v0[0]/sum(v0)
    np.testing.assert_allclose(target, expected, atol=1e-12)
    return probability, target


def run(probability, start, seed, *, shuffled=False):
    rng = np.random.default_rng(seed)
    x = np.broadcast_to(start, (REPEATS,)).astype(float).copy()
    total = np.zeros(REPEATS)
    out = {}
    for k in range(1, STEPS+1):
        # For balanced symmetric Gaussian cues, threshold zero is fixed;
        # binomial counts are exactly equivalent to individual correctness flags.
        p = np.full((2, REPEATS), .5) if shuffled else probability(x)
        wins = rng.binomial(BATCH, p)
        step = .20*B/(1+(k-1)/50)**.65
        x = np.clip(x+step*(wins[1]-wins[0])/BATCH, 0., B)
        np.testing.assert_array_less(-1e-15, x)
        np.testing.assert_array_less(x, B+1e-15)
        total += x
        if k in (j//2 for j in CHECKPOINTS):
            out[k] = total.copy()
        if k in CHECKPOINTS:
            out[k] = (total-out.pop(k//2))/(k//2)
    return out


def describe(paths, start, probability, target):
    def q(values):
        return np.quantile(values, [.1, .5, .9]).tolist()
    return {
        str(k): {
            'share_quantiles_10_50_90': q(x/B),
            'absolute_share_error_quantiles_10_50_90': q(abs(x-target)/B),
            'worst_correctness_gain_quantiles_10_50_90': q(
                probability(x).min(axis=0)-probability(start).min(axis=0)),
            'fraction_moving_in_target_direction': float(np.mean((x-start)*(target-start)>0)),
            'replicate_budget': x.tolist(),
        } for k, x in paths.items()
    }


def main():
    rng = np.random.SeedSequence(SEED)
    out = {'protocol': {'seed': SEED, 'budget': B, 'repeats': REPEATS,
                        'batches': STEPS, 'batch_per_output': BATCH,
                        'checkpoint_batches': CHECKPOINTS,
                        'step': '0.20*B/(1+(k-1)/50)**0.65',
                        'summary': 'average over second half of each checkpoint',
                        'initial_slow_budgets': [0., X0, B]}, 'windows': {}}
    saved = {}
    for t in (.5, 3.):
        probability, target = env(t)
        results = {}
        for key, start in (('zero', 0.), ('figure', X0), ('full', B)):
            paths = run(probability, start, rng.spawn(1)[0])
            results[key] = describe(paths, start, probability, target)
            assert results[key][str(STEPS)]['absolute_share_error_quantiles_10_50_90'][1] < .025
            if key == 'figure':
                saved[t] = paths[STEPS]
        sham = run(probability, X0, SEED+1, shuffled=True)
        out['windows'][str(t)] = {
            'target_slow_budget': target, 'target_slow_share': target/B,
            'baseline_correctness_slow_fast': probability(X0).tolist(),
            'active': results, 'label_free_sham': describe(sham, X0, probability, target),
        }
    np.testing.assert_array_equal(
        out['windows']['0.5']['label_free_sham'][str(STEPS)]['replicate_budget'],
        out['windows']['3.0']['label_free_sham'][str(STEPS)]['replicate_budget'])
    probability, target = env(3.)
    out['short_to_long'] = describe(run(probability, saved[.5], rng.spawn(1)[0]),
                                    saved[.5], probability, target)
    assert out['short_to_long'][str(STEPS)]['fraction_moving_in_target_direction'] > .95
    dest = Path(__file__).with_name('feedback_allocation_results.json')
    dest.write_text(json.dumps(out, indent=2)+'\n')
    for t, record in out['windows'].items():
        print(t, 'target budget', record['target_slow_budget'],
              'initial P', record['baseline_correctness_slow_fast'])
        for key in record['active']:
            print(key, {k: tuple(np.round(d['share_quantiles_10_50_90'], 4))
                        for k, d in record['active'][key].items()})
        print('sham', record['label_free_sham'][str(STEPS)]['share_quantiles_10_50_90'])
    print('switch', out['short_to_long'][str(STEPS)]['share_quantiles_10_50_90'])


if __name__ == '__main__':
    main()
