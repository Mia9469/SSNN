"""Check the readout-cost inversion against independent state-space moments.

No optimization or fitting of biological data is performed. Existing result
files are read-only; this writes a separate record for the algebraic bridge.
"""
from pathlib import Path
import json

import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import brentq

from window_crossover import route, readout


def moments(a_out, a_in, m, duration, drive=(1., 1.)):
    return readout(route(a_out, a_in, m, 0., 0., drive), duration,
                   "box", n=64)


def main():
    rows = []
    worst = 0.
    for a_s, a_f in ((.5, 2.), (.7, 1.1), (.2, 3.)):
        kappa = 1/(2*a_s*a_f*(a_s+a_f))
        for duration in (.1, .5, 1.39930089008, 3., 10.):
            local, propagated, signal = [], [], []
            for a_out, a_in in ((a_s, a_f), (a_f, a_s)):
                # Separate the two noise sources by Lyapunov solves.
                local.append(moments(a_out, a_in, 1., duration,
                                     (1., 0.))["variance"])
                propagated.append(moments(a_out, a_in, 1., duration,
                                          (0., 1.))["variance"])
                signal.append(moments(a_out, a_in, 1., duration)["mean"])
            np.testing.assert_allclose(propagated[0], propagated[1],
                                       rtol=2e-10)
            np.testing.assert_allclose(signal[0], signal[1], rtol=2e-10)
            # Independent direct integration of the local lag covariance.
            for a, v0 in zip((a_s, a_f), local):
                direct = quad(lambda u: (duration-u)*np.exp(-a*u)/a,
                              0, duration, epsabs=1e-12)[0]/duration**2
                np.testing.assert_allclose(v0, direct, rtol=2e-10)
            h, v1 = signal[0], propagated[0]
            ceiling = 2*h/np.sqrt(v1)
            for eta in (.2, .6, .9):
                target = eta*ceiling
                costs = []
                for a_out, a_in, v0 in zip((a_s, a_f), (a_f, a_s), local):
                    predicted_cost = kappa*v0*target**2/(4*h*h-target**2*v1)
                    # Find the required coupling from measured state-space d'.
                    coupling = brentq(lambda m: moments(a_out, a_in, m,
                                            duration)["dprime"]-target,
                                        0., 1e4, xtol=1e-12)
                    j = np.array([[-a_out, coupling], [0., -a_in]])
                    sigma = solve_continuous_lyapunov(j, -np.eye(2))
                    actual_cost = np.trace(sigma)-1/(2*a_out)-1/(2*a_in)
                    err = abs(predicted_cost-actual_cost)/predicted_cost
                    worst = max(worst, err)
                    np.testing.assert_allclose(actual_cost, predicted_cost,
                                               rtol=2e-9, atol=2e-12)
                    costs.append(float(actual_cost))
                np.testing.assert_allclose(costs[0]/costs[1],
                                           local[0]/local[1], rtol=2e-9)
                rows.append({"rates": [a_s, a_f], "duration": duration,
                             "ceiling_fraction": eta,
                             "target_dprime": float(target),
                             "slow_output_cost": costs[0],
                             "fast_output_cost": costs[1],
                             "relative_price": float(local[0]/local[1])})

    # For the manuscript example, the same target-discrimination price
    # crosses the actual allocation ratio at the original d' crossover.
    def local_var(a, t):
        return quad(lambda u: (t-u)*np.exp(-a*u)/a, 0, t)[0]/t**2
    gamma = 2.34/.9
    crossing = brentq(lambda t: local_var(.5, t)/local_var(2., t)-gamma**2,
                     .01, 10.)
    reference = json.loads((Path(__file__).parent/
                            "window_crossover_results.json").read_text())
    np.testing.assert_allclose(crossing, reference["box_crossing"],
                               rtol=1e-9)
    result = {"description": "Exact reformulation of existing pair/readout laws",
              "inversion_cases": len(rows)*2,
              "relative_price_cases": len(rows),
              "maximum_cost_relative_error": worst,
              "allocation_ratio": gamma**2,
              "budget_price_crossing": crossing,
              "rows": rows, "passed": True}
    (Path(__file__).parent/"readout_budget_results.json").write_text(
        json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
