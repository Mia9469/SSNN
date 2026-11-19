"""Exact readout crossover and bounded deterministic sensitivity checks."""
from functools import lru_cache
from itertools import product
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
A_SLOW, A_FAST = .5, 2.
M_LOW, M_HIGH = .9, 2.34
T_SHORT, T_LONG = .5, 3.
SIGNAL_HORIZON = 8.
SHAPES = ("box", "triangle", "exponential")
FEEDBACK = (-.10, -.05, 0., .05, .10)  # divided by a_slow
COLOR = (0., .10, .25, .50, 1.)       # tau times a_fast


@lru_cache(None)
def quadrature(n):
    z, w = leggauss(n)
    return (z+1)/2, w/2


def route(a_out, a_in, m, feedback_fraction, color, drive=(1., 1.)):
    r = A_SLOW * feedback_fraction
    j = np.array([[-a_out, m], [r, -a_in]])
    if color == 0:
        f, q = j, np.diag(drive)
    else:
        tau = color / A_FAST
        f = np.block([[j, np.eye(2)], [np.zeros((2, 2)), -np.eye(2)/tau]])
        q = np.diag([0., 0., drive[0]/tau**2, drive[1]/tau**2])
    assert np.max(np.linalg.eigvals(f).real) < 0
    sigma = solve_continuous_lyapunov(f, -q)
    return j, f, (sigma+sigma.T)/2


def signal(j, t):
    """Output response to the source component of a unit joint cue."""
    center = np.trace(j)/2
    delta = np.sqrt(((j[0, 0]-j[1, 1])/2)**2 + j[0, 1]*j[1, 0])
    return j[0, 1]*np.exp((center+delta)*t)*(-np.expm1(-2*delta*t))/(2*delta*np.sqrt(2))


def readout(model, duration, shape, n=32):
    """duration = reciprocal squared weight integral for each kernel."""
    j, f, cov = model
    if shape == "exponential":
        tau = duration/2
        mean = np.linalg.inv(np.eye(2)-tau*j)[0, 1]/np.sqrt(2)
        var = np.linalg.solve(np.eye(len(f))-tau*f, cov)[0, 0]
    else:
        z, w = quadrature(n)
        support = duration if shape == "box" else 4*duration/3
        # After t=support*z: window weights times dt and the lag-overlap
        # kernel times du. The factor 2 counts both lag signs.
        weight = np.ones_like(z) if shape == "box" else 2*(1-z)
        overlap = (1-z) if shape == "box" else 4*((1-z)**3/3+z*(1-z)**2/2)
        mean = np.sum(w*weight*signal(j, support*z))
        lag = expm(support*z[:, None, None]*f)
        var = 2*np.sum(w*overlap*(lag[:, 0, :] @ cov[:, 0]))
    assert var > 0
    return {"mean": float(mean), "variance": float(var), "dprime": float(2*mean/np.sqrt(var))}


def check_with_accumulator(model, duration, shape):
    """Independent check: evolve the covariance of an explicit readout state."""
    j, f, cov = model
    del j
    d = len(f)+1
    drift = np.zeros((d,d))
    drift[:-1,:-1] = f
    noise = np.zeros((d,d))
    noise[:-1,:-1] = -(f@cov+cov@f.T)
    s0 = np.zeros((d,d))
    s0[:-1,:-1] = cov
    x0 = np.zeros(d)
    x0[1] = 1/np.sqrt(2)
    support = duration if shape=="box" else 4*duration/3
    if shape=="exponential":
        support = 20*duration
    def rhs(t, z):
        if shape == "box":
            wt = 1/support
        elif shape == "triangle":
            wt = 2*(1-t/support)/support
        else:
            wt = np.exp(-2*t/duration)*2/duration
        drift[-1,0] = wt
        x, s = z[:d], z[d:].reshape(d,d)
        return np.r_[drift@x, (drift@s+s@drift.T+noise).ravel()]
    final = solve_ivp(rhs, [0,support], np.r_[x0,s0.ravel()], method="DOP853",
                      rtol=2e-11, atol=1e-13)
    assert final.success
    reference = readout(model,duration,shape,n=64)
    np.testing.assert_allclose([final.y[d-1,-1], final.y[-1,-1]],
                               [reference["mean"],reference["variance"]],rtol=1e-9)


def models(fb, color, state="A", drive=(1., 1.)):
    g = (M_HIGH, M_LOW) if state == "A" else (M_LOW, M_HIGH)
    return (route(A_SLOW, A_FAST, g[0], fb[0], color, drive),
            route(A_FAST, A_SLOW, g[1], fb[1], color, drive))


def matching(a, b):
    def rel(x, y):
        return float(np.max(np.abs(x-y))/max(np.max(np.abs(x)), np.max(np.abs(y))))

    def eigen(model, cov=False):
        arrays = [np.linalg.eigvalsh(v[2][:2, :2]) if cov else
                  np.linalg.eigvals(v[0]).real for v in model]
        return np.sort(np.concatenate(arrays))

    def peak_samples(model):
        t = np.r_[0, np.geomspace(1e-3, 30., 64)]
        return np.maximum.reduce([np.linalg.svd(expm(t[:, None, None]*v[0]),
                                   compute_uv=False)[:, 0]**2 for v in model])

    c, d = (sum(np.trace(x[2][:2, :2]) for x in v) for v in (a, b))
    return {"decay_eigenvalue_relative_error": rel(eigen(a), eigen(b)),
            "stationary_cost_relative_error": abs(c-d)/max(c,d),
            "covariance_eigenvalue_relative_error": rel(eigen(a, True), eigen(b, True)),
            "gain_curve_sample_relative_error": rel(peak_samples(a), peak_samples(b))}


def stress(fb, color, shape, cache):
    a, b, match, amp = cache[(fb, color)]
    scores = []
    for t in (T_SHORT, T_LONG):
        rs, rf = [readout(v, t, shape) for v in a]
        for v, row in zip(a, (rs, rf)):
            finer = readout(v, t, shape, n=64)
            np.testing.assert_allclose(list(row.values()), list(finer.values()),
                                       rtol=1e-10, atol=2e-13)
        scores.append({"duration": t, "slow": rs, "fast": rf,
                       "dprime_ratio": rs["dprime"]/rf["dprime"],
                       "signal_ratio": rs["mean"]/rf["mean"]})
    passed = (amp > 1 and min(x["signal_ratio"] for x in scores)>1 and
              scores[0]["dprime_ratio"]>1 and scores[1]["dprime_ratio"]<1)
    # For T -> 0, the output mean ratio tends to m_high/m_low while
    # the variance tends to the instantaneous stationary output variance.
    zero_ratio = M_HIGH/M_LOW*np.sqrt(a[1][2][0,0]/a[0][2][0,0])
    def difference(t):
        if t == 0:
            return zero_ratio-1
        ds,df = (readout(v,t,shape)["dprime"] for v in a)
        return ds/df-1
    # Search only for the specified slow-to-fast sign change; no assumption
    # of a unique crossing under perturbations. Retain the fixed-window gate.
    search_grid = [0., .025, .05, .10, .20, T_SHORT, 1., 2., T_LONG]
    if passed:
        bracket = (T_SHORT,T_LONG)
    else:
        diff = [difference(t) for t in search_grid]
        bracket = next(((x,y) for x,y,dx,dy in
                        zip(search_grid,search_grid[1:],diff,diff[1:])
                        if dx>0>dy), None)
    cross = brentq(difference, *bracket) if bracket else None
    return {"feedback_over_a_slow": list(fb), "tau_times_a_fast": color,
            "window": shape, "test_points": scores, "minimum_instantaneous_signal_ratio": amp,
            "zero_duration_ratio_limit":float(zero_ratio),
            "passes_fixed_window_reversal": passed,
            "bracketed_slow_to_fast_crossing":cross, "matching": match}


def plot(base, baseline_cross, traces, summary):
    plt.rcParams.update({"font.size":8.6,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.9), layout="constrained")
    a,b,c,d = axes.ravel()
    slow, fast = "#A64D35", "#235B93"
    t=np.linspace(0,5,251)
    for m,col,ls,label in zip(base,(slow,fast),("-", "--"),
                              ("Slow output, stronger branch", "Fast output, weaker branch")):
        a.plot(t,signal(m[0],t),color=col,ls=ls,label=label)
    a.set(title="(a) Stronger signal at every time",xlabel="Time after cue",
          ylabel="Mean response to positive cue")
    a.legend(frameon=False,fontsize=7.2,loc="upper right")
    duration = np.array(traces["duration"])
    for i,col,ls,label in [(0,slow,"-","Slow output"),(1,fast,"--","Fast output")]:
        b.plot(duration,np.array(traces["dprime"])[:,i],c=col,ls=ls,label=label)
    b.axvline(baseline_cross,c=".45",lw=.9,ls=":")
    b.text(baseline_cross+.08,.07,r"$T_\times$",color=".35")
    b.set(title="(b) Readout ranking reverses",
          xlabel=r"Window duration $T$",ylabel=r"Cue separation $d'$")
    b.legend(frameon=False,fontsize=7.2,loc="lower right")
    c.plot(duration,traces["local_variance_ratio"],c=".24",label="Direct output noise ratio")
    c.axhline((M_HIGH/M_LOW)**2,color="#458571",ls="--",label="Squared gain ratio")
    c.axvline(baseline_cross,c=".45",lw=.9,ls=":")
    c.set(title="(c) The crossover is set by local noise",
          xlabel=r"Window duration $T$",ylabel="Slow / fast ratio")
    c.legend(frameon=False,fontsize=7.2,loc="lower right")

    families = [("White, no return", "box",(0.,0.),0.),
                ("Paired return", "box",(.10,.10),0.),
                ("Colored drive", "box",(0.,0.),1.),
                ("Front-weighted", "triangle",(0.,0.),0.),
                ("Exponential", "exponential",(0.,0.),0.),
                ("One-sided return", "box",(.10,0.),0.),
                ("Paired + colored", "box",(.10,.10),1.)]
    for i,(label,shape,fb,color) in enumerate(families):
        r = next(row for row in summary if row["window"]==shape and
                 tuple(row["feedback_over_a_slow"])==fb and row["tau_times_a_fast"]==color)
        x,y=[s["dprime_ratio"] for s in r["test_points"]]
        d.plot([x,y],[i,i],color=".74",lw=1,zorder=1)
        d.plot(x,i,"o",color=slow,ms=4,label="Short" if i==0 else None)
        d.plot(y,i,"s",mfc="white",mec=fast,ms=4,label="Long" if i==0 else None)
    d.axvline(1,c=".35",lw=.9,ls=":")
    d.set(title="(d) Selected sensitivity checks",xlabel=r"Cue separation ratio $d'_{\rm slow}/d'_{\rm fast}$",
          yticks=np.arange(len(families)),yticklabels=[x[0] for x in families])
    d.set_ylim(len(families)-.15,-1.25)
    d.legend(frameon=False,fontsize=7.1,ncol=2,loc="upper center")
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"fig_window_crossover.{ext}",dpi=230)


def main():
    cache = {}
    fb_sets = [(r,r) for r in FEEDBACK] + [(r,0.) for r in FEEDBACK if r!=0.]
    for fb, color in product(fb_sets, COLOR):
        a, b = (models(fb,color,state) for state in ("A","B"))
        # Both pair traces are identical. The ratio of their sinh transfer
        # factors is monotone with time, so the minimum is at 0+ or horizon.
        amp = min(M_HIGH/M_LOW, signal(a[0][0],SIGNAL_HORIZON)/
                  signal(a[1][0],SIGNAL_HORIZON))
        cache[(fb,color)] = (a,b,matching(a,b),float(amp))
    rows = [stress(fb, color, shape, cache)
            for fb,color,shape in product(fb_sets,COLOR,SHAPES)]
    for fb, color in [((0.,0.),0.),((.10,0.),1.)]:
        for model, shape in product(cache[(fb,color)][0],SHAPES):
            check_with_accumulator(model,T_LONG,shape)
    assert all(max(row["matching"].values())<1e-12 for row in rows
               if len(set(row["feedback_over_a_slow"]))==1)
    base, b, base_match, _ = cache[((0.,0.),0.)]
    local_only = models((0.,0.),0.,drive=(1.,0.))
    source_only = models((0.,0.),0.,drive=(0.,1.))
    def v0(a,t):
        return (a*t+np.expm1(-a*t))/(a**3*t*t)

    baseline_cross = brentq(lambda t:v0(A_SLOW,t)/v0(A_FAST,t)-(M_HIGH/M_LOW)**2,.1,8.)
    t = baseline_cross
    d = [readout(v,t,"box")["dprime"] for v in base]
    np.testing.assert_allclose(d[0], d[1], rtol=1e-12)
    np.testing.assert_allclose([readout(m,t,"box")["dprime"] for m in local_only][0],
                               readout(local_only[1],t,"box")["dprime"],rtol=1e-12)
    exp_cross = 2*((M_HIGH/M_LOW)**2-A_FAST/A_SLOW)/(
        A_FAST**2/A_SLOW-(M_HIGH/M_LOW)**2*A_SLOW)
    ds,df=[readout(v,exp_cross,"exponential")["dprime"] for v in base]
    np.testing.assert_allclose(ds,df,rtol=1e-12)
    duration = np.geomspace(.06, 7, 121)
    traces = {"duration":duration.tolist(),"dprime":[[readout(v,t,"box")["dprime"] for v in base]
                                                      for t in duration],
              "local_variance_ratio":[float(v0(A_SLOW,t)/v0(A_FAST,t)) for t in duration]}
    src_tie_error = max(abs(readout(source_only[0],t,"box")["dprime"]-
                            readout(source_only[1],t,"box")["dprime"]) for t in duration)
    assert src_tie_error < 1e-10
    np.testing.assert_allclose(np.linalg.eigvalsh(base[0][2]),np.linalg.eigvalsh(b[1][2]),
                               rtol=1e-12)
    summary = {}
    for tag, group in [("paired", [r for r in rows if len(set(r["feedback_over_a_slow"]))==1]),
                       ("one_sided",[r for r in rows if len(set(r["feedback_over_a_slow"]))!=1])]:
        summary[tag]={"cases":len(group),
                      "passing":sum(r["passes_fixed_window_reversal"] for r in group),
                      "earlier_crossing_in_failed_cases":sum(
                          not r["passes_fixed_window_reversal"] and
                          r["bracketed_slow_to_fast_crossing"] is not None and
                          r["bracketed_slow_to_fast_crossing"] < T_SHORT for r in group),
                      "worst_matching":{k:max(r["matching"][k] for r in group) for k in base_match}}
    record = {"a_slow":A_SLOW,"a_fast":A_FAST,"m_low":M_LOW,"m_high":M_HIGH,
              "short_duration":T_SHORT,"long_duration":T_LONG,
              "signal_rank_horizon":SIGNAL_HORIZON,"box_crossing":float(baseline_cross),
              "exponential_crossing_effective_duration":float(exp_cross),
              "baseline_cost":float(sum(np.trace(v[2]) for v in base)),
              "baseline_matching":base_match,"source_only_max_dprime_difference":src_tie_error,
              "independent_accumulator_checks":12,
              "local_only_crossing":float(baseline_cross),"summary":summary,"tests":rows}
    (HERE/"window_crossover_results.json").write_text(json.dumps(record,indent=2)+"\n")
    plot(base,baseline_cross,traces,rows)
    print(json.dumps({k:v for k,v in record.items() if k!="tests"},indent=2))


if __name__=="__main__":
    main()
