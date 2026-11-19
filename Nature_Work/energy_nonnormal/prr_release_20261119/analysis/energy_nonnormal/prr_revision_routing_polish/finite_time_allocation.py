"""Independent checks of modular costs, noise spectra, allocation and decoding.

All examples are synthetic. No recording data or preceding result is rewritten.
"""
from pathlib import Path
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.linalg import block_diag, expm, solve_continuous_lyapunov, solve_toeplitz, toeplitz
from scipy.optimize import brentq, minimize

from window_crossover import models, readout, signal

HERE = Path(__file__).resolve().parent
INK, SLOW, FAST, TEAL = "#182630", "#a0422b", "#245e94", "#29776e"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.3, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42})


def covariance(j, q):
    s = solve_continuous_lyapunov(j, -q)
    assert np.max(np.linalg.eigvals(j).real) < 0
    assert np.linalg.norm(j@s+s@j.T+q) < 1e-9*(1+np.linalg.norm(q))
    return (s+s.T)/2


def box_moments(j, q, cue, c, duration):
    """Exact integrals of matrix exponentials, not discretized sample means."""
    cov = covariance(j, q)
    e = expm(j*duration)
    mean = c @ np.linalg.solve(j, (e-np.eye(len(j))) @ cue)/duration
    integ = np.linalg.solve(j, np.linalg.solve(j, e-np.eye(len(j))-j*duration))
    var = 2*c @ integ @ cov @ c/duration**2
    assert var >= -1e-11
    return float(mean), float(max(var, 0)), cov


def modular_checks():
    rng = np.random.default_rng(271828)
    rows, worst_cost, worst_readout = [], 0., 0.
    for case in range(24):
        ns, no = 2+case%4, 2+(case//4)%4
        def internal(n):
            a = rng.uniform(.03, .22, (n, n))
            np.fill_diagonal(a, 0.)
            if case%2:
                a *= rng.choice([-1., 1.], a.shape)
            return a-np.diag(np.sum(abs(a), axis=1)+rng.uniform(.4, 1., n))
        a, d = internal(ns), internal(no)
        b = rng.uniform(.1, 1., (no, ns)); b /= np.linalg.norm(b)
        cue = np.r_[np.zeros(no), np.ones(ns)/np.sqrt(ns)]
        c = np.r_[np.ones(no)/np.sqrt(no), np.zeros(ns)]
        qs, qo = np.diag(rng.uniform(.4, 1.5, ns)), np.diag(rng.uniform(.4, 1.5, no))
        q = block_diag(qo, qs)
        def generator(m):
            return np.block([[d, m*b], [np.zeros((ns, no)), a]])
        j0, j1 = generator(0.), generator(1.)
        base = np.trace(covariance(j0, q))
        src = covariance(j1, block_diag(np.zeros((no, no)), qs))
        kappa = np.trace(src[:no, :no])
        assert kappa > 0 and np.linalg.norm(a@a) > 0 and np.linalg.norm(d@d) > 0
        for duration in (.35, 1.6):
            h, propagated, _ = box_moments(j1, block_diag(np.zeros((no, no)), qs), cue, c, duration)
            _, local, _ = box_moments(j0, block_diag(qo, np.zeros((ns, ns))), cue, c, duration)
            fmax, price = 4*h*h/propagated, kappa*local/propagated
            assert min(local, propagated, fmax, price) > 0
            for m in (.05, .2, .8, 2., 4.):
                mean, var, cov = box_moments(generator(m), q, cue, c, duration)
                extra = np.trace(cov)-base
                predicted = fmax*extra/(extra+price)
                actual = 4*mean*mean/var
                cost_err = abs(extra-kappa*m*m)/(kappa*m*m)
                read_err = abs(actual-predicted)/max(actual, 1e-15)
                worst_cost, worst_readout = max(worst_cost, cost_err), max(worst_readout, read_err)
                np.testing.assert_allclose(extra, kappa*m*m, rtol=2e-9)
                np.testing.assert_allclose(actual, predicted, rtol=2e-9)
                rows.append({"case": case, "source_dim": ns, "output_dim": no,
                             "T": duration, "gain": m, "x_over_price": extra/price,
                             "squared_d_fraction": actual/fmax})
            # Separate quadrature checks on each module at unit forward gain.
            z, w = leggauss(48); u=(z+1)*duration/2
            cov=covariance(j1, q)
            hq=sum(ww*(c@expm(j1*t)@cue) for t,ww in zip(u,w))/2
            vq=sum(ww*(duration-t)*(c@expm(j1*t)@cov@c) for t,ww in zip(u,w))/duration
            hm, vm, _ = box_moments(j1, q, cue, c, duration)
            np.testing.assert_allclose([hm,vm], [hq,vq], rtol=2e-10)
    return {"modules": 24, "moment_cases": len(rows), "rows": rows,
            "maximum_relative_cost_error": worst_cost,
            "maximum_relative_readout_error": worst_readout}


def coefficients(duration):
    base = models((0., 0.), 0.)
    fmax, price, current = [], [], []
    for model, a in zip(base, (.5, 2.)):
        j=model[0]
        m=j[0,1]
        row=readout(model, duration, "box", n=64)
        local=(a*duration+np.expm1(-a*duration))/(a**3*duration**2)
        src=(row["variance"]-local)/(m*m)
        h=row["mean"]/m
        fmax.append(4*h*h/src);price.append(.2*local/src);current.append(.2*m*m)
    np.testing.assert_allclose(fmax[0],fmax[1],rtol=2e-11)
    return np.array(fmax),np.array(price),np.array(current)


def scores(x, ceiling, price):
    return ceiling*x/(x+price)


def sum_optimum(budget, ceiling, price, weight=None):
    p=np.ones(len(price)) if weight is None else np.asarray(weight)
    if budget == 0:
        return np.zeros(len(price))
    upper=max(p*ceiling/price)
    def allocation(lam):
        return np.maximum(np.sqrt(p*ceiling*price/lam)-price,0.)
    lam=brentq(lambda z: sum(allocation(z))-budget,upper*1e-15,upper,
               xtol=1e-15,rtol=1e-13)
    x=allocation(lam)
    np.testing.assert_allclose(sum(x),budget,rtol=2e-9,atol=1e-12)
    return x


def fair_optimum(budget, ceiling, price):
    if np.ptp(ceiling)<1e-10*max(ceiling):
        return budget*price/sum(price)
    y=brentq(lambda y: sum(price*y/(ceiling-y))-budget,0.,
              min(ceiling)*(1-1e-12),xtol=1e-14)
    return price*y/(ceiling-y)


def allocation_checks():
    rng=np.random.default_rng(161803)
    checks=[]
    for n in (2,4,8):
        for k in range(12):
            f=np.exp(rng.uniform(-1.,1.,n));c=np.exp(rng.uniform(-2.,2.,n))
            p=np.exp(rng.uniform(-.5,.5,n));b=10**rng.uniform(-2.,1.)
            x=sum_optimum(b,f,c,p)
            u=sum(p*scores(x,f,c))
            def objective(z):
                return -sum(p*scores(b*z,f,c))
            def gradient(z):
                return -b*p*f*c/(b*z+c)**2
            opt=minimize(objective,np.ones(n)/n,jac=gradient,method="SLSQP",
                bounds=[(0.,1.)]*n,constraints={"type":"eq","fun":lambda z:sum(z)-1.,
                "jac":lambda z:np.ones(n)},options={"ftol":1e-13,"maxiter":1000})
            assert opt.success,opt.message
            assert abs(u+opt.fun)<2e-8*max(1.,u)
            xf=fair_optimum(b,f,c);ss=scores(xf,f,c)
            np.testing.assert_allclose(np.max(ss),np.min(ss),rtol=2e-9)
            np.testing.assert_allclose(sum(xf),b,rtol=2e-9)
            checks.append({"n":n,"budget":b,"sum_objective_error":abs(u+opt.fun)})
    examples=[]
    tgrid=np.unique(np.r_[np.linspace(.08,5.5,100),.5,1.39930089008,3.])
    for t in tgrid:
        f,c,x=coefficients(t);b=sum(x)
        xs=sum_optimum(b,f,c);xf=fair_optimum(b,f,c)
        examples.append({"T":float(t),"budget":float(b),"ceiling":f.tolist(),
            "price":c.tolist(),"fixed":x.tolist(),"joint_optimal":xs.tolist(),
            "fair_optimal":xf.tolist(),"joint_fixed":float(np.sqrt(sum(scores(x,f,c)))),
            "joint_optimum":float(np.sqrt(sum(scores(xs,f,c)))),
            "min_fixed":float(np.sqrt(min(scores(x,f,c)))),
            "min_optimum":float(np.sqrt(min(scores(xf,f,c))))})
    crossing=brentq(lambda t: (lambda f,c,x: c[0]/c[1]-x[0]/x[1])(*coefficients(t)),.5,3.)
    f,c,x=coefficients(crossing)
    np.testing.assert_allclose(fair_optimum(sum(x),f,c),x,rtol=1e-10)
    return {"random_optimization_checks":checks,"examples":examples,
            "fair_allocation_crossing":crossing}


SPECTRA={
    "OU": [[1.,.5,0.]],
    "OU mixture": [[.6,.25,0.],[.4,1.,0.]],
    "Oscillatory mixture": [[.85,.3,0.],[.15,.7,1.8]],
    "No-reversal control": [[1.,2.,0.]],
}
FAST_SPEC=[[.25,2.,0.]]


def local_cov(spec,t):
    return sum(a*np.exp(-r*abs(t))*np.cos(w*t) for a,r,w in spec)


def local_box(spec,t):
    # Each component a exp(-r|u|)cos(wu) has a nonnegative spectral density.
    result=0.
    for a,r,w in spec:
        z=(r-1j*w)*t
        fun=(.5-z/6+z*z/24-z**3/120+z**4/720) if abs(z)<1e-3 else (z+np.expm1(-z))/(z*z)
        result+=2*a*np.real(fun)
    return float(result)


def spectral_checks():
    grid=np.geomspace(.001,100.,300);rho=(2.34/.9)**2
    out=[]
    for name,spec in SPECTRA.items():
        r0=sum(v[0] for v in spec)/.25
        p0=sum(2*a*r/(r*r+w*w) for a,r,w in spec)/.25
        ratio=np.array([local_box(spec,t)/local_box(FAST_SPEC,t) for t in grid])
        roots=[]
        for lo,hi,y0,y1 in zip(grid,grid[1:],ratio[:-1],ratio[1:]):
            if (y0-rho)*(y1-rho)<0:
                roots.append(brentq(lambda t:local_box(spec,t)/local_box(FAST_SPEC,t)-rho,lo,hi))
        predicted=min(r0,p0)<rho<max(r0,p0)
        assert not predicted or len(roots)>0
        for t in (.03,.5,3.,20.):
            direct=2*quad(lambda u:(t-u)*local_cov(spec,u),0,t,epsabs=1e-12)[0]/t**2
            np.testing.assert_allclose(local_box(spec,t),direct,rtol=2e-10)
        np.testing.assert_allclose(local_box(spec,1e-6)/local_box(FAST_SPEC,1e-6),r0,rtol=3e-6)
        np.testing.assert_allclose(local_box(spec,1e6)/local_box(FAST_SPEC,1e6),p0,rtol=5e-6)
        out.append({"name":name,"components":spec,"instantaneous_ratio":r0,
            "zero_frequency_ratio":p0,"predicted_reversal":predicted,
            "crossings_in_sampled_range":roots,"T":grid.tolist(),"price_ratio":ratio.tolist()})
    return {"allocation_ratio":rho,"fast_components":FAST_SPEC,"examples":out}


def decoder_checks():
    out=[];worst=0.
    for t in np.unique(np.r_[np.linspace(.15,5.,50),.5,1.39930089008,3.]) :
        levels=[]
        for dt in (.02,.01):
            routes=[]
            for model in models((0.,0.),0.):
                j,f,cov=model;ts=np.linspace(0,t,int(np.ceil(t/dt))+1)
                mu=signal(j,ts);lag=expm(ts[:,None,None]*f)[:,0,:]@cov[:,0]
                c=toeplitz(lag)
                box=2*np.mean(mu)/np.sqrt(np.mean(c))
                matched=2*(mu@mu)/np.sqrt(mu@c@mu)
                oracle=2*np.sqrt(mu@solve_toeplitz(lag,mu))
                assert np.isfinite([box,matched,oracle]).all()
                assert oracle>=max(box,matched)*(1-1e-9)
                routes.append([float(box),float(matched),float(oracle)])
            levels.append(routes)
        ratios=[np.array(v)[0]/np.array(v)[1] for v in levels]
        # Assess convergence of the temporal oracle, not the point-sampling
        # approximation to the continuous boxcar.
        err=abs(ratios[0][2]/ratios[1][2]-1);worst=max(worst,err)
        out.append({"T":float(t),"dprime_slow":levels[1][0],"dprime_fast":levels[1][1],
            "ratios":ratios[1].tolist()})
    assert min(r["ratios"][2] for r in out)>1
    return {"readouts":["uniform point average","signal-matched","covariance-optimal"],
            "fine_step":.01,"coarse_step":.02,"max_relative_oracle_ratio_change":worst,
            "minimum_oracle_ratio":min(r["ratios"][2] for r in out),"examples":out}


def figures(record):
    fig,axes=plt.subplots(2,2,figsize=(7.4,5.3))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.90,hspace=.50,wspace=.36)
    fig.suptitle("The receiver determines the optimal allocation",x=.02,ha="left",fontsize=13)
    a,b,c,d=axes.ravel()
    a.set_title("(a) Internal recurrence is allowed",loc="left",fontweight="semibold")
    a.axis("off");a.set(xlim=(0,1),ylim=(0,1))
    for x,name in [(.26,"Source module\nStable $A_i$"),(.78,"Output module\nStable $D_i$")]:
        a.add_patch(FancyBboxPatch((x-.19,.48),.38,.33,boxstyle="round,pad=.01",fc="#eef2f4",ec=INK))
        a.text(x,.65,name,ha="center",va="center",fontsize=8.6)
        a.annotate("",xy=(x+.12,.48),xytext=(x-.12,.48),
                   arrowprops={"arrowstyle":"->","connectionstyle":"arc3,rad=.55","color":INK})
    a.annotate("",xy=(.57,.64),xytext=(.46,.64),arrowprops={"arrowstyle":"->","color":TEAL,"lw":1.6})
    a.text(.52,.87,r"$m_i B_i$",ha="center",color=TEAL)
    a.text(.02,.29,r"Exact cost: $x_i=\kappa_i m_i^2$",fontsize=9)
    a.text(.02,.08,r"Readout: $d_i'^2=F_i\,x_i/(x_i+C_{*,i})$",fontsize=9)

    b.set_title("(b) Noise spectra predict reversals",loc="left",fontweight="semibold")
    for row,col,ls in zip(record["spectral"]["examples"],(INK,TEAL,SLOW,FAST),("-","--","-.",":")):
        b.semilogx(row["T"],row["price_ratio"],color=col,ls=ls,lw=1.3,label=row["name"])
    b.axhline(record["spectral"]["allocation_ratio"],color=".55",ls="--",lw=.9)
    b.text(.025,7.7,"Fixed allocation ratio",fontsize=7.5,color=".4")
    b.set(xlim=(.02,40),ylim=(2.8,24),xlabel="Window duration $T$",ylabel="Local noise price ratio")
    b.legend(frameon=False,loc="upper left",fontsize=7,handlelength=1.8)

    c.set_title("(c) Different tasks, different budgets",loc="left",fontweight="semibold")
    rows=record["allocation"]["examples"];ts=[r["T"] for r in rows]
    c.plot(ts,[r["fixed"][0]/r["budget"] for r in rows],c=".55",ls=":",label="Fixed allocation")
    c.plot(ts,[r["fair_optimal"][0]/r["budget"] for r in rows],c=SLOW,lw=1.5,label="Protect worst output")
    c.plot(ts,[r["joint_optimal"][0]/r["budget"] for r in rows],c=FAST,ls="--",lw=1.5,label="Joint discrimination")
    cross=record["allocation"]["fair_allocation_crossing"]
    c.axvline(cross,color=".6",ls=":",lw=.8)
    c.text(cross+.1,.72,r"$T_\times=1.399$",fontsize=7.6)
    c.set(xlim=(0,5.5),ylim=(.5,.98),xlabel="Window duration $T$",ylabel="Slow-output budget share")
    c.text(3.0,.843,"Fixed allocation",color=".45",fontsize=7.7)
    c.text(2.8,.946,"Protect worst output",color=SLOW,fontsize=7.7)
    c.text(2.8,.705,"Joint discrimination",color=FAST,fontsize=7.7)

    d.set_title("(d) Reliability at a fixed total cost",loc="left",fontweight="semibold")
    f,p,x=coefficients(3.);bs=np.linspace(0,3,150)
    for label,policy,col,ls in [
        ("Optimal joint",lambda b:sum_optimum(b,f,p),INK,"-"),
        ("Equal shares",lambda b:np.array([b/2,b/2]),TEAL,"--"),
        ("Fixed ratio",lambda b:b*x/sum(x),SLOW,"-."),
        ("All in fast",lambda b:np.array([0.,b]),FAST,":")]:
        d.plot(bs,[sum(scores(policy(b),f,p)) for b in bs],c=col,ls=ls,lw=1.4,label=label)
    d.axvline(sum(x),c=".6",ls=":",lw=.8)
    d.set(xlabel="Total excess variance budget $B$",ylabel=r"Joint squared separation $d_{\rm joint}'^2$",xlim=(0,3))
    d.legend(frameon=False,loc="lower right",fontsize=7.1)
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"fig_prr4_allocation.{ext}",dpi=220)
    plt.close(fig)

    fig,(a,b)=plt.subplots(1,2,figsize=(7.4,2.9))
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.84,wspace=.36)
    a.set_title("(a) Recurrent-module validation",loc="left")
    r=record["modules"]["rows"]
    a.scatter([z["x_over_price"] for z in r],[z["squared_d_fraction"] for z in r],s=9,
              color=TEAL,alpha=.4,label="24 modules, 240 checks")
    xx=np.geomspace(1e-5,1e3,300);a.plot(xx,xx/(1+xx),c=INK,lw=1,label=r"$x/(1+x)$")
    a.set(xscale="log",xlabel=r"Cost / half-ceiling price $x_i/C_{*,i}$",ylabel=r"Squared separation / ceiling $d_i'^2/F_i$")
    a.legend(frameon=False,loc="upper left",fontsize=7.3)
    rows=record["decoders"]["examples"]
    for k,label,col,ls in zip(range(3),("Uniform window","Signal-matched","Temporal optimum"),(INK,TEAL,SLOW),("-","--","-.")):
        b.plot([z["T"] for z in rows],[z["ratios"][k] for z in rows],c=col,ls=ls,label=label)
    b.axhline(1,color=".6",ls=":",lw=.9)
    b.set(xlabel="Available readout duration $T$",ylabel=r"Separation ratio $d'_{\rm slow}/d'_{\rm fast}$")
    b.set_title("(b) The receiver constraint matters",loc="left")
    b.legend(frameon=False,loc="upper right",fontsize=7.1)
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"fig_prrS_allocation_checks.{ext}",dpi=220)
    plt.close(fig)


def main():
    with warnings.catch_warnings():
        # Accelerate can flag spurious BLAS warnings; explicit finite-value,
        # covariance-residual and cross-method assertions remain active.
        warnings.filterwarnings("ignore",category=RuntimeWarning,message=".*encountered in matmul")
        result={"modules":modular_checks(),"allocation":allocation_checks(),
                "spectral":spectral_checks(),"decoders":decoder_checks(),"passed":True}
        figures(result)
    (HERE/"finite_time_allocation_results.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"modules":result["modules"]["modules"],
          "module_checks":result["modules"]["moment_cases"],
          "module_cost_error":result["modules"]["maximum_relative_cost_error"],
          "module_readout_error":result["modules"]["maximum_relative_readout_error"],
          "optimization_checks":len(result["allocation"]["random_optimization_checks"]),
          "crossing":result["allocation"]["fair_allocation_crossing"],
          "minimum_oracle_ratio":result["decoders"]["minimum_oracle_ratio"],
          "oracle_resolution_change":result["decoders"]["max_relative_oracle_ratio_change"],
          "passed":True},indent=2))


if __name__=="__main__":
    main()
