"""Redraw the cost summary and check the exact directed-pair target law."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent


def kernel(t, a, b):
    gap = abs(a-b)
    return t*np.exp(-a*t) if gap == 0 else np.exp(-min(a,b)*t)*(-np.expm1(-gap*t))/gap


def target_cost(a, b, gain):
    if gain == 1:
        # Plot the right-hand limit, not the zero cost of the unit gain at t=0.
        return 2/(a+b), 0.0, 2*np.sqrt(a*b)
    if not (min(a,b)>0 and gain>1):
        raise ValueError("positive decay rates and squared gain >1 required")

    def log_required(z):
        t = np.exp(z)
        # Log domain avoids underflow at the long-time edge.
        gap = abs(a-b)
        log_kernel = (z if gap == 0 else np.log(-np.expm1(-gap*t)/gap))-min(a,b)*t
        return np.log(gain-np.exp(-2*a*t))+np.log(gain-np.exp(-2*b*t))-np.log(gain)-2*log_kernel

    grid = np.linspace(np.log(1e-7/max(a,b)),np.log(80/min(a,b)),240)
    vals = log_required(grid)
    k = np.argmin(vals)
    if not 0<k<len(grid)-1:
        raise AssertionError("target minimum not bracketed")
    opt = minimize_scalar(log_required,bounds=(grid[k-1],grid[k+1]),method="bounded",
                          options={"xatol":1e-12})
    m2 = np.exp(opt.fun)
    return float(m2/(2*a*b*(a+b))), float(np.exp(opt.x)), float(np.sqrt(m2))


def check(a,b,gain):
    cost,t,m = target_cost(a,b,gain)
    j = np.array([[-a,m],[0,-b]])
    p = expm(j*t)
    np.testing.assert_allclose(np.linalg.norm(p,2)**2,gain,rtol=3e-9)
    np.testing.assert_allclose(p[0,1],m*kernel(t,a,b),rtol=1e-11)
    sigma = solve_continuous_lyapunov(j,-np.eye(2))
    np.testing.assert_allclose(np.trace(sigma)-.5/a-.5/b,cost,rtol=2e-11)
    times = np.geomspace(1e-6/max(a,b),50/min(a,b),300)
    assert max(np.linalg.norm(expm(j*u),2)**2 for u in times) <= gain*(1+1e-8)
    return {"a_i":a,"a_k":b,"gain":gain,"cost":cost,"time":t,"coupling":m}


def main():
    record=json.loads((HERE.parent/"fixed_budget_selection_results.json").read_text())
    checks=[check(a,b,g) for a,b in [(.2,.2),(.1,.3),(.04,.36),(.8,.2)]
            for g in [1.0001,1.2,2.,4.]]
    for a,b in [(.2,.2),(.1,.3),(.04,.36),(.8,.2)]:
        near=target_cost(a,b,1+1e-8)[0]
        assert abs(near*(a+b)/2-1)<1e-4
    plt.rcParams.update({"font.size":7.8,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42})
    fig=plt.figure(figsize=(7.3,4.35),layout="constrained")
    gs=fig.add_gridspec(2,6)
    ax={(0,j):fig.add_subplot(gs[0,2*j:2*j+2]) for j in range(3)}
    ax.update({(1,0):fig.add_subplot(gs[1,:3]), (1,1):fig.add_subplot(gs[1,3:])})
    ink="#234b70"
    a=ax[0,0];a.set_title("(a) One-pass channel",loc="left")
    a.set_xlim(0,1);a.set_ylim(0,1);a.axis("off")
    for x,name in [(.30,"$v$"),(.76,"$u$")]:
        a.add_patch(Circle((x,.49),.091,fc="#e8f1f7",ec=ink,lw=1))
        a.text(x,.49,name,ha="center",va="center",fontsize=12)
        a.annotate("",(x,.585),(x,.78),arrowprops=dict(arrowstyle="->",color=".42"))
        a.text(x,.34,r"leak $a$",ha="center")
    a.text(.53,.85,"Independent background drive",ha="center",color=".32")
    a.annotate("",(.205,.49),(.036,.49),arrowprops=dict(arrowstyle="->",color=ink))
    a.text(.055,.58,"$b=v$",ha="left",color=ink)
    a.annotate("",(.662,.49),(.392,.49),arrowprops=dict(arrowstyle="->",lw=1.6,color=ink))
    a.text(.53,.57,r"$s$",ha="center",color=ink)
    a.text(.53,.17,r"$K=suv^\top,\quad K^2=0$",ha="center",fontsize=9)
    a.text(.53,.055,"No second pass through the coupling",ha="center",fontsize=6.9)
    a=ax[0,1];a.set_title("(b) Splitting the budget",loc="left")
    for (key,row),color,ls in zip(record["fixed_budget_curves"].items(),
                                 ["#492678","#226080","#348c85","#7a9630","#a66c31"],
                                 ["-","--","-.",":",(0,(3,1,1,1))]):
        a.plot(row["cost_ratio"],row["Gmax"],c=color,ls=ls,label=f"$r={key}$")
    onset=record["parameters"]["rank_one_amplification_threshold_ratio"]
    a.vlines(onset,.86,1.65,color=".55",ls=":",lw=.9)
    a.set(xlabel=r"Cost $C_0/C_{\rm normal}$",ylabel=r"Peak squared gain $G_{\max}^{(2)}$")
    a.legend(frameon=False,loc="upper left",fontsize=6.5,ncol=2)
    a=ax[0,2];a.set_title("(c) Chains at matched cost",loc="left")
    ch=record["higher_order_chains"]
    for (key,row),ls in zip(ch["ratio_to_rank_one"].items(),["-","--","-."]):
        a.plot(ch["cost_ratio_grid"],row,ls=ls,label=f"$L={key}$")
    a.axhline(1,c=".4",ls=":",lw=.9)
    a.set(xlabel=r"Cost $C_0/C_{\rm normal}$",ylabel="Gain / rank-one gain",ylim=(0,1.07))
    a.legend(frameon=False,loc="lower left",fontsize=6.5)
    a=ax[1,0];a.set_title("(d) Pair onset",loc="left")
    p=record["general_spectrum_pair_onset"]
    a.plot(p["decay_rate_sum"],p["exact_onset_cost"],c=ink)
    a.plot(.4,5,"o",ms=4,color="#ba5538")
    a.annotate(r"$a_i=a_k=0.2$",(.4,5),(.9,4.65),
               arrowprops=dict(arrowstyle="-",color=".4"),fontsize=7)
    a.set(xlabel=r"Decay sum $a_i+a_k$",ylabel=r"Onset cost $\Delta C_{\rm on}$")
    a=ax[1,1];a.set_title("(e) Pair target gain",loc="left")
    targets=np.r_[1.,np.geomspace(1.0001,4,80)]
    rows=[]
    for (i,k),c,ls in zip([(.2,.2),(.1,.3),(.04,.36)],["#234b70","#a24e33","#318271"],["-","--","-."]):
        costs=[target_cost(i,k,g)[0] for g in targets]
        rows.append({"a_i":i,"a_k":k,"gain":targets.tolist(),"excess_cost":costs})
        a.plot(targets,costs,c=c,ls=ls,label=f"{i:g}, {k:g}")
    a.legend(title=r"$a_i, a_k$",frameon=False,fontsize=6.5,title_fontsize=6.5,loc="upper left")
    a.set(xlabel=r"Target squared gain $G$",ylabel=r"Minimum excess cost $\Delta C_G$")
    # Rate scale does not affect cost ratios at fixed sum after rescaling t.
    imbalances=np.geomspace(.01,.99,36)
    sweep_gains=1+np.geomspace(1e-6,1e3,55)
    ratios=[]
    for g in sweep_gains:
        equal=target_cost(.2,.2,g)[0]
        ratios.extend(target_cost(.2*(1-d),.2*(1+d),g)[0]/equal for d in imbalances)
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"fig_cost_frontier.{ext}",dpi=240)
    (HERE/"pair_target_results.json").write_text(json.dumps({
        "checks":checks,"curves":rows,"at_G_1":"right-hand onset limit",
        "fixed_decay_sum_sweep":{"decay_sum":.4,"imbalance":imbalances.tolist(),
                                 "gain":sweep_gains.tolist(),
                                 "cost_ratio_to_equal_decay":np.reshape(
                                     ratios,(len(sweep_gains),len(imbalances))).tolist(),
                                 "minimum_ratio":min(ratios)}},indent=2)+"\n")
    print(f"Validated {len(checks)} pair targets, onset limits, and Lyapunov costs")
    print(f"Fixed-sum scan: {len(ratios)} unequal pairs and targets; min ratio {min(ratios):.9g}")


if __name__=="__main__":
    main()
