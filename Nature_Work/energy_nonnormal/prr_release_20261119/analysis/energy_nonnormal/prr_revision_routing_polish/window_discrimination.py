"""Finite-window cue discrimination for the unchanged gain-routing example."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gain_routing import build_model, group_vector


def average_stats(a, gains, start, end, cue=1.):
    """Stationary Gaussian noise and a fixed boxcar, including all time lags."""
    length = end-start
    h = quad(lambda t: t*np.exp(-a*t), start, end)[0]/length
    v0 = quad(lambda u: (length-u)*np.exp(-a*u)/a, 0, length)[0]/length**2
    v1 = quad(lambda u: (length-u)*np.exp(-a*u)*(1+a*u)/(2*a**3),
              0, length)[0]/length**2
    means = cue*h*gains
    cov = v0*np.eye(len(gains)) + v1*np.outer(gains,gains)
    d = 2*means/np.sqrt(np.diag(cov))
    joint = 2*np.sqrt(means@np.linalg.solve(cov,means))
    ceiling = 2*cue*h/np.sqrt(v1)
    fraction = gains*np.sqrt(v1)/np.sqrt(np.diag(cov))
    channel_cost = gains**2/(4*a**3)
    half_squared_ceiling_cost = v0/(4*a**3*v1)
    np.testing.assert_allclose(d/ceiling, fraction)
    np.testing.assert_allclose(fraction**2,
                               channel_cost/(channel_cost+half_squared_ceiling_cost))
    np.testing.assert_allclose(half_squared_ceiling_cost*fraction**2/(1-fraction**2),
                               channel_cost)
    return {"mean_positive": means.tolist(), "variance": np.diag(cov).tolist(),
            "dprime": d.tolist(), "accuracy": ndtr(d/2).tolist(),
            "joint_dprime": float(joint), "joint_accuracy": float(ndtr(joint/2)),
            "dprime_ceiling": float(ceiling), "fraction_of_ceiling": fraction.tolist(),
            "half_squared_ceiling_cost": float(half_squared_ceiling_cost)}


def main():
    model = build_model()
    a = model["a"]
    high, low = model["gain_high"], model["gain_low"]
    profiles = [np.array([high,low]),np.array([low,high])]
    outputs = np.array([group_vector(40,20,30),group_vector(40,30,40)])
    cue = group_vector(40,0,10)
    for idx, g in enumerate(profiles):
        j, sigma = model["generators"][idx], model["covariances"][idx]
        for u in [0., .2, 1., 2.5]:
            p = expm(j*u)
            analytic = np.exp(-a*u)*(np.eye(2)/(2*a) +
                                       np.outer(g,g)*(1+a*u)/(4*a**3))
            np.testing.assert_allclose(outputs@p@sigma@outputs.T,analytic,atol=1e-12)
            np.testing.assert_allclose(outputs@p@cue,g*u*np.exp(-a*u),atol=1e-12)
    lo, hi = [brentq(lambda x:x*np.exp(1-x)-.5,*bounds) for bounds in [(0,1),(1,6)]]
    windows = {"early":[0,.5], "peak_centered":[.5,1.5], "whole_response":[0,3], "late":[2,3]}
    results = {}
    for name,(start,end) in windows.items():
        r = [average_stats(a,g,start/a,end/a) for g in profiles]
        assert abs(r[0]["joint_dprime"]-r[1]["joint_dprime"])<1e-12
        np.testing.assert_allclose(r[0]["dprime"],r[1]["dprime"][::-1])
        results[name] = {"window_at": [start,end], "states":r}
    record = {"a":a,"unit_cue_amplitude":1.,"noise":"stationary, isotropic white drive",
              "channel_amplitude_half_peak_at":[lo,hi], "window_results":results,
              "gain_profiles": [g.tolist() for g in profiles],
              "stationary_costs":model["costs"]}
    (HERE/"window_results.json").write_text(json.dumps(record,indent=2)+"\n")
    plt.rcParams.update({"font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42})
    fig, axes = plt.subplots(2,2,figsize=(7.2,4.65),layout="constrained")
    ax = axes.ravel()
    t = np.linspace(0,4,401)
    colors = ["#215F9D","#B14B2B"]
    style = ["-", "--"]
    ax[0].set(xlim=(0,1),ylim=(0,1),title="(a) Same wiring, opposite branch gains")
    ax[0].set_axis_off()
    ax[0].text(.08,.93,"State",color=".36",ha="center")
    ax[0].text(.33,.93,"Input",color=".36",ha="center")
    ax[0].text(.85,.93,"Fixed outputs",color=".36",ha="center")
    for k,y in enumerate((.69,.29)):
        ax[0].text(.08,y,"AB"[k],weight="bold",ha="center",va="center")
        ax[0].add_patch(Circle((.33,y),.043,fc="white",ec=".3",transform=ax[0].transData))
        ax[0].text(.33,y,r"$b$",ha="center",va="center")
        ax[0].text(.21,y,r"$\pm$",ha="center",va="center")
        for i,dy in enumerate((.118,-.118)):
            gy = profiles[k][i]
            col = colors[i]
            ax[0].annotate("",(.78,y+dy),(.39,y),arrowprops={
                "arrowstyle":"-|>", "color":col,
                "lw":2.2 if gy==high else .9, "linestyle":"-"})
            ax[0].text(.59,y+.68*dy, f"{gy:g}", ha="center",
                       va="bottom" if dy>0 else "top",color=col)
            ax[0].text(.83,y+dy,rf"$c_{i+1}$",va="center",ha="center",color=col)
    ax[0].text(.53,.04,"Same total variance in both states",ha="center",color=".32")

    ax[1].axvspan(.5,1.5,color=".91",label="Window used in (c)")
    for i,(gi,c,ls) in enumerate(zip(profiles[0],colors,style)):
        ax[1].plot(t,gi*(t/a)*np.exp(-t),c=c,ls=ls,label=f"Output {i+1}")
    # Mark duration at half peak separately from the readout window.
    ax[1].plot([lo,lo,hi,hi],[.045,.018,.018,.045],c=".35",lw=.8)
    ax[1].text((lo+hi)/2,.06,"Above half peak",ha="center",fontsize=7,color=".3")
    ax[1].set(title="(b) Response to a positive cue",xlabel=r"Time after cue, $at$",
              ylabel="Mean response in state A",ylim=(0,1.48))
    ax[1].legend(frameon=False,fontsize=7.2,loc="upper right")

    r = results["peak_centered"]["states"]
    for k in range(2):
        ax[2].bar(np.arange(2)+(k-.5)*.34,r[k]["fraction_of_ceiling"],width=.31,
                  color=".27" if k==0 else "white",edgecolor=".27",
                  hatch="" if k==0 else "////",label=f"State {'AB'[k]}")
    ax[2].set_xticks([0,1],["Output 1","Output 2"])
    ax[2].axhline(1.,color=".5",ls=":",lw=.9)
    ax[2].set(title=r"(c) Readout in the reference window",ylabel=r"Cue separation, $d'/d'_{\rm ref}$",
              ylim=(0,1.22))
    ax[2].legend(frameon=False,fontsize=7.2,loc="upper center",ncol=2)

    horizons = np.linspace(.05,5,160)
    stats = [average_stats(a,profiles[0],0,h/a) for h in horizons]
    ref = r[0]["dprime_ceiling"]
    for k,(c,ls,label) in enumerate(zip(colors,style,["Stronger branch","Weaker branch"])):
        ax[3].plot(horizons,[x["dprime"][k]/ref for x in stats],color=c,ls=ls,label=label)
    ax[3].plot(horizons,[x["joint_dprime"]/ref for x in stats],color=".2",lw=1.15,
               ls=":",label="Joint, either state")
    ax[3].set(title="(d) Varying the window duration",xlabel=r"Window length from cue, $aT$",
              ylabel=r"Cue separation, $d'/d'_{\rm ref}$",ylim=(0,1.14))
    ax[3].legend(frameon=False,fontsize=7.2,loc="lower right")
    for suffix in ("pdf","png"):
        fig.savefig(HERE/f"fig_window_discrimination.{suffix}",dpi=220)
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    main()
