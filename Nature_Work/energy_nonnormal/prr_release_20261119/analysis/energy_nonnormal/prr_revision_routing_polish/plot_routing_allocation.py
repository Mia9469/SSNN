"""Present the saved allocation results without modifying the underlying study.

Only the new routing figure and a derived regret summary are written.
"""
from pathlib import Path
import json
import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from finite_time_allocation import scores, sum_optimum

HERE = Path(__file__).resolve().parent
INK, SLOW, FAST, TEAL = "#182630", "#a0422b", "#245e94", "#29776e"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.1, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.2, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42})


def main():
    source = HERE / "finite_time_allocation_results.json"
    before = source.read_bytes()
    record = json.loads(before)
    row = next(r for r in record["allocation"]["examples"] if r["T"] == 3.)
    f, p, fixed = (np.asarray(row[k]) for k in ("ceiling", "price", "fixed"))
    budget = sum(fixed)
    rows = []
    for b in np.unique(np.r_[np.linspace(0., 3., 151), budget]):
        optimum = sum_optimum(b, f, p)
        uopt = sum(scores(optimum, f, p))
        ufix = sum(scores(b*fixed/budget, f, p))
        ueq = sum(scores(np.repeat(b/2, 2), f, p))
        assert uopt >= max(ufix, ueq)-1e-10
        rows.append({"budget":float(b),"optimal_utility":float(uopt),
            "fixed_regret":float(max(0., uopt-ufix)),
            "equal_regret":float(max(0., uopt-ueq))})
    at = next(r for r in rows if r["budget"] == budget)
    np.testing.assert_allclose(at["fixed_regret"],
        row["joint_optimum"]**2-row["joint_fixed"]**2, rtol=2e-10)
    summary = {"source_sha256":hashlib.sha256(before).hexdigest(),
        "duration":3.,"reference_budget":float(budget),
        "reference_fixed_dprime":row["joint_fixed"],
        "reference_optimal_dprime":row["joint_optimum"],
        "reference_fixed_regret":at["fixed_regret"],
        "reference_equal_regret":at["equal_regret"],"rows":rows}

    fig, axes = plt.subplots(2,2,figsize=(7.4,5.3))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.89,hspace=.53,wspace=.36)
    fig.suptitle("Noise timescales and tasks determine routing allocation",
                 x=.02,ha="left",fontsize=12.5)
    a,b,c,d=axes.ravel()
    a.set_title("(a) Recurrent pathways retain exact pricing",loc="left",fontweight="semibold")
    a.axis("off");a.set(xlim=(0,1),ylim=(0,1))
    for x,name in [(.26,"Source module\nStable $A_i$"),(.78,"Output module\nStable $D_i$")]:
        a.add_patch(FancyBboxPatch((x-.19,.48),.38,.33,boxstyle="round,pad=.01",
                                  fc="#eef2f4",ec=INK))
        a.text(x,.65,name,ha="center",va="center",fontsize=8.5)
        a.annotate("",xy=(x+.12,.48),xytext=(x-.12,.48),
            arrowprops={"arrowstyle":"->","connectionstyle":"arc3,rad=.55","color":INK})
    a.annotate("",xy=(.57,.64),xytext=(.46,.64),
               arrowprops={"arrowstyle":"->","color":TEAL,"lw":1.6})
    a.text(.52,.87,r"$m_iB_i$",ha="center",color=TEAL)
    a.text(.02,.29,r"Cost: $x_i=\kappa_i m_i^2$",fontsize=9)
    a.text(.02,.08,r"Readout: $d_i'^2=F_i x_i/(x_i+C_{*,i})$",fontsize=9)

    b.set_title("(b) Local noise predicts a crossing",loc="left",fontweight="semibold")
    for r,col,ls,name in zip(record["spectral"]["examples"],
            (INK,TEAL,SLOW,FAST),("-","--","-.",":"),("OU","OU mix.","Osc. mix.","Control")):
        label=f"{name}: {r['instantaneous_ratio']:g} $\\to$ {r['zero_frequency_ratio']:.4g}"
        b.semilogx(r["T"],r["price_ratio"],c=col,ls=ls,lw=1.3,label=label)
    rho=record["spectral"]["allocation_ratio"]
    b.axhline(rho,c=".55",ls="--",lw=.9)
    b.text(.025,7.8,f"Budget ratio = {rho:.2f}",fontsize=7.4,color=".4")
    b.set(xlim=(.02,40),ylim=(2.8,24),xlabel="Window duration $T$",
          ylabel=r"Local-noise variance ratio $L_1/L_2$")
    b.legend(title="Endpoints: short $\\to$ long",title_fontsize=7.2,
             frameon=False,loc="upper left",fontsize=7.0,handlelength=1.8)

    c.set_title("(c) The task changes the budget shares",loc="left",fontweight="semibold")
    ex=record["allocation"]["examples"];ts=[r["T"] for r in ex]
    c.plot(ts,[r["fair_optimal"][0]/r["budget"] for r in ex],c=SLOW,lw=1.5)
    c.plot(ts,[r["fixed"][0]/r["budget"] for r in ex],c=".55",ls=":",lw=1.2)
    c.plot(ts,[r["joint_optimal"][0]/r["budget"] for r in ex],c=FAST,ls="--",lw=1.5)
    cross=record["allocation"]["fair_allocation_crossing"]
    c.axvline(cross,c=".6",ls=":",lw=.8)
    c.text(cross+.10,.742,r"$T_\times=1.399$",fontsize=7.4)
    c.text(2.75,.947,"Protect worst output",c=SLOW,fontsize=7.6)
    c.text(3.05,.844,"Fixed allocation",c=".45",fontsize=7.6)
    c.text(2.8,.704,"Joint discrimination",c=FAST,fontsize=7.6)
    c.set(xlim=(0,5.5),ylim=(.5,.98),xlabel="Window duration $T$",
          ylabel="Slow-output budget share")

    d.set_title("(d) What does reallocation gain?",loc="left",fontweight="semibold")
    bx=[r["budget"] for r in rows]
    d.plot(bx,[r["fixed_regret"] for r in rows],c=SLOW,ls="-.",lw=1.5,label="Fixed-ratio loss")
    d.plot(bx,[r["equal_regret"] for r in rows],c=TEAL,ls="--",lw=1.5,label="Equal-share loss")
    d.axhline(0,c=INK,lw=.8)
    d.axvline(budget,c=".6",ls=":",lw=.9)
    d.plot(budget,at["fixed_regret"],"o",c=SLOW,ms=4)
    d.annotate("Squared-discrimination gain\n"+r"$\Delta d_{\rm joint}'^2=0.0830$",xy=(budget,at["fixed_regret"]),
               xytext=(1.40,.152),fontsize=7.3,color=SLOW,
               arrowprops={"arrowstyle":"-","lw":.7,"color":SLOW})
    d.set(xlim=(0,3),ylim=(-.009,.28),xlabel="Total excess variance budget $B$",
          ylabel=r"Squared-discrimination loss $U^*-U$")
    d.legend(frameon=False,loc="upper right",fontsize=7.2,handlelength=1.8)
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"fig_prr4_allocation_polished.{ext}",dpi=225)
    plt.close(fig)
    assert source.read_bytes() == before
    (HERE/"routing_allocation_presentation.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"},indent=2))


if __name__=="__main__":
    main()
