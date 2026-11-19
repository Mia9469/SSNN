"""Build the three argument figures without changing the numerical records."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from plot_cost_frontier import target_cost
from window_discrimination import average_stats
from window_crossover import models, readout, signal

HERE = Path(__file__).resolve().parent
INK, MUTED, PALE = "#182630", "#4d606b", "#e9eef1"
RUST, BLUE, TEAL, GOLD, GRAY = "#a0422b", "#245e94", "#29776e", "#ecd9aa", "#788892"
LS = ("-", "--", "-.", (0,(1,2)), (0,(4,1.2,1,1.2)))
plt.rcParams.update({"font.size":9,"axes.titlesize":9.7,"axes.labelsize":9,
    "axes.spines.top":False,"axes.spines.right":False,"axes.labelcolor":INK,
    "text.color":INK,"axes.edgecolor":INK,"pdf.fonttype":42,
    "xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.6,"axes.titlepad":7})


def load(name):
    return json.loads((HERE/name).read_text())


def save(fig, stem):
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"{stem}.{ext}",dpi=225)
    plt.close(fig)


def heading(fig,title,subtitle):
    fig.suptitle(title,x=.014,ha="left",fontsize=13,fontweight="semibold")
    fig.text(.014,1-.38/fig.get_figheight(),subtitle,color=MUTED,fontsize=8.6,ha="left")


def panel(ax,label):
    ax.set_title(label,loc="left",fontweight="semibold")


def diagram(ax):
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.set_axis_off()


def arrow(ax,xy0,xy1,*,color=INK,lw=1.3,style="-",head="-|>"):
    ax.annotate("",xy=xy1,xytext=xy0,arrowprops=dict(arrowstyle=head,lw=lw,color=color,
        linestyle=style,shrinkA=0,shrinkB=0,mutation_scale=9))


def node(ax,x,y,w,h,label,*,color=INK,fill="white"):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=.009",
        ec=color,fc=fill,lw=.85))
    ax.text(x,y,label,ha="center",va="center",fontsize=8.6,color=color)


def cost_figure():
    r=json.loads((HERE.parent/"fixed_budget_selection_results.json").read_text())
    p=load("pair_target_results.json")
    fig=plt.figure(figsize=(7.4,5.36))
    heading(fig,"Transient amplification has a variance cost",
            "Whole-state response to a brief perturbation   |   Independent background drive")
    gs=fig.add_gridspec(2,2,left=.097,right=.988,bottom=.095,top=.857,hspace=.61,wspace=.35)
    a,b,c,d=[fig.add_subplot(gs[i,j]) for i,j in np.ndindex(2,2)]
    diagram(a)
    panel(a,"(a) The same link carries cue and noise")
    for x,label in [(.32,"Source $v$"),(.79,"Target $u$")]:
        node(a,x,.56,.21,.16,label,fill="#f0f4f7")
        a.text(x,.89,"Local noise",ha="center",color=MUTED,fontsize=8.2)
        arrow(a,(x,.83),(x,.665),color=GRAY,style="--")
        a.text(x,.379,r"decay $a$",ha="center",fontsize=8.2,color=MUTED)
    a.text(.081,.65,"Cue",color=BLUE,fontsize=8.5,ha="center")
    arrow(a,(.015,.56),(.201,.56),color=BLUE)
    arrow(a,(.44,.56),(.668,.56),lw=1.6,color=TEAL)
    a.text(.55,.65,r"strength $s$",color=TEAL,ha="center",fontsize=8.3)
    a.plot([.015,.985],[.318,.318],c=PALE,lw=.9)
    a.text(.015,.248,r"Mean target response: $s t e^{-at}$",color=BLUE)
    a.text(.015,.108,r"Extra stationary variance: $s^2/(4a^3)$",color=TEAL)

    panel(b,"(b) One channel gives higher gain")
    cols=(INK,BLUE,TEAL,RUST,GRAY)
    for (key,row),col,style in zip(r["fixed_budget_curves"].items(),cols,LS):
        x=np.asarray(row["cost_ratio"])-1
        y=row["Gmax"]
        b.plot(x,y,c=col,ls=style,lw=1.65 if key=="1" else 1.25,label=key)
        if key in ("1","2","4"):
            b.plot(.10,np.interp(.10,x,y),"o",c=col,mec="white",mew=.45,ms=4.6,zorder=5)
    b.axhline(1,color=GRAY,lw=.8,ls=":")
    b.axvline(.10,color=GRAY,lw=.9,ls=":")
    b.text(.108,3.65,"Same budget",color=MUTED,fontsize=8)
    b.set(xlim=(-.004,.238),ylim=(.87,4.65),xticks=[0,.05,.1,.15,.2],
          xlabel=r"Extra variance budget $\Delta C/C_{\rm normal}$",
          ylabel=r"Peak squared gain $G_{\max}^{(2)}$")
    b.legend(title="Channels sharing budget",frameon=False,loc="upper left",
             ncol=2,columnspacing=.9,handlelength=1.65,title_fontsize=7.6)

    panel(c,"(c) Decay rates set the onset")
    diagram(c)
    c.text(.012,.91,r"Source $a_k$",fontsize=8.8)
    arrow(c,(.264,.914),(.382,.914),color=TEAL)
    c.text(.412,.91,r"Target $a_i$",fontsize=8.8)
    c.text(.012,.763,r"Coupling threshold $2\sqrt{a_i a_k}$",fontsize=8.8)
    c.text(.012,.642,r"Extra variance at threshold $2/(a_i+a_k)$",fontsize=8.8)
    c.plot([.01,.99],[.548,.548],c=PALE,lw=.9)
    for x,t in zip((.023,.435,.82),(r"Pair $(a_i,a_k)$","Coupling","Variance")):
        c.text(x,.482,t,color=MUTED,fontsize=8)
    palette=(BLUE,TEAL,RUST)
    for k,(row,col) in enumerate(zip(p["curves"],palette)):
        ai,ak=row["a_i"],row["a_k"]
        y=.364-.12*k
        c.plot([.012,.058],[y+.013]*2,c=col,lw=2,ls=LS[k])
        c.text(.085,y,f"{ai:.2f}, {ak:.2f}")
        c.text(.48,y,f"{2*np.sqrt(ai*ak):.3f}")
        c.text(.865,y,"5")
    c.text(.013,.004,"Equal rate sums; equal onset variance.",fontsize=8.1,color=MUTED)

    panel(d,"(d) Same onset, different target costs")
    for row,col,style in zip(p["curves"],palette,LS):
        d.plot(row["gain"],row["excess_cost"],c=col,ls=style,lw=1.55)
        at2=target_cost(row["a_i"],row["a_k"],2.)[0]
        d.plot(2.,at2,"o",ms=4.4,mec="white",mew=.45,c=col,zorder=4)
    d.axvline(2,color=GRAY,lw=.8,ls=":")
    d.text(1.11,59,"Same pairs as (c)",fontsize=8.1,color=MUTED)
    d.text(2.09,27.7,"27.02",color=RUST,fontsize=8,va="top")
    d.text(2.10,13.2,"15.63",color=BLUE,fontsize=8,va="top")
    d.set(xlim=(.94,4.03),ylim=(2,66),xticks=[1,2,3,4],
          xlabel=r"Target peak squared gain $G$",ylabel=r"Minimum extra variance $\Delta C_G$")
    save(fig,"fig_prr1_cost")

    fig,ax=plt.subplots(figsize=(4.55,2.55),layout="constrained")
    ch=r["higher_order_chains"]
    for (key,row),col,ls in zip(ch["ratio_to_rank_one"].items(),palette,LS):
        ax.plot(np.asarray(ch["cost_ratio_grid"])-1,row,c=col,ls=ls,label=f"{key} nodes")
    ax.axhline(1,c=GRAY,ls=":",lw=.85)
    ax.set(xlabel=r"Extra variance budget $\Delta C/C_{\rm normal}$",
           ylabel="Peak gain / one-channel gain",ylim=(0,1.08))
    ax.legend(frameon=False,loc="lower left")
    save(fig,"fig_prrS_chain")


def readout_grid(title,subtitle):
    fig=plt.figure(figsize=(7.4,6.35))
    heading(fig,title,subtitle)
    gs=fig.add_gridspec(3,2,height_ratios=[.68,1,1],left=.103,right=.98,
                        bottom=.078,top=.855,hspace=.57,wspace=.36)
    return fig,[fig.add_subplot(gs[0,:])]+[fig.add_subplot(gs[i,j]) for i in (1,2) for j in (0,1)]


def cue_figure():
    record=load("window_results.json")
    a_rate=record["a"]
    start,end=record["window_results"]["peak_centered"]["window_at"]
    state=record["window_results"]["peak_centered"]["states"][0]
    gains=np.asarray(record["gain_profiles"])
    fig,(a,b,c,d,e)=readout_grid("Routing gains yield diminishing reliability returns",
                                "Fixed outputs   |   Equal decay rates   |   Read out two cues in noise")
    panel(a,"(a) Swap branch gains without changing wiring or total variance")
    diagram(a)
    cols=(RUST,BLUE)
    for k,x in enumerate((.04,.535)):
        a.text(x,.81,f"State {'AB'[k]}",fontweight="semibold",fontsize=9.1)
        node(a,x+.055,.445,.106,.22,r"Cue $\pm b$",fill="#f0f4f7")
        for i,y in enumerate((.58,.305)):
            m=gains[k,i]
            arrow(a,(x+.12,.445),(x+.269,y),color=cols[i],lw=2.15 if m>1 else .95)
            a.text(x+.185,.445+.68*(y-.445),f"{m:g}",color=cols[i],fontsize=8.1,
                   va="bottom" if i==0 else "top",ha="center")
            node(a,x+.349,y,.136,.16,f"Output {i+1}",color=cols[i])
    a.plot([.5,.5],[.21,.84],color=PALE,lw=1)
    a.text(.5,.064,r"Both states: total stationary variance $22.34$",ha="center",color=MUTED,fontsize=8.5)

    panel(b,"(b) Average within one fixed window")
    t=np.linspace(0,4,301)
    b.axvspan(start,end,color=GOLD,alpha=.55)
    for i in range(2):
        b.plot(t,gains[0,i]*t/a_rate*np.exp(-t),color=cols[i],ls=LS[i],lw=1.55,label=f"Output {i+1}")
    b.text(1.,1.235,"Readout\nwindow",ha="center",va="bottom",fontsize=7.6,color=MUTED)
    b.legend(frameon=False,loc="upper right",handlelength=1.55)
    b.set(xlim=(0,4),ylim=(0,1.55),xlabel=r"Time after cue $at$",ylabel="Mean response to positive cue")

    panel(c,"(c) The two cue distributions overlap")
    x=np.linspace(-4.6,4.6,450)
    for i,y in enumerate((1.01,.06)):
        mu=state["mean_positive"][i]
        sd=np.sqrt(state["variance"][i])
        c.axhline(y,c=PALE,lw=.8)
        for sg,ls in [(-1,"--"),(1,"-")]:
            pdf=.39*np.exp(-.5*((x-sg*mu)/sd)**2)/sd
            c.plot(x,y+pdf,c=cols[i],ls=ls,lw=1.35)
            c.plot([sg*mu,sg*mu],[y-.023,y+.041],c=cols[i],lw=.8)
        c.text(-4.51,y+(.35 if i==0 else .63),
               rf"Output {i+1}: $d'={state['dprime'][i]:.2f}$",color=cols[i],fontsize=8.1)
    c.legend(handles=[Line2D([0],[0],c=MUTED,ls="--",label=r"$-b$"),
                      Line2D([0],[0],c=MUTED,ls="-",label=r"$+b$")],
             frameon=False,loc="upper right",ncol=2,handlelength=1.4,columnspacing=.75)
    c.text(4.43,.824,r"$d'$: mean gap / noise SD",ha="right",fontsize=7.5,color=MUTED)
    c.set(xlim=(-4.6,4.6),ylim=(0,1.78),xlabel="Window-averaged activity",yticks=[])
    c.spines["left"].set_visible(False)

    panel(d,"(d) Discrimination saturates with cost")
    cstar=state["half_squared_ceiling_cost"]
    x_cost=np.geomspace(.025,12,180)
    d.plot(x_cost,np.sqrt(x_cost/(1+x_cost)),c=INK,lw=1.5)
    d.axhline(1,c=GRAY,ls=":",lw=.85)
    d.text(.028,1.027,"Shared-noise ceiling",color=MUTED,fontsize=8)
    for i,(g,f) in enumerate(zip(gains[0],state["fraction_of_ceiling"])):
        x0=(g*g/(4*a_rate**3))/cstar
        d.plot(x0,f,"o",c=cols[i],mec="white",mew=.55,ms=6,zorder=4)
        d.text(x0*(.91 if i==0 else 1.28),f+(-.14 if i==0 else -.09),
               f"{100*f:.0f}%",ha="right" if i==0 else "left",color=cols[i],fontsize=8.5)
    d.set(xscale="log",xlim=(.025,12),ylim=(0,1.11),
          xlabel=r"Branch variance cost $C_i/C_*$",ylabel=r"Fraction of ceiling $d'_i/d'_\infty$")
    d.set_xticks([.03,.1,.3,1,3,10],["0.03","0.1","0.3","1","3","10"])

    panel(e,"(e) Equal rates: no ranking reversal")
    tw=np.linspace(.04,5,160)
    scores=[average_stats(a_rate,gains[0],0,T/a_rate) for T in tw]
    ref=state["dprime_ceiling"]
    for i,col in enumerate(cols):
        e.plot(tw,[s["dprime"][i]/ref for s in scores],c=col,ls=LS[i],lw=1.5,label=f"Output {i+1}")
    e.plot(tw,[s["joint_dprime"]/ref for s in scores],c=INK,ls=":",lw=1.25)
    for i,col in enumerate(cols):
        e.text(5.12,scores[-1]["dprime"][i]/ref,f"Output {i+1}",
               color=col,va="center",fontsize=7.6)
    e.plot([.17,.53],[.976,.976],c=INK,ls=":",lw=1.2)
    e.text(.61,.976,"Joint, either state",va="center",fontsize=7.8)
    e.set(xlim=(0,6.33),xticks=np.arange(6),ylim=(0,1.045),xlabel=r"Window length from cue $aT$",
          ylabel=r"Cue separation $d'/d'_{\rm ref}$")
    save(fig,"fig_prr2_routing_polished")


def reversal_figure():
    r=load("window_crossover_results.json")
    model=models((0.,0.),0.)
    fig,(a,b,c,d,e)=readout_grid("Readout time changes the value of a fixed allocation",
                                 "Two separate sources receive the same cue   |   Wiring and gains stay fixed during readout")
    panel(a,"(a) Use the same two decay rates in opposite positions")
    diagram(a)
    for i,y in enumerate((.662,.337)):
        col=(RUST,BLUE)[i]
        aa,bb=((2.,.5),(.5,2.))[i]
        node(a,.21,y,.151,.194,rf"Source: $a={aa:g}$",fill="#f0f4f7")
        arrow(a,(.291,y),(.459,y),lw=2 if i==0 else 1,color=col)
        a.text(.376,y+.052,("2.34","0.90")[i],color=col,ha="center",va="bottom",fontsize=8)
        node(a,.555,y,.177,.194,f"{('Slow','Fast')[i]} output: "+f"{bb:g}",color=col)
        arrow(a,(.102,.496),(.124,y),lw=.85)
        arrow(a,(.555,y+.17),(.555,y+.108),color=GRAY,lw=.8,style="--")
    a.text(.092,.491,r"Cue $\pm b$",ha="right",va="center",fontsize=8.7)
    a.text(.555,.896,"Local output noise",ha="center",color=MUTED,fontsize=7.6)
    a.plot([.677,.677],[.198,.887],c=PALE,lw=1)
    a.text(.718,.775,"Two readout lengths",fontsize=8.5,fontweight="semibold")
    arrow(a,(.723,.568),(.804,.568),color=MUTED,lw=1.2,head="|-|")
    arrow(a,(.723,.321),(.953,.321),color=MUTED,lw=1.2,head="|-|")
    a.text(.815,.565,r"Short: $T=0.5$",va="center",fontsize=8.2,color=MUTED)
    a.text(.723,.194,r"Long: $T=3$",fontsize=8.2,color=MUTED)
    a.text(.012,.032,"One fixed gain state: only the readout window changes below.",
           fontsize=8.2,color=MUTED)

    panel(b,"(b) Mean-response ranking stays fixed")
    t=np.linspace(0,5,271)
    for v,col,ls,label in zip(model,(RUST,BLUE),LS,("Slow output","Fast output")):
        b.plot(t,signal(v[0],t),c=col,ls=ls,lw=1.6,label=label)
    b.legend(frameon=False,loc="upper right",handlelength=1.65)
    b.text(2.48,.32,r"Slow signal $=2.6\times$ fast",fontsize=8.1,color=MUTED)
    b.set(xlim=(0,5),ylim=(0,.6),xlabel="Time after cue",ylabel="Mean response to positive cue")

    panel(c,"(c) The better readout changes")
    T=np.linspace(.022,5.5,131)
    def dr(t0):
        s,f=(readout(v,t0,"box")["dprime"] for v in model)
        return s/f
    tc=r["box_crossing"]
    ratios=np.array([dr(t0) for t0 in T])
    lo,hi=min(.91,min(ratios)-.009),max(1.123,max(ratios)+.009)
    c.axhspan(1,hi,fc=RUST,alpha=.055)
    c.axhspan(lo,1,fc=BLUE,alpha=.055)
    c.axhline(1,c=GRAY,ls=":",lw=.9)
    c.axvline(tc,c=GRAY,ls=":",lw=.9)
    c.plot(T,ratios,c=INK,lw=1.6)
    for t0,label,dy,col in [(.5,"Short",.013,RUST),(3,"Long",.014,BLUE)]:
        c.plot(t0,dr(t0),"o",c=col,ms=5,mec="white",mew=.5)
        c.text(t0+.13,dr(t0)+dy,label,color=col,fontsize=8)
    c.text(3.37,1.061,"Slow output wins",color=RUST,fontsize=8.1)
    c.text(3.38,.968,"Fast output wins",color=BLUE,fontsize=8.1)
    c.text(tc+.1,1.008,r"$T_\times=1.399$",color=MUTED,fontsize=7.7)
    c.set(xlim=(0,5.5),ylim=(lo,hi),xlabel=r"Readout window $T$",
          ylabel=r"Cue separation ratio $d'_{\rm slow}/d'_{\rm fast}$")

    panel(d,"(d) The budget-price crossing")
    def v(a,t):
        return (a*t+np.expm1(-a*t))/(a**3*t*t)
    gain=(r["m_high"]/r["m_low"])**2
    d.plot(T,v(r["a_slow"],T)/v(r["a_fast"],T),c=INK,lw=1.5,label="Cost of equal discrimination")
    d.axhline(gain,c=TEAL,ls="--",lw=1.3,label="Actual budget allocation")
    d.axvline(tc,c=GRAY,ls=":",lw=.85)
    d.plot(tc,gain,"o",c=INK,ms=4)
    d.legend(frameon=False,loc="lower right",handlelength=1.6)
    d.text(.13,11.8,"Remove local noise: the readouts tie",fontsize=7.9,color=MUTED)
    d.set(xlim=(0,5.5),ylim=(3.4,12.6),xlabel=r"Readout window $T$",ylabel="Slow / fast budget ratio")

    panel(e,"(e) Does the reversal survive?")
    selections=[("Baseline","box",(0.,0.),0.),
                ("Return on both","box",(.10,.10),0.),
                ("Colored noise","box",(0.,0.),1.),
                ("Triangle weights","triangle",(0.,0.),0.),
                ("Exp. weights","exponential",(0.,0.),0.),
                ("Return on one","box",(.10,0.),0.),
                ("Both + colored","box",(.10,.10),1.)]
    for i,(label,shape,fb,color) in enumerate(selections):
        row=next(x for x in r["tests"] if x["window"]==shape and
                 tuple(x["feedback_over_a_slow"])==fb and x["tau_times_a_fast"]==color)
        x,y=[v["dprime_ratio"] for v in row["test_points"]]
        e.plot([x,y],[i,i],c=GRAY,lw=.72,alpha=.58)
        e.plot(x,i,"o",c=RUST,ms=4.3,label="Short" if i==0 else None)
        e.plot(y,i,"s",mfc="white",mec=BLUE,ms=4.15,label="Long" if i==0 else None)
    e.axvline(1,c=GRAY,ls=":",lw=.9)
    e.set(xlim=(.897,1.085),ylim=(6.55,-1.5),xticks=[.9,.95,1.,1.05],yticks=np.arange(7),
          xlabel=r"Cue separation ratio $d'_{\rm slow}/d'_{\rm fast}$")
    e.set_yticklabels([s[0] for s in selections],fontsize=7.3)
    e.legend(frameon=False,loc="upper center",ncol=2,handletextpad=.35,columnspacing=.9)
    save(fig,"fig_prr3_fixed_state_polished")


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=("1", "2", "3"),
                        default=("2", "3"),
                        help="Default: polished Figures 2 and 3; Figure 1 requires explicit selection.")
    args = parser.parse_args()
    for key, draw in (("1",cost_figure),("2",cue_figure),("3",reversal_figure)):
        if key in args.figures:
            draw()
