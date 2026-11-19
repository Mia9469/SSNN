"""Mouse-level held-out scores; models and sessions are paired within mouse."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    old = json.loads((HERE.parent/"alm_multisession_results.json").read_text())
    new = json.loads((HERE/"alm_normal_d12.json").read_text())
    mice = sorted(new["mice"])
    old_by_session = {r["file"]:r["full_geometry_residual_spearman"] for r in old["sessions"]}
    for r in new["sessions"]:
        assert abs(r["full"]["r"]-old_by_session[r["file"]])<1e-10
    values = np.array([
        [np.mean([r["same_spectrum_normal_median_spearman"] for r in old["sessions"] if r["mouse"]==m]),
         new["mice"][m]["normal"]["r"], new["mice"][m]["normal_window_fit"]["r"],
         new["mice"][m]["full"]["r"]] for m in mice
    ])
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42})
    fig, ax = plt.subplots(figsize=(5.2,2.85),layout="constrained")
    for i,y in enumerate(values):
        ax.plot(np.arange(4),y,marker=["o","s","^","D","v","P"][i],lw=.85,ms=4,
                color=plt.get_cmap("tab10")(i),alpha=.8,label=f"Mouse {i+1}")
    med = np.median(values,axis=0)
    ax.plot(np.arange(4),med,"k_",ms=12,mew=2,label="Mouse median")
    ax.axhline(0,c=".55",lw=.75,ls=":")
    ax.set_xticks(np.arange(4),["Random\nnormal","One-step\nnormal fit","Window\nnormal fit","Full\noperator"])
    ax.set_ylabel("Held-out energy Spearman correlation")
    ax.legend(frameon=False,fontsize=7,loc="lower center",bbox_to_anchor=(.5,1.005),ncol=4)
    for ext in ("pdf","png"):
        fig.savefig(HERE/f"fig_alm_normal_comparison.{ext}",dpi=220)


if __name__=="__main__":
    main()
