"""
hoyer_compare.py
================
Reviewer-killer baseline: Hoyer sparsity vs our Weight+PopCV regularizer.

Hoyer sparsity measure (Hurley & Rickard 2009; widely used in SNN training,
e.g. Kim et al. 2023):

    H(x) = (sqrt(n) - ||x||_1 / ||x||_2) / (sqrt(n) - 1)

H ∈ [0, 1] is 1 when the vector is maximally sparse (one non-zero) and 0 when
uniform.  Pushing H toward 1 drives a lifetime-sparse firing pattern — a
natural competitor to our heterogeneity-based CV regularizer.

Our hypothesis: for SNNs the right objective is *heterogeneity* (CV), not
*sparsity* (Hoyer / L1).  The two coincide in some regimes but diverge when
neurons have different means.  If Hoyer gives smaller ICE uplift than CV, the
paper's "CV-not-sparsity" framing is supported.

Three conditions × 2 seeds × 200 epochs ≈ 6 GPU-hours on a 3090.

  A. Baseline          : TET loss only
  B. Hoyer (lifetime)  : TET + λ * (-H(fr_per_neuron))   (maximise H)
  C. Weight+PopCV      : TET + WeightCV + PopFR-CV       (our best)

Outputs
  hoyer_compare_results.json
  hoyer_compare.png
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from timm.models import create_model

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from model import sdt          # noqa: registers 'sdt'
from spikingjelly.clock_driven import functional
import criterion_v2

# ── CLI (supports parallel-seed mode matching spikformer_compare) ────────────
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
_ap.add_argument("--tag", type=str, default="")
_ap.add_argument("--epochs", type=int, default=200)
_ap.add_argument("--seed", type=int, default=None,
                 help="run exactly one seed → hoyer_seed<N>.json")
_ap.add_argument("--merge", action="store_true",
                 help="aggregate hoyer_seed*.json into the main results file")
_ap.add_argument("--force", action="store_true")
CLI, _ = _ap.parse_known_args()

DATASET    = CLI.dataset
N_CLASSES  = 100 if DATASET == "cifar100" else 10
TAG        = f"_{CLI.tag}" if CLI.tag else ""

DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

N_EPOCHS    = CLI.epochs
BATCH_TRAIN = 64
BATCH_VAL   = 128
EVAL_EVERY  = 10
LR          = 3e-4
MIN_LR      = 1e-5
WARMUP_EP   = 20
WEIGHT_DECAY = 0.06

LAMBDA_WEIGHT_CV = 0.001
LAMBDA_POP_CV    = 0.0005
LAMBDA_HOYER     = 0.001     # matched to WeightCV magnitude so comparison is fair
REG_START        = WARMUP_EP

N_SEEDS = 2

CONFIGS = [
    {"name": "baseline",   "label": "Baseline",     "reg": None},
    {"name": "hoyer",      "label": "Hoyer",        "reg": "hoyer"},
    {"name": "weight_pop", "label": "Weight+PopCV", "reg": "weight_pop"},
]
COLORS = ["#1565C0", "#EF6C00", "#6A1B9A"]

RESULTS_FILE = os.path.join(REPO, f"hoyer_compare_results{TAG}.json")
FIGURE_FILE  = os.path.join(REPO, f"hoyer_compare{TAG}.png")

print(f"Device  : {DEVICE}  |  dataset: {DATASET.upper()} ({N_CLASSES})")
print(f"Epochs  : {N_EPOCHS}  |  seeds: {N_SEEDS if CLI.seed is None else 1}")
print(f"λ_W={LAMBDA_WEIGHT_CV} λ_pop={LAMBDA_POP_CV} λ_Hoyer={LAMBDA_HOYER}")


# ── Hoyer regularizer on firing-rate vectors ──────────────────────────────────
def hoyer_fr_loss(spike_tensors, lambda_hoyer=LAMBDA_HOYER):
    """Maximise Hoyer sparsity of per-neuron firing rates, averaged over layers.

    For each layer, compute fr = mean over (T, B) of binary spikes → vector of
    shape (N,).  Hoyer measure = (sqrt(n) - L1/L2) / (sqrt(n) - 1).  We add
    `-lambda * H` to the loss so minimisation increases H.
    """
    vals = []
    for s in spike_tensors:
        T_, B = s.shape[0], s.shape[1]
        fr = s.float().view(T_, B, -1).mean(dim=(0, 1))   # (N,)
        n  = fr.numel()
        if n < 2:
            continue
        l1 = fr.abs().sum()
        l2 = fr.pow(2).sum().sqrt() + 1e-8
        sn = float(math.sqrt(n))
        H  = (sn - l1 / l2) / (sn - 1.0)
        vals.append(H)
    if not vals:
        return 0.0
    return -lambda_hoyer * torch.stack(vals).mean()


# ── Data ──────────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
aa_params = dict(translate_const=int(32 * 0.45),
                 img_mean=tuple(int(c * 255) for c in MEAN))
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    rand_augment_transform("rand-m9-n1-mstd0.4-inc1", aa_params),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

_DSCLS = torchvision.datasets.CIFAR100 if DATASET == "cifar100" else torchvision.datasets.CIFAR10
ds_train = _DSCLS(os.path.join(REPO, "data"), train=True,  download=True, transform=train_tf)
ds_val   = _DSCLS(os.path.join(REPO, "data"), train=False, download=True, transform=val_tf)

_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_TRAIN,
                                           shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=BATCH_VAL,
                                           shuffle=False, num_workers=_nw, pin_memory=_gpu)

mixup_fn = Mixup(mixup_alpha=0.5, cutmix_alpha=0.0, prob=1.0,
                 switch_prob=0.5, mode="batch",
                 label_smoothing=0.1, num_classes=N_CLASSES)


def make_model() -> nn.Module:
    return create_model(
        "sdt",
        T=4, num_classes=N_CLASSES, img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=256, num_heads=8, mlp_ratios=4, depths=2,
        sr_ratios=1, pooling_stat="0011", spike_mode="lif",
        dvs_mode=False, TET=True, in_channels=3,
        qkv_bias=False, drop_rate=0.0, drop_path_rate=0.2, drop_block_rate=None,
    ).to(DEVICE)


def get_lr(epoch):
    if epoch < WARMUP_EP:
        return LR * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, N_EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


def tet_loss(outputs, labels, ce_fn):
    return sum(ce_fn(outputs[t], labels) for t in range(outputs.size(0))) / outputs.size(0)


@torch.no_grad()
def evaluate(model):
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr, layer_pop, layer_life = defaultdict(list), defaultdict(list), defaultdict(list)
    layer_hoyer = defaultdict(list)

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        hook = {}
        outputs, hook = model(imgs, hook=hook)
        functional.reset_net(model)
        lm = outputs.mean(0)
        total_loss    += ce_fn(lm, labels).item() * imgs.size(0)
        total_correct += (lm.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)
        for k, s in hook.items():
            T_, B = s.shape[0], s.shape[1]
            flat = s.float().view(T_, B, -1)
            fr   = flat.mean(dim=0)
            layer_fr[k].append(fr.mean().item())
            m  = fr.mean(dim=1, keepdim=True) + 1e-8
            sd = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
            layer_pop[k].append((sd / m).mean().item())
            mean_b = fr.mean(dim=0) + 1e-8
            std_b  = fr.std(dim=0, unbiased=False) + 1e-8
            layer_life[k].append((std_b / mean_b).mean().item())
            # Hoyer measure on per-neuron fr (averaged over batch too)
            v  = fr.mean(dim=0)                       # (N,)
            n  = v.numel()
            l1 = v.abs().sum().item()
            l2 = (v.pow(2).sum().sqrt() + 1e-8).item()
            sn = math.sqrt(n)
            H  = (sn - l1 / l2) / (sn - 1.0)
            layer_hoyer[k].append(H)

    weight_cv = criterion_v2.compute_weight_cv(model)
    acc     = total_correct / total_n * 100.0
    mean_fr = float(np.mean([np.mean(v) for v in layer_fr.values()]))
    cv_fr   = float(np.mean([np.mean(v) for v in layer_pop.values()]))
    cv_life = float(np.mean([np.mean(v) for v in layer_life.values()]))
    hoyer   = float(np.mean([np.mean(v) for v in layer_hoyer.values()]))
    return {"acc": acc, "ce_loss": total_loss / total_n,
            "mean_fr": mean_fr, "cv_fr": cv_fr, "cv_life": cv_life,
            "hoyer": hoyer, "weight_cv": weight_cv,
            "ice": acc / (mean_fr + 1e-8)}


def train_one_epoch(model, opt, config, epoch):
    model.train()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    total, count = 0.0, 0
    reg = config["reg"]
    apply_reg = reg is not None and epoch >= REG_START

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = mixup_fn(imgs, labels)
        opt.zero_grad()
        need_hook = apply_reg and reg in ("hoyer", "weight_pop")
        hook = {} if need_hook else None
        outputs, hook = model(imgs, hook=hook)
        loss = tet_loss(outputs, labels, ce_fn)

        if apply_reg:
            if reg == "hoyer" and hook:
                loss = loss + hoyer_fr_loss(list(hook.values()), lambda_hoyer=LAMBDA_HOYER)
            elif reg == "weight_pop":
                loss = loss + criterion_v2.weight_cv_loss(model, lambda_weight_cv=LAMBDA_WEIGHT_CV)
                if hook:
                    loss = loss + criterion_v2.firing_rate_cv_loss(list(hook.values()),
                                                                   lambda_cv=LAMBDA_POP_CV)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        functional.reset_net(model)
        total += loss.item() * imgs.size(0); count += imgs.size(0)
    return total / count


METRIC_KEYS = ("acc", "ce_loss", "mean_fr", "cv_fr", "cv_life", "hoyer", "weight_cv", "ice")


def run_one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base = make_model()
    models = {c["name"]: copy.deepcopy(base) for c in CONFIGS}
    opts = {c["name"]: torch.optim.AdamW(models[c["name"]].parameters(),
                                          lr=LR, weight_decay=WEIGHT_DECAY)
            for c in CONFIGS}
    hists = {c["name"]: {k: [] for k in (*METRIC_KEYS, "epoch")} for c in CONFIGS}

    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for opt in opts.values():
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        t0 = time.time()
        for c in CONFIGS:
            train_one_epoch(models[c["name"]], opts[c["name"]], c, epoch)
        dt = time.time() - t0

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            ep = epoch + 1
            for c in CONFIGS:
                m = evaluate(models[c["name"]])
                hists[c["name"]]["epoch"].append(ep)
                for k in METRIC_KEYS:
                    hists[c["name"]][k].append(m[k])
            print(f"[seed {seed}] Ep {ep:>3}/{N_EPOCHS}  [{dt:.0f}s]")
            for c in CONFIGS:
                last = {k: hists[c["name"]][k][-1] for k in METRIC_KEYS}
                print(f"    {c['label']:14s} Acc={last['acc']:.2f}  MFR={last['mean_fr']:.4f}  "
                      f"Hoyer={last['hoyer']:.3f}  CV_pop={last['cv_fr']:.3f}  "
                      f"ICE={last['ice']:.1f}")
        else:
            print(f"[seed {seed}] Ep {epoch+1:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{dt:.0f}s]")

    final = {c["name"]: evaluate(models[c["name"]]) for c in CONFIGS}
    return hists, final


def _aggregate(all_hists, all_final):
    avg = {c["name"]: {"epoch": all_hists[0][c["name"]]["epoch"]} for c in CONFIGS}
    for c in CONFIGS:
        for k in METRIC_KEYS:
            M = np.array([h[c["name"]][k] for h in all_hists])
            avg[c["name"]][k] = M.mean(axis=0).tolist()
            avg[c["name"]][k + "_std"] = M.std(axis=0).tolist()
    data = {"hists": avg,
            "configs": [c["name"] for c in CONFIGS],
            "labels":  {c["name"]: c["label"] for c in CONFIGS},
            "n_seeds": len(all_hists),
            "dataset": DATASET}
    for c in CONFIGS:
        data[f"final_{c['name']}_summary"] = {
            k: float(np.mean([f[c["name"]][k] for f in all_final])) for k in METRIC_KEYS}
    return data


def run():
    all_hists, all_final = [], []
    t0 = time.time()
    for s in range(N_SEEDS):
        print(f"\n=== seed {s + 1}/{N_SEEDS} ===")
        h, f = run_one_seed(s)
        all_hists.append(h); all_final.append(f)
    print(f"\nTotal: {(time.time() - t0)/60:.1f} min")
    data = _aggregate(all_hists, all_final)
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results → {RESULTS_FILE}")
    plot(data)


def _seed_file(seed):
    return os.path.join(REPO, f"hoyer_seed{seed}{TAG}.json")


def run_single_seed(seed):
    t0 = time.time()
    hists, final = run_one_seed(seed)
    out = {"seed": seed, "hists": hists, "final": final,
           "configs": [c["name"] for c in CONFIGS],
           "labels": {c["name"]: c["label"] for c in CONFIGS},
           "n_epochs": N_EPOCHS, "dataset": DATASET,
           "runtime_min": (time.time() - t0) / 60}
    path = _seed_file(seed)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[seed {seed}] done → {path}")


def merge_seeds():
    paths = sorted(glob.glob(os.path.join(REPO, f"hoyer_seed*{TAG}.json")))
    if not paths:
        print("No hoyer_seed*.json files to merge."); return
    print(f"Merging {len(paths)} per-seed files")
    per_seed = [json.load(open(p)) for p in paths]
    data = _aggregate([s["hists"] for s in per_seed], [s["final"] for s in per_seed])
    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Merged → {RESULTS_FILE} (n_seeds={len(per_seed)})")
    plot(data)


def plot(data):
    hists, names, labels = data["hists"], data["configs"], data["labels"]
    epochs = hists[names[0]]["epoch"]
    n_seeds = data.get("n_seeds", 1)

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"Hoyer vs CV — SDT / {data.get('dataset', 'cifar10').upper()} "
                 f"({N_EPOCHS} ep × {n_seeds} seed(s))",
                 fontsize=13, fontweight="bold", y=0.995)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    def _panel(ax, key, title, ylabel):
        for i, nm in enumerate(names):
            y   = np.array(hists[nm][key])
            std = np.array(hists[nm].get(key + "_std", np.zeros_like(y)))
            ax.plot(epochs, y, color=COLORS[i], lw=2, marker="o", ms=3,
                    label=labels[nm], alpha=0.9)
            ax.fill_between(epochs, y - std, y + std, color=COLORS[i], alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _panel(fig.add_subplot(gs[0, 0]), "acc",      "Top-1 Accuracy ↑",     "%")
    _panel(fig.add_subplot(gs[0, 1]), "ice",      "ICE = Acc / MFR ↑",    "ICE")
    _panel(fig.add_subplot(gs[0, 2]), "mean_fr",  "Mean FR ↓",            "MFR")
    _panel(fig.add_subplot(gs[1, 0]), "hoyer",    "Hoyer sparsity ↑",     "H")
    _panel(fig.add_subplot(gs[1, 1]), "cv_fr",    "Population CV(FR) ↑",  "CV_pop")
    _panel(fig.add_subplot(gs[1, 2]), "cv_life",  "Lifetime CV ↑",        "CV_life")

    n = len(epochs) - 1
    lines = [f"Final (epoch {epochs[n]}, mean over {n_seeds} seed(s)):"]
    for key, label, hi in [
        ("acc",     "Acc %    ", True),
        ("ice",     "ICE      ", True),
        ("mean_fr", "Mean FR  ", False),
        ("hoyer",   "Hoyer    ", True),
        ("cv_fr",   "CV_pop   ", True),
        ("cv_life", "CV_life  ", True),
    ]:
        vb = hists[names[0]][key][n]
        row = f"  {label}  Base={vb:.4f}"
        for nm in names[1:]:
            vc = hists[nm][key][n]; d = vc - vb
            ok = (hi and d > 0) or (not hi and d < 0)
            row += f"  {labels[nm]}={vc:.4f}(Δ{d:+.4f}{'✓' if ok else ''})"
        lines.append(row)
    fig.text(0.01, -0.02, "\n".join(lines), fontsize=8, family="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    print(f"Figure → {FIGURE_FILE}")
    plt.close()


if __name__ == "__main__":
    if CLI.merge:
        merge_seeds()
    elif CLI.seed is not None:
        run_single_seed(CLI.seed)
    else:
        if os.path.exists(RESULTS_FILE) and not CLI.force:
            print(f"Loading cached {RESULTS_FILE}")
            with open(RESULTS_FILE) as f:
                plot(json.load(f))
        else:
            run()
