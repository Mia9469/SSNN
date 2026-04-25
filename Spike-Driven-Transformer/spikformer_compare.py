"""
spikformer_compare.py
=====================
P2.4: Cross-architecture generalisation check for the CV-regularisation paper.

Trains Spikformer (Zhou 2023, SSA attention) from scratch under two conditions:

  A. Baseline       : TET loss only
  B. Weight+PopCV   : TET + weight_cv_loss + firing_rate_cv_loss (after warmup)

Why only two conditions?
    The 4-way SDT ablation already established the pairwise marginal effects of
    WeightCV vs PopFR-CV.  The Spikformer experiment is a generalisation check:
    does the best combined objective (Weight+PopCV) still yield the +1%-ish
    accuracy / ICE uplift on a *different* spiking attention mechanism?  Two
    conditions × three seeds × 200 epochs ≈ 12 GPU-hours on a single 3090, far
    cheaper than the full 4-way rerun.

Outputs
  spikformer_compare_results.json  — full history for both configs
  spikformer_compare.png           — multi-panel figure matching the SDT one
"""

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

import criterion_v2
import spikformer_model  # noqa: registers 'spikformer' in the timm registry
from spikingjelly.clock_driven import functional

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Hyper-params (matched to the SDT run for fair comparison) ────────────────
N_EPOCHS        = 200         # lighter than SDT's 300 — Spikformer converges faster
BATCH_TRAIN     = 64
BATCH_VAL       = 128
EVAL_EVERY      = 10
LR              = 3e-4
MIN_LR          = 1e-5
WARMUP_EP       = 20
WEIGHT_DECAY    = 0.06

LAMBDA_WEIGHT_CV = 0.001
# Spikformer's PopFR-CV knee is ~5× tighter than SDT's (see
# lambda_sweep_spikformer_results.json): at λ=5e-4 Spikformer partially
# collapses (Acc~70%), while λ=1e-4 sits in the safe regime and gives
# the largest clean ICE uplift (+56% over baseline at Acc=81.5%).
LAMBDA_POP_CV    = 0.0001
POP_CV_START     = WARMUP_EP

N_SEEDS = 1                   # set to 3 if GPU-hours allow; 1 is enough for sign-check

# ── Dataset CLI (parsed early so output filenames can use the tag) ───────────
_bootstrap_ap = argparse.ArgumentParser(add_help=False)
_bootstrap_ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
_bootstrap_ap.add_argument("--tag", default="")
_bootstrap_ap.add_argument("--lambda-pop", type=float, default=None,
                           dest="lambda_pop",
                           help="override LAMBDA_POP_CV at runtime (default: 1e-4)")
_bootstrap_ap.add_argument("--lambda-weight", type=float, default=None,
                           dest="lambda_weight",
                           help="override LAMBDA_WEIGHT_CV at runtime (default: 1e-3)")
_BOOT, _ = _bootstrap_ap.parse_known_args()
if _BOOT.lambda_pop is not None:
    LAMBDA_POP_CV = _BOOT.lambda_pop
if _BOOT.lambda_weight is not None:
    LAMBDA_WEIGHT_CV = _BOOT.lambda_weight

DATASET     = _BOOT.dataset
NUM_CLASSES = 100 if DATASET == "cifar100" else 10
_TAG        = f"_{_BOOT.tag}" if _BOOT.tag else ""

RESULTS_FILE = os.path.join(REPO, f"spikformer_compare_results{_TAG}.json")
FIGURE_FILE  = os.path.join(REPO, f"spikformer_compare{_TAG}.png")
print(f"Dataset : {DATASET.upper()} ({NUM_CLASSES} classes)  |  out → {os.path.basename(RESULTS_FILE)}")

CONFIGS = [
    {"name": "baseline",  "label": "Baseline",      "use_weight_cv": False, "use_pop_cv": False},
    {"name": "weight_pop","label": "Weight+PopCV",  "use_weight_cv": True,  "use_pop_cv": True},
]
COLORS = ["#1565C0", "#6A1B9A"]

print(f"Device  : {DEVICE}")
print(f"Model   : Spikformer (SSA attention, depth 2, dim 256, heads 8, T=4)")
print(f"Epochs  : {N_EPOCHS} × {N_SEEDS} seeds  |  LR={LR}")
print(f"λ_W={LAMBDA_WEIGHT_CV}  λ_pop={LAMBDA_POP_CV}  pop_cv_start={POP_CV_START}")


# ── Model factory ─────────────────────────────────────────────────────────────
def make_model() -> nn.Module:
    return create_model(
        "spikformer",
        T=4, num_classes=NUM_CLASSES,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=256, num_heads=8,
        mlp_ratios=4, depths=2,
        pooling_stat="0011", spike_mode="lif",
        TET=True, in_channels=3,
    ).to(DEVICE)


# ── Data (CIFAR-10 only; matching the SDT baseline recipe) ───────────────────
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
ds_train = _DSCLS(os.path.join(REPO, "data"),
                  train=True,  download=True, transform=train_tf)
ds_val   = _DSCLS(os.path.join(REPO, "data"),
                  train=False, download=True, transform=val_tf)

_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_TRAIN,
                                           shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=BATCH_VAL,
                                           shuffle=False, num_workers=_nw, pin_memory=_gpu)

mixup_fn = Mixup(
    mixup_alpha=0.5, cutmix_alpha=0.0, prob=1.0,
    switch_prob=0.5, mode="batch",
    label_smoothing=0.1, num_classes=NUM_CLASSES,
)


def get_lr(epoch):
    if epoch < WARMUP_EP:
        return LR * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, N_EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


def tet_loss(outputs, labels, ce_fn):
    T_ = outputs.size(0)
    return sum(ce_fn(outputs[t], labels) for t in range(T_)) / T_


# ── Evaluation (same metric schema as the SDT runner) ────────────────────────
def evaluate(model):
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_fr:   dict[str, list] = defaultdict(list)
    layer_pop:  dict[str, list] = defaultdict(list)
    layer_life: dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            hook = {}
            outputs, hook = model(imgs, hook=hook)
            functional.reset_net(model)

            logits_mean = outputs.mean(0)
            loss = ce_fn(logits_mean, labels)
            total_loss    += loss.item() * imgs.size(0)
            total_correct += (logits_mean.argmax(1) == labels).sum().item()
            total_n       += imgs.size(0)

            for k, s in hook.items():
                T_, B = s.shape[0], s.shape[1]
                flat = s.float().view(T_, B, -1)
                fr = flat.mean(dim=0)                   # (B, N)
                layer_fr[k].append(fr.mean().item())
                mean_n = fr.mean(dim=1, keepdim=True) + 1e-8
                std_n  = fr.std(dim=1, keepdim=True, unbiased=False) + 1e-8
                layer_pop[k].append((std_n / mean_n).mean().item())
                fr_flat = fr.view(B, -1)
                mean_b = fr_flat.mean(dim=0) + 1e-8
                std_b  = fr_flat.std(dim=0, unbiased=False) + 1e-8
                layer_life[k].append((std_b / mean_b).mean().item())

        weight_cv = criterion_v2.compute_weight_cv(model)

    ce_loss = total_loss / total_n
    acc     = total_correct / total_n * 100.0
    mean_fr = float(np.mean([np.mean(v) for v in layer_fr.values()]))
    cv_fr   = float(np.mean([np.mean(v) for v in layer_pop.values()]))
    cv_life = float(np.mean([np.mean(v) for v in layer_life.values()])) if layer_life else 0.0

    return {
        "cv_fr": cv_fr, "cv_life": cv_life, "weight_cv": weight_cv,
        "mean_fr": mean_fr, "ce_loss": ce_loss, "acc": acc,
        "ice": acc / (mean_fr + 1e-8),
        "layer_cv":   {k: float(np.mean(v)) for k, v in layer_pop.items()},
        "layer_mfr":  {k: float(np.mean(v)) for k, v in layer_fr.items()},
        "layer_lcv":  {k: float(np.mean(v)) for k, v in layer_life.items()},
    }


# ── Training loop ────────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, config, epoch):
    model.train()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    total, count = 0.0, 0
    apply_pop_cv = config["use_pop_cv"] and epoch >= POP_CV_START

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        imgs, labels = mixup_fn(imgs, labels)
        optimizer.zero_grad()

        hook = {} if apply_pop_cv else None
        outputs, hook = model(imgs, hook=hook)

        loss = tet_loss(outputs, labels, ce_fn)
        if config["use_weight_cv"]:
            loss = loss + criterion_v2.weight_cv_loss(model, lambda_weight_cv=LAMBDA_WEIGHT_CV)
        if apply_pop_cv and hook:
            loss = loss + criterion_v2.firing_rate_cv_loss(list(hook.values()),
                                                           lambda_cv=LAMBDA_POP_CV)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        functional.reset_net(model)

        total += loss.item() * imgs.size(0)
        count += imgs.size(0)
    return total / count


METRIC_KEYS = ("cv_fr", "cv_life", "weight_cv", "mean_fr", "ce_loss", "acc", "ice")


def run_one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base = make_model()
    models = {c["name"]: copy.deepcopy(base) for c in CONFIGS}
    optimizers = {c["name"]: torch.optim.AdamW(models[c["name"]].parameters(),
                                                lr=LR, weight_decay=WEIGHT_DECAY)
                  for c in CONFIGS}
    hists = {c["name"]: {k: [] for k in (*METRIC_KEYS, "epoch")} for c in CONFIGS}

    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for opt in optimizers.values():
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        t0 = time.time()
        tr = {c["name"]: train_one_epoch(models[c["name"]], optimizers[c["name"]], c, epoch)
              for c in CONFIGS}
        elapsed = time.time() - t0

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            ep = epoch + 1
            for c in CONFIGS:
                m = evaluate(models[c["name"]])
                hists[c["name"]]["epoch"].append(ep)
                for k in METRIC_KEYS:
                    hists[c["name"]][k].append(m[k])
            print(f"[seed {seed}] Ep {ep:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{elapsed:.0f}s]")
            for c in CONFIGS:
                last = {k: hists[c["name"]][k][-1] for k in METRIC_KEYS}
                print(f"    {c['label']:14s} CV_pop={last['cv_fr']:.4f}  CV_L={last['cv_life']:.4f}  "
                      f"CV_w={last['weight_cv']:.4f}  MFR={last['mean_fr']:.4f}  "
                      f"CE={last['ce_loss']:.4f}  Acc={last['acc']:.2f}%  ICE={last['ice']:.1f}")
        else:
            ls = "  ".join(f"{c['label']}={tr[c['name']]:.4f}" for c in CONFIGS)
            print(f"[seed {seed}] Ep {epoch+1:>3}/{N_EPOCHS}  lr={lr_now:.2e}  [{elapsed:.0f}s]  {ls}")

    final = {c["name"]: evaluate(models[c["name"]]) for c in CONFIGS}
    return hists, final


def run():
    all_hists, all_final = [], []
    t0 = time.time()
    for s in range(N_SEEDS):
        print(f"\n=== seed {s + 1}/{N_SEEDS} ===")
        h, f = run_one_seed(s)
        all_hists.append(h); all_final.append(f)
    print(f"\nTotal: {(time.time() - t0)/60:.1f} min across {N_SEEDS} seed(s)")

    avg_hists = {c["name"]: {"epoch": all_hists[0][c["name"]]["epoch"]} for c in CONFIGS}
    for c in CONFIGS:
        for k in METRIC_KEYS:
            M = np.array([h[c["name"]][k] for h in all_hists])
            avg_hists[c["name"]][k] = M.mean(axis=0).tolist()
            avg_hists[c["name"]][k+"_std"] = M.std(axis=0).tolist()

    data = {"hists": avg_hists,
            "configs": [c["name"] for c in CONFIGS],
            "labels":  {c["name"]: c["label"] for c in CONFIGS},
            "n_seeds": N_SEEDS}
    for c in CONFIGS:
        for field in ("layer_cv", "layer_mfr", "layer_lcv"):
            keys = sorted(all_final[0][c["name"]][field].keys())
            data[f"final_{c['name']}_{field}"] = {
                k: float(np.mean([f[c["name"]][field][k] for f in all_final])) for k in keys}
        data[f"final_{c['name']}_summary"] = {
            k: float(np.mean([f[c["name"]][k] for f in all_final])) for k in METRIC_KEYS}

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results → {RESULTS_FILE}")
    plot(data)


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot(data):
    hists  = data["hists"]
    names  = data["configs"]
    labels = data["labels"]
    epochs = hists[names[0]]["epoch"]
    n_seeds = data.get("n_seeds", 1)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"Spikformer × (Baseline / Weight+PopCV) — CIFAR-10, {N_EPOCHS} ep × {n_seeds} seed(s)",
        fontsize=13, fontweight="bold", y=1.00)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    def _panel(ax, key, title, ylabel):
        for i, nm in enumerate(names):
            y   = np.array(hists[nm][key])
            std = np.array(hists[nm].get(key+"_std", np.zeros_like(y)))
            ax.plot(epochs, y, color=COLORS[i], lw=2, marker="o", ms=3,
                    label=labels[nm], alpha=0.9)
            ax.fill_between(epochs, y - std, y + std, color=COLORS[i], alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _panel(fig.add_subplot(gs[0, 0]), "cv_fr",    "Population CV(FR) ↑",  "CV")
    _panel(fig.add_subplot(gs[0, 1]), "cv_life",  "Lifetime CV ↑",         "CV")
    _panel(fig.add_subplot(gs[0, 2]), "weight_cv","Weight CV ↑",           "CV")
    _panel(fig.add_subplot(gs[1, 0]), "mean_fr",  "Mean Firing Rate ↓",    "MFR")
    _panel(fig.add_subplot(gs[1, 1]), "ce_loss",  "CE Loss ↓",             "CE")
    _panel(fig.add_subplot(gs[1, 2]), "acc",      "Top-1 Accuracy ↑",      "%")

    n = len(epochs) - 1
    lines = [f"Final (epoch {epochs[n]}, mean over {n_seeds} seed(s)):"]
    for key, label, hi in [
        ("cv_fr",    "Pop CV(FR) ", True),
        ("cv_life",  "Lifetime CV", True),
        ("weight_cv","Weight CV  ", True),
        ("mean_fr",  "Mean FR    ", False),
        ("ce_loss",  "CE Loss    ", False),
        ("acc",      "Acc %      ", True),
        ("ice",      "ICE        ", True),
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


def run_with_cache():
    if os.path.exists(RESULTS_FILE) and "--force" not in sys.argv:
        print(f"Loading cached {RESULTS_FILE}")
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        plot(data)
    else:
        run()


# ── Parallel-seed mode ───────────────────────────────────────────────────────
# Usage for 3 parallel seeds on 3 GPUs:
#   CUDA_VISIBLE_DEVICES=0 python spikformer_compare.py --seed 0 &
#   CUDA_VISIBLE_DEVICES=1 python spikformer_compare.py --seed 1 &
#   CUDA_VISIBLE_DEVICES=2 python spikformer_compare.py --seed 2 &
#   wait
#   python spikformer_compare.py --merge       # aggregate into main JSON + plot

def _seed_file(seed):
    return os.path.join(REPO, f"spikformer_seed{seed}{_TAG}.json")


def run_single_seed(seed):
    t0 = time.time()
    hists, final = run_one_seed(seed)
    dt = (time.time() - t0) / 60
    out = {
        "seed": seed,
        "hists": hists,
        "final": final,
        "configs": [c["name"] for c in CONFIGS],
        "labels":  {c["name"]: c["label"] for c in CONFIGS},
        "n_epochs": N_EPOCHS,
        "runtime_min": dt,
    }
    path = _seed_file(seed)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[seed {seed}] done in {dt:.1f} min → {path}")


def merge_seeds():
    paths = sorted(glob.glob(os.path.join(REPO, f"spikformer_seed*{_TAG}.json")))
    # Exclude cross-tag files (e.g., when TAG="" don't sweep in _c100 files)
    if not _TAG:
        paths = [p for p in paths if "_c100" not in os.path.basename(p)
                 and "_cifar100" not in os.path.basename(p)]
    if not paths:
        print("No spikformer_seed*.json files to merge."); return
    print(f"Merging {len(paths)} per-seed files:")
    for p in paths:
        print(f"  {os.path.basename(p)}")

    per_seed = [json.load(open(p)) for p in paths]
    all_hists = [s["hists"] for s in per_seed]
    all_final = [s["final"] for s in per_seed]

    avg_hists = {c["name"]: {"epoch": all_hists[0][c["name"]]["epoch"]} for c in CONFIGS}
    for c in CONFIGS:
        for k in METRIC_KEYS:
            M = np.array([h[c["name"]][k] for h in all_hists])
            avg_hists[c["name"]][k] = M.mean(axis=0).tolist()
            avg_hists[c["name"]][k+"_std"] = M.std(axis=0).tolist()

    data = {"hists": avg_hists,
            "configs": [c["name"] for c in CONFIGS],
            "labels":  {c["name"]: c["label"] for c in CONFIGS},
            "n_seeds": len(per_seed)}
    for c in CONFIGS:
        for field in ("layer_cv", "layer_mfr", "layer_lcv"):
            keys = sorted(all_final[0][c["name"]][field].keys())
            data[f"final_{c['name']}_{field}"] = {
                k: float(np.mean([f[c["name"]][field][k] for f in all_final])) for k in keys}
        data[f"final_{c['name']}_summary"] = {
            k: float(np.mean([f[c["name"]][k] for f in all_final])) for k in METRIC_KEYS}

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nMerged → {RESULTS_FILE} (n_seeds={len(per_seed)})")
    plot(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None,
                    help="Run exactly one seed and write spikformer_seed<N>.json")
    ap.add_argument("--merge", action="store_true",
                    help="Merge all per-seed JSONs into the main results file + plot")
    ap.add_argument("--force", action="store_true",
                    help="Bypass cached spikformer_compare_results.json in default mode")
    args, _ = ap.parse_known_args()

    if args.merge:
        merge_seeds()
    elif args.seed is not None:
        run_single_seed(args.seed)
    else:
        run_with_cache()
