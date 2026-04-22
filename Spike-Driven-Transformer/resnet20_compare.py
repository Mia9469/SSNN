"""
resnet20_compare.py
===================
Continuous-activation control experiment for the CV-regularisation paper.

Trains ResNet-20 on CIFAR-10 from scratch under four conditions, matching the
SDT ablation in ``scratch_train_compare.py``:

  A. Baseline     : CE loss only
  B. WeightCV     : CE + weight_cv_loss              (weight heterogeneity)
  C. ActCV        : CE + population_cv_loss          (activation heterogeneity)
  D. Weight+ActCV : CE + both CV terms

Why this script exists
----------------------
The SDT study showed that maximising Pop-CV collapses Pop-CV by ~91% in a binary
spike network (the "dead-neuron trap").  The paper's mechanism-dichotomy claim
predicts that the *same* objective applied to a continuous-activation ResNet
should instead make Pop-CV rise monotonically, matching the SparseNet result.
This script is that smoking-gun control.

Output format mirrors scratch_compare_results.json so the downstream cross-scale
ICE-vs-CV plot can overlay both architectures without custom code.
"""

import copy
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
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import criterion_v2

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ── Training hyper-params ─────────────────────────────────────────────────────
N_EPOCHS        = 200
BATCH_TRAIN     = 128
BATCH_VAL       = 256
EVAL_EVERY      = 5
LR              = 0.1           # SGD-momentum CIFAR recipe
MIN_LR          = 1e-4
WARMUP_EP       = 5
WEIGHT_DECAY    = 5e-4
MOMENTUM        = 0.9

LAMBDA_WEIGHT_CV = 0.001
LAMBDA_ACT_CV    = 0.001
ACT_CV_START     = WARMUP_EP    # mirror the SDT warmup gate for comparability

N_SEEDS = 3                      # 3-seed average per condition

RESULTS_FILE = os.path.join(REPO, "resnet20_compare_results.json")
FIGURE_FILE  = os.path.join(REPO, "resnet20_compare.png")

# ── Ablation configs ──────────────────────────────────────────────────────────
CONFIGS = [
    {"name": "baseline",    "label": "Baseline",      "use_weight_cv": False, "use_act_cv": False},
    {"name": "weight_cv",   "label": "WeightCV",      "use_weight_cv": True,  "use_act_cv": False},
    {"name": "act_cv",      "label": "ActCV",         "use_weight_cv": False, "use_act_cv": True},
    {"name": "weight_act",  "label": "Weight+ActCV",  "use_weight_cv": True,  "use_act_cv": True},
]
COLORS = ["#1565C0", "#C62828", "#2E7D32", "#6A1B9A"]

print(f"Device  : {DEVICE}")
print(f"Epochs  : {N_EPOCHS}  |  eval every {EVAL_EVERY}  |  seeds={N_SEEDS}")
print(f"Configs : {[c['label'] for c in CONFIGS]}")


# ── ResNet-20 (CIFAR) ─────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        # The two post-ReLU activations are what the paper treats as the
        # "continuous spike rate" analogue.  We expose them as named Identity
        # modules so we can hook on them cleanly from outside.
        self.act1 = nn.ReLU(inplace=False)
        self.act2 = nn.ReLU(inplace=False)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.act2(out)
        return out


class ResNet20(nn.Module):
    """Standard He-2016 CIFAR ResNet-20: 3×3n+2 layers with n=3."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.act0  = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(16, 16, 3, stride=1)
        self.layer2 = self._make_layer(16, 32, 3, stride=2)
        self.layer3 = self._make_layer(32, 64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, in_ch, out_ch, n, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act0(self.bn1(self.conv1(x)))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def register_act_hooks(model: nn.Module):
    """
    Attach forward hooks on every ReLU module so their outputs are collected
    into a list per forward pass.  Returns ``(storage, handles)``; the caller
    clears ``storage`` between forward passes.

    Gradient flows through the stored tensors (we do *not* detach), so the
    activation-CV loss can be back-propagated the same way TET + CV is used in
    the SDT experiment.
    """
    storage = {"acts": [], "names": []}
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, nn.ReLU):
            def _hook(_mod, _inp, out, _name=name):
                storage["acts"].append(out)
                storage["names"].append(_name)
            handles.append(m.register_forward_hook(_hook))
    return storage, handles


# ── Data ──────────────────────────────────────────────────────────────────────
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
val_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

ds_train = torchvision.datasets.CIFAR10(os.path.join(REPO, "data"),
                                        train=True, download=True, transform=train_tf)
ds_val   = torchvision.datasets.CIFAR10(os.path.join(REPO, "data"),
                                        train=False, download=True, transform=val_tf)

_gpu = torch.cuda.is_available()
_nw  = 4 if _gpu else 2
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_TRAIN,
                                           shuffle=True,  num_workers=_nw, pin_memory=_gpu)
val_loader   = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_VAL,
                                           shuffle=False, num_workers=_nw, pin_memory=_gpu)


# ── LR schedule (cosine with warmup, matches SDT script for comparability) ───
def get_lr(epoch: int) -> float:
    if epoch < WARMUP_EP:
        return LR * (epoch + 1) / WARMUP_EP
    t = (epoch - WARMUP_EP) / max(1, N_EPOCHS - WARMUP_EP)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model: nn.Module, storage, handles) -> dict:
    model.eval()
    ce_fn = nn.CrossEntropyLoss().to(DEVICE)
    total_loss, total_correct, total_n = 0.0, 0, 0
    layer_pop_cv:  dict[str, list] = defaultdict(list)
    layer_mact:    dict[str, list] = defaultdict(list)

    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        storage["acts"].clear(); storage["names"].clear()
        logits = model(imgs)
        loss = ce_fn(logits, labels)
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)

        for name, a in zip(storage["names"], storage["acts"]):
            B = a.shape[0]
            flat = a.abs().view(B, -1)
            mean_n = flat.mean(dim=1) + 1e-8
            std_n  = flat.std(dim=1, unbiased=False) + 1e-8
            layer_pop_cv[name].append((std_n / mean_n).mean().item())
            layer_mact[name].append(flat.mean().item())

    weight_cv = criterion_v2.compute_weight_cv(model)

    ce_loss = total_loss / total_n
    acc     = total_correct / total_n * 100.0
    mact    = float(np.mean([np.mean(v) for v in layer_mact.values()]))
    cv_pop  = float(np.mean([np.mean(v) for v in layer_pop_cv.values()]))

    return {
        "cv_pop":   cv_pop,
        "weight_cv": weight_cv,
        "mean_act": mact,
        "ce_loss":  ce_loss,
        "acc":      acc,
        "ice":      acc / (mact + 1e-8),          # Acc / mean|a| — ResNet analogue of ICE_snn
        "layer_cv":  {k: float(np.mean(v)) for k, v in layer_pop_cv.items()},
        "layer_mact":{k: float(np.mean(v)) for k, v in layer_mact.items()},
    }


# ── Training loop (one config, one epoch) ────────────────────────────────────
def train_one_epoch(model, optimizer, config, epoch, storage) -> float:
    model.train()
    ce_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(DEVICE)
    total, count = 0.0, 0
    apply_act_cv = config["use_act_cv"] and epoch >= ACT_CV_START

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        storage["acts"].clear(); storage["names"].clear()
        logits = model(imgs)
        loss = ce_fn(logits, labels)

        if config["use_weight_cv"]:
            loss = loss + criterion_v2.weight_cv_loss(model, lambda_weight_cv=LAMBDA_WEIGHT_CV)

        if apply_act_cv and storage["acts"]:
            loss = loss + criterion_v2.population_cv_loss(storage["acts"],
                                                          lambda_cv=LAMBDA_ACT_CV)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item() * imgs.size(0)
        count += imgs.size(0)

    return total / count


# ── Single-seed run ───────────────────────────────────────────────────────────
METRIC_KEYS = ("cv_pop", "weight_cv", "mean_act", "ce_loss", "acc", "ice")


def run_one_seed(seed: int):
    """Run all 4 configs under a single seed and return hists + final metrics."""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    base_model = ResNet20(num_classes=10).to(DEVICE)
    models = {c["name"]: copy.deepcopy(base_model) for c in CONFIGS}
    optimizers = {
        c["name"]: torch.optim.SGD(
            models[c["name"]].parameters(),
            lr=LR, momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY, nesterov=True,
        )
        for c in CONFIGS
    }
    stores, handle_bags = {}, {}
    for c in CONFIGS:
        s, h = register_act_hooks(models[c["name"]])
        stores[c["name"]] = s; handle_bags[c["name"]] = h

    hists = {c["name"]: {k: [] for k in (*METRIC_KEYS, "epoch")} for c in CONFIGS}

    for epoch in range(N_EPOCHS):
        lr_now = get_lr(epoch)
        for opt in optimizers.values():
            for pg in opt.param_groups:
                pg["lr"] = lr_now

        t0 = time.time()
        tr_losses = {}
        for c in CONFIGS:
            tr_losses[c["name"]] = train_one_epoch(
                models[c["name"]], optimizers[c["name"]], c, epoch, stores[c["name"]])
        elapsed = time.time() - t0

        do_eval = (epoch + 1) % EVAL_EVERY == 0 or epoch == 0
        if do_eval:
            ep_num = epoch + 1
            for c in CONFIGS:
                m = evaluate(models[c["name"]], stores[c["name"]], handle_bags[c["name"]])
                hists[c["name"]]["epoch"].append(ep_num)
                for k in METRIC_KEYS:
                    hists[c["name"]][k].append(m[k])

            print(f"[seed {seed}] Ep {ep_num:>3}/{N_EPOCHS}  lr={lr_now:.2e} [{elapsed:.0f}s]")
            for c in CONFIGS:
                last = {k: hists[c["name"]][k][-1] for k in METRIC_KEYS}
                print(f"    {c['label']:14s} "
                      f"CV_pop={last['cv_pop']:.4f}  CV_w={last['weight_cv']:.4f}  "
                      f"Mact={last['mean_act']:.3f}  CE={last['ce_loss']:.4f}  "
                      f"Acc={last['acc']:.2f}%  ICE={last['ice']:.2f}")
        else:
            loss_str = "  ".join(f"{c['label']}={tr_losses[c['name']]:.4f}" for c in CONFIGS)
            print(f"[seed {seed}] Ep {epoch+1:>3}/{N_EPOCHS}  lr={lr_now:.2e} [{elapsed:.0f}s]  {loss_str}")

    final = {c["name"]: evaluate(models[c["name"]], stores[c["name"]], handle_bags[c["name"]])
             for c in CONFIGS}

    # release hooks
    for hs in handle_bags.values():
        for h in hs:
            h.remove()

    return hists, final


# ── Multi-seed aggregation ────────────────────────────────────────────────────
def run():
    all_hists = []
    all_finals = []
    t0 = time.time()
    for seed in range(N_SEEDS):
        print(f"\n=== Seed {seed + 1}/{N_SEEDS} ===")
        h, f = run_one_seed(seed)
        all_hists.append(h); all_finals.append(f)
    total_min = (time.time() - t0) / 60
    print(f"\nTotal runtime: {total_min:.1f} min across {N_SEEDS} seeds × 4 configs")

    # Average histories across seeds (assumes identical epoch grids)
    avg_hists = {c["name"]: {"epoch": all_hists[0][c["name"]]["epoch"]} for c in CONFIGS}
    for c in CONFIGS:
        for k in METRIC_KEYS:
            M = np.array([h[c["name"]][k] for h in all_hists])    # (seeds, epochs)
            avg_hists[c["name"]][k]     = M.mean(axis=0).tolist()
            avg_hists[c["name"]][k+"_std"] = M.std(axis=0).tolist()

    # Average final layer_cv / layer_mact across seeds
    data = {"hists": avg_hists,
            "configs": [c["name"] for c in CONFIGS],
            "labels":  {c["name"]: c["label"] for c in CONFIGS},
            "n_seeds": N_SEEDS}
    for c in CONFIGS:
        layers = sorted(all_finals[0][c["name"]]["layer_cv"].keys())
        data[f"final_{c['name']}_layer_cv"]  = {
            k: float(np.mean([f[c["name"]]["layer_cv"][k]  for f in all_finals])) for k in layers}
        data[f"final_{c['name']}_layer_mact"] = {
            k: float(np.mean([f[c["name"]]["layer_mact"][k] for f in all_finals])) for k in layers}
        data[f"final_{c['name']}_summary"] = {
            k: float(np.mean([f[c["name"]][k] for f in all_finals])) for k in METRIC_KEYS
        }
        data[f"final_{c['name']}_summary_std"] = {
            k: float(np.std([f[c["name"]][k] for f in all_finals])) for k in METRIC_KEYS
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved → {RESULTS_FILE}")
    plot(data)


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(data: dict):
    hists  = data["hists"]
    names  = data["configs"]
    labels = data["labels"]
    epochs = hists[names[0]]["epoch"]
    n_seeds = data.get("n_seeds", 1)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"ResNet-20 Ablation (continuous-activation control) — "
        f"{N_EPOCHS} epochs × {n_seeds} seeds, CIFAR-10",
        fontsize=13, fontweight="bold", y=1.00)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    def _panel(ax, key, title, ylabel):
        for i, nm in enumerate(names):
            y   = np.array(hists[nm][key])
            std = np.array(hists[nm].get(key + "_std", np.zeros_like(y)))
            ax.plot(epochs, y, color=COLORS[i], lw=2, marker="o", ms=3,
                    label=labels[nm], alpha=0.9)
            ax.fill_between(epochs, y - std, y + std, color=COLORS[i], alpha=0.15)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    _panel(fig.add_subplot(gs[0, 0]), "cv_pop",    "Population CV(|act|) ↑",      "CV")
    _panel(fig.add_subplot(gs[0, 1]), "weight_cv", "Weight CV ↑",                  "CV")
    _panel(fig.add_subplot(gs[0, 2]), "acc",       "Top-1 Accuracy ↑",            "%")
    _panel(fig.add_subplot(gs[1, 0]), "mean_act",  "Mean |activation| ↓",          "mean |a|")
    _panel(fig.add_subplot(gs[1, 1]), "ce_loss",   "CE Loss ↓",                    "CE")
    _panel(fig.add_subplot(gs[1, 2]), "ice",       "ICE  = Acc / mean|a|  ↑",      "ICE")

    # Summary box
    n = len(epochs) - 1
    lines = [f"Final (epoch {epochs[n]}, mean ± std over {n_seeds} seeds):"]
    for key, label, hi in [
        ("cv_pop",   "Pop CV(act) ", True),
        ("weight_cv","Weight CV   ", True),
        ("mean_act", "Mean |a|    ", False),
        ("ce_loss",  "CE Loss     ", False),
        ("acc",      "Acc %       ", True),
        ("ice",      "ICE         ", True),
    ]:
        vb = hists[names[0]][key][n]
        row = f"  {label}  Base={vb:.4f}"
        for nm in names[1:]:
            vc = hists[nm][key][n]
            d  = vc - vb
            ok = (hi and d > 0) or (not hi and d < 0)
            row += f"  {labels[nm]}={vc:.4f}(Δ{d:+.4f}{'✓' if ok else ''})"
        lines.append(row)
    lines.append(f"\nλ_weight={LAMBDA_WEIGHT_CV}  λ_act={LAMBDA_ACT_CV}  "
                 f"act_start=ep{ACT_CV_START}  SGD mom={MOMENTUM}  wd={WEIGHT_DECAY}")
    fig.text(0.01, -0.02, "\n".join(lines), fontsize=8, family="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    plt.savefig(FIGURE_FILE, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {FIGURE_FILE}")
    plt.close()

    # Console table
    print("\n" + "=" * 90)
    print(f"  {'Metric':<18}" + "".join(f"  {labels[nm]:>14}" for nm in names))
    print("-" * 90)
    for key, label, hi in [
        ("cv_pop",   "Pop CV(act)(↑)", True),
        ("weight_cv","Weight CV(↑)",   True),
        ("mean_act", "Mean|a|(↓)",     False),
        ("ce_loss",  "CE Loss(↓)",     False),
        ("acc",      "Top-1 Acc%(↑)",  True),
        ("ice",      "ICE(↑)",         True),
    ]:
        row = f"  {label:<18}"
        vb = hists[names[0]][key][n]
        for nm in names:
            vc = hists[nm][key][n]
            d  = vc - vb
            ok = (hi and d > 0) or (not hi and d < 0)
            mark = "✓" if (nm != names[0] and ok) else " "
            row += f"  {vc:>12.4f}{mark}"
        print(row)
    print("=" * 90)


def run_with_cache():
    if os.path.exists(RESULTS_FILE) and "--force" not in sys.argv:
        print(f"Loading cached results from {RESULTS_FILE}")
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        plot(data)
    else:
        run()


if __name__ == "__main__":
    run_with_cache()
