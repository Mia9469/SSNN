"""
spikformer_model.py
===================
Minimal, self-contained implementation of Spikformer (Zhou et al., ICLR 2023)
for the CV-regularisation cross-architecture check (P2.4).

Why this file exists separately from ``model/spikeformer.py``:
    The SDT model shipped in this repo implements Yao-2023 spike-driven
    attention (element-wise K⊙V followed by a sum), *not* the Zhou-2023
    Spikformer attention (matmul Q·K^T → LIF → @V).  The mechanism-dichotomy
    claim in the paper is supposed to hold across spiking architectures, so we
    need an honest Spikformer implementation as a second data point.

Architecture highlights:
  * MS_SPS patch embedding (reused from the existing repo, produces T,B,C,H,W).
  * After the patch embedding we flatten the spatial grid into tokens
    (T, B, N, C) with N = H·W, matching the Spikformer reference code.
  * SpikingSelfAttention: Q / K / V via Linear+BN+LIF, scaled dot-product
    attention, gated by a LIF neuron on the attention tensor, then Linear+BN+LIF
    output projection.
  * SpikingMLP: two Linear+BN+LIF blocks with expansion ratio 4.
  * Classification head: mean-pool over tokens → LIF → Linear → (optional
    time-dimension average depending on the TET flag).

The forward pass returns ``(logits, hook)`` with the same hook contract used by
the SDT script, so ``criterion_v2.firing_rate_cv_loss`` / ``weight_cv_loss`` can
be applied without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

from module import MS_SPS


# ── helpers ────────────────────────────────────────────────────────────────────
def _lif(spike_mode: str = "lif", v_threshold: float = 1.0):
    """Factory for the backend-consistent LIF neuron used throughout the model."""
    if spike_mode == "lif":
        return MultiStepLIFNode(tau=2.0, v_threshold=v_threshold,
                                detach_reset=True, backend="torch")
    if spike_mode == "plif":
        return MultiStepParametricLIFNode(init_tau=2.0, v_threshold=v_threshold,
                                          detach_reset=True, backend="torch")
    raise ValueError(f"unknown spike_mode={spike_mode}")


# ── Spiking Self-Attention (Zhou 2023) ────────────────────────────────────────
class SpikingSelfAttention(nn.Module):
    """
    Q, K, V via Linear+BN+LIF.  Attention = LIF( (Q · K^T) · scale ).
    Output = attention · V, then Linear+BN+LIF projection.

    Unlike MS_SSA_Conv (which uses element-wise K⊙V summed over tokens),
    this is the canonical Spikformer formulation: a full N×N attention matrix
    gated by a LIF neuron.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, scale=0.125,
                 spike_mode="lif", layer_idx=0):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale
        self.layer_idx = layer_idx

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_bn     = nn.BatchNorm1d(dim)
        self.q_lif    = _lif(spike_mode)

        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_bn     = nn.BatchNorm1d(dim)
        self.k_lif    = _lif(spike_mode)

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_bn     = nn.BatchNorm1d(dim)
        self.v_lif    = _lif(spike_mode)

        # Spikformer's attention-gate uses a lower threshold (0.5)
        self.attn_lif = _lif(spike_mode, v_threshold=0.5)

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn     = nn.BatchNorm1d(dim)
        self.proj_lif    = _lif(spike_mode)

    def _spk(self, x_flat, linear, bn, lif, T, B, N):
        """Linear → BN(over channel dim) → reshape back to (T,B,N,C) → LIF."""
        y = linear(x_flat)                                  # (T·B, N, C)
        y = bn(y.transpose(-1, -2)).transpose(-1, -2)       # BN over channel
        y = y.reshape(T, B, N, self.dim).contiguous()
        return lif(y)

    def forward(self, x, hook=None):
        T, B, N, C = x.shape
        identity = x

        x_flat = x.flatten(0, 1)        # (T·B, N, C)

        q = self._spk(x_flat, self.q_linear, self.q_bn, self.q_lif, T, B, N)
        k = self._spk(x_flat, self.k_linear, self.k_bn, self.k_lif, T, B, N)
        v = self._spk(x_flat, self.v_linear, self.v_bn, self.v_lif, T, B, N)
        if hook is not None:
            hook[f"SSA{self.layer_idx}_q_lif"] = q
            hook[f"SSA{self.layer_idx}_k_lif"] = k
            hook[f"SSA{self.layer_idx}_v_lif"] = v

        # (T, B, h, N, d)
        qh = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kh = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        vh = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        # Spikformer attention: Q · K^T * scale, then LIF-gated
        attn = (qh @ kh.transpose(-2, -1)) * self.scale      # (T, B, h, N, N)
        attn = self.attn_lif(attn)                            # LIF expects (T, ...)
        if hook is not None:
            hook[f"SSA{self.layer_idx}_attn_lif"] = attn

        out = (attn @ vh).transpose(2, 3).reshape(T, B, N, C).contiguous()

        out_flat = out.flatten(0, 1)
        out = out_flat @ self.proj_linear.weight.t() + (
            self.proj_linear.bias if self.proj_linear.bias is not None else 0.0)
        out = self.proj_bn(out.transpose(-1, -2)).transpose(-1, -2)
        out = out.reshape(T, B, N, C).contiguous()
        out = self.proj_lif(out)
        if hook is not None:
            hook[f"SSA{self.layer_idx}_proj_lif"] = out

        return out + identity, hook


class SpikingMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, spike_mode="lif", layer_idx=0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc1_bn = nn.BatchNorm1d(hidden)
        self.fc1_lif = _lif(spike_mode)
        self.fc2 = nn.Linear(hidden, dim)
        self.fc2_bn = nn.BatchNorm1d(dim)
        self.fc2_lif = _lif(spike_mode)
        self.layer_idx = layer_idx

    def forward(self, x, hook=None):
        T, B, N, C = x.shape
        identity = x
        flat = x.flatten(0, 1)
        y = self.fc1(flat)
        y = self.fc1_bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, -1)
        y = self.fc1_lif(y)
        if hook is not None:
            hook[f"MLP{self.layer_idx}_fc1_lif"] = y

        flat2 = y.flatten(0, 1)
        y = self.fc2(flat2)
        y = self.fc2_bn(y.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C)
        y = self.fc2_lif(y)
        if hook is not None:
            hook[f"MLP{self.layer_idx}_fc2_lif"] = y
        return y + identity, hook


class SpikformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, spike_mode="lif", layer=0):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads,
                                         spike_mode=spike_mode, layer_idx=layer)
        self.mlp  = SpikingMLP(dim, mlp_ratio=mlp_ratio,
                               spike_mode=spike_mode, layer_idx=layer)

    def forward(self, x, hook=None):
        x, hook = self.attn(x, hook=hook)
        x, hook = self.mlp(x, hook=hook)
        return x, hook


# ── Full Spikformer model ──────────────────────────────────────────────────────
class Spikformer(nn.Module):
    def __init__(self, img_size_h=32, img_size_w=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dims=256, num_heads=8, mlp_ratios=4,
                 depths=2, T=4, pooling_stat="0011", spike_mode="lif",
                 TET=True, **unused):
        super().__init__()
        self.T = T
        self.TET = TET
        self.num_classes = num_classes

        # Reuse the existing spiking patch-split from the repo.  MS_SPS returns
        # (T, B, C, H, W); we flatten into (T, B, N, C) before the SSA blocks.
        self.patch_embed = MS_SPS(
            img_size_h=img_size_h, img_size_w=img_size_w,
            patch_size=patch_size, in_channels=in_channels,
            embed_dims=embed_dims, pooling_stat=pooling_stat,
            spike_mode=spike_mode,
        )

        self.blocks = nn.ModuleList([
            SpikformerBlock(embed_dims, num_heads=num_heads,
                            mlp_ratio=mlp_ratios, spike_mode=spike_mode, layer=i)
            for i in range(depths)
        ])

        self.head_lif = _lif(spike_mode)
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, hook=None):
        # Expand (B, C, H, W) to (T, B, C, H, W)
        if x.dim() < 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            x = x.transpose(0, 1).contiguous()

        x, _, hook = self.patch_embed(x, hook=hook)          # (T, B, C, H', W')
        T, B, C, H, W = x.shape
        x = x.flatten(3).transpose(-1, -2).contiguous()      # (T, B, N, C)

        for blk in self.blocks:
            x, hook = blk(x, hook=hook)

        x = x.mean(dim=2)                                     # pool over tokens → (T, B, C)
        x = self.head_lif(x)
        if hook is not None:
            hook["head_lif"] = x
        logits = self.head(x)                                 # (T, B, n_classes)

        if not self.TET:
            logits = logits.mean(0)
        return logits, hook


@register_model
def spikformer(**kwargs):
    model = Spikformer(**kwargs)
    model.default_cfg = _cfg()
    return model
