# GitHub 上图像处理 SNN 模型 Top 5 推荐

> 筛选标准：引用较高、时间最新、能在本地轻松复现（无特殊硬件要求、无过大数据集依赖）、使用公开数据集或提供数据集
>
> 调研时间：2026年3月

---

## 1. Spikformer — `ZK-Zhou/spikformer`

| 项目 | 详情 |
|---|---|
| **GitHub** | https://github.com/ZK-Zhou/spikformer |
| **Stars** | ~394 |
| **论文** | *Spikformer: When Spiking Neural Network Meets Transformer* (ICLR 2023) |
| **数据集** | CIFAR-10, CIFAR-10-DVS, ImageNet-1K (CIFAR 自动下载) |
| **复现难度** | ⭐ 低。CIFAR-10 训练只需单 GPU，代码简洁，依赖 `spikingjelly==0.0.0.0.12` + PyTorch |
| **亮点** | 首次将 Transformer 架构引入 SNN，开创性工作，被后续大量引用；CIFAR 实验可在单卡完成 |

---

## 2. Spike-Driven Transformer — `BICLab/Spike-Driven-Transformer`

| 项目 | 详情 |
|---|---|
| **GitHub** | https://github.com/BICLab/Spike-Driven-Transformer |
| **Stars** | ~303 |
| **论文** | *Spike-driven Transformer* (NeurIPS 2023) |
| **数据集** | CIFAR-10, CIFAR-100, CIFAR-10-DVS, DVS128 Gesture, ImageNet-1K |
| **复现难度** | ⭐ 低-中。CIFAR/DVS 数据集自动下载，单 GPU 可训练；提供预训练权重可直接测试 |
| **亮点** | 纯脉冲驱动 Transformer，无乘法运算（只有 mask + addition），能耗极低，ImageNet 达 77.1% top-1 |

---

## 3. QKFormer — `zhouchenlin2096/QKFormer`

| 项目 | 详情 |
|---|---|
| **GitHub** | https://github.com/zhouchenlin2096/QKFormer |
| **Stars** | ~140 |
| **论文** | *QKFormer: Hierarchical Spiking Transformer using Q-K Attention* (NeurIPS 2024 Spotlight) |
| **数据集** | CIFAR-10, CIFAR-100, CIFAR-10-DVS, DVS128 Gesture, ImageNet-1K |
| **复现难度** | ⭐ 低-中。CIFAR 实验可单卡训练；提供 Google Drive 预训练模型下载；代码结构清晰 |
| **亮点** | SNN 首次在 ImageNet 上突破 85% top-1 准确率（85.65%），NeurIPS 2024 最新成果 |

---

## 4. Spikingformer — `TheBrainLab/Spikingformer`

| 项目 | 详情 |
|---|---|
| **GitHub** | https://github.com/TheBrainLab/Spikingformer |
| **Stars** | ~127 |
| **论文** | *Spikingformer: A Key Foundation Model for Spiking Neural Networks* (AAAI 2026) |
| **数据集** | CIFAR-10, CIFAR-100, CIFAR-10-DVS, DVS128 Gesture, ImageNet-1K |
| **复现难度** | ⭐ 低。与 Spikformer 代码结构类似，CIFAR 实验单卡即可；提供训练日志和预训练模型 |
| **亮点** | 纯事件驱动 SNN Transformer，在 Spikformer 基础上提升 1.04%，能耗降低 57.34%；时间极新(AAAI 2026) |

---

## 5. SpikingJelly — `fangwei123456/spikingjelly`

| 项目 | 详情 |
|---|---|
| **GitHub** | https://github.com/fangwei123456/spikingjelly |
| **Stars** | ~1900 |
| **论文** | 发表于 *Science Advances* (2023)，ICLR 2026 新论文 |
| **数据集** | 内置 MNIST, Fashion-MNIST, CIFAR-10/100, CIFAR-10-DVS, N-MNIST, DVS128 Gesture 等十多种 |
| **复现难度** | ⭐⭐ 极低。一行命令即可训练 MNIST；支持 CPU/GPU |
| **亮点** | 最主流 SNN 框架（非单一模型而是完整平台），自带多种图像分类示例（LIF FC MNIST, Conv FMNIST, ANN2SNN 等），是上述所有模型的底层依赖；教程极其完善；92%+ MNIST / 98.44% ANN-SNN转换 |

**快速启动示例：**
```bash
python -m spikingjelly.activation_based.examples.lif_fc_mnist -tau 2.0 -T 100 -device cuda:0 -b 64 -epochs 100 -data-dir <PATH to MNIST> -amp -opt adam -lr 1e-3 -j 8
```

---

## 总结对比

| 排名 | 项目 | 会议/年份 | Stars | CIFAR-10 可单卡 | 需要 ImageNet? | 预训练模型 |
|:---:|---|---|---:|:---:|:---:|:---:|
| 1 | Spikformer | ICLR 2023 | 394 | ✅ | 可选 | ✅ |
| 2 | Spike-Driven Transformer | NeurIPS 2023 | 303 | ✅ | 可选 | ✅ |
| 3 | QKFormer | NeurIPS 2024 | 140 | ✅ | 可选 | ✅ |
| 4 | Spikingformer | AAAI 2026 | 127 | ✅ | 可选 | ✅ |
| 5 | SpikingJelly | Science Adv. 2023 | 1900 | ✅ | 不需要 | 内置示例 |

---

## 快速上手建议

- **最快看到结果**：从 **SpikingJelly** 的内置 MNIST/FMNIST 示例开始（无需任何额外数据下载）
- **复现顶会 SOTA 结果**：从 **Spikformer** 或 **QKFormer** 的 CIFAR-10 实验入手（数据集自动下载，单 GPU 即可训练）
