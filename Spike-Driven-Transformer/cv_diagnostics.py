#!/usr/bin/env python3

"""
CV Loss 诊断和监控工具

帮助诊断：
1. 当前模型的firing rate CV值是否在变化
2. 论证权重CV是否会改进性能
3. 监控不同CV正则化策略的效果
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class CVDiagnostics:
    """诊断和监控CV值的工具类"""
    
    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.cv_history = {
            'firing_rate_cv': [],
            'weight_cv': [],
            'activation_cv': [],
            'epoch': []
        }
    
    def compute_firing_rate_cv(self, spike_outputs):
        """
        计算firing rate的CV值。
        
        Args:
            spike_outputs: (T, B, ...) 二进制spike张量
        
        Returns:
            cv_firing: float - 平均firing rate CV
            cv_per_neuron: dict - 各神经元的CV统计
        """
        T = spike_outputs.shape[0]
        
        # 计算每个神经元的放电率
        firing_rate = spike_outputs.sum(dim=0) / T
        
        B = firing_rate.shape[0]
        firing_rate_flat = firing_rate.view(B, -1)
        
        # 计算population级别的CV（所有神经元）
        mean_fr = firing_rate_flat.mean()
        std_fr = firing_rate_flat.std() + 1e-8
        cv_global = std_fr / (mean_fr + 1e-8)
        
        # 计算每个样本的CV
        cv_per_batch = []
        for b in range(B):
            fr = firing_rate_flat[b]
            mean_b = fr.mean()
            std_b = fr.std() + 1e-8
            cv_b = std_b / (mean_b + 1e-8)
            cv_per_batch.append(cv_b.item())
        
        stats = {
            'cv_global': float(cv_global),
            'cv_mean_batch': np.mean(cv_per_batch),
            'cv_std_batch': np.std(cv_per_batch),
            'sparsity': float((firing_rate_flat == 0).sum() / firing_rate_flat.numel()),
            'mean_firing_rate': float(firing_rate_flat.mean()),
        }
        
        return float(cv_global), stats
    
    def compute_weight_cv(self, layer_types=(nn.Conv2d, nn.Linear)):
        """
        计算权重的CV值。
        
        论证：如果权重CV高（权重多样性强），模型的表征能力更强。
        
        Returns:
            cv_weight: float - 平均权重CV
            cv_per_layer: dict - 各层权重CV
        """
        layer_cvs = {}
        all_cvs = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types) and hasattr(module, 'weight'):
                weight = module.weight
                
                if weight.dim() > 1:
                    # 计算每个filter/neuron的权重CV
                    w_flat = weight.view(weight.shape[0], -1)
                    
                    mean_w = w_flat.mean(dim=1)
                    std_w = w_flat.std(dim=1, unbiased=False) + 1e-8
                    
                    cv_w = std_w / (torch.abs(mean_w) + 1e-8)
                    cv_mean = cv_w.mean().item()
                    
                    layer_cvs[name] = {
                        'cv_mean': cv_mean,
                        'cv_std': cv_w.std().item(),
                        'cv_max': cv_w.max().item(),
                        'cv_min': cv_w.min().item(),
                        'weight_norm': weight.norm().item(),
                    }
                    all_cvs.append(cv_mean)
        
        avg_cv = np.mean(all_cvs) if all_cvs else 0.0
        
        return avg_cv, layer_cvs
    
    def compute_activation_cv(self, hidden_states):
        """
        计算隐层激活的CV值。
        
        Args:
            hidden_states: (B, N) 或 (T, B, N) 激活张量
        
        Returns:
            cv_activation: float - 激活CV值
            stats: dict - 详细统计
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.mean(dim=0)
        
        if hidden_states.shape[0] < 2:
            return 0.0, {}
        
        # 计算神经元间的CV（不同神经元的激活值标准差）
        mean_act = hidden_states.mean(dim=0)
        std_act = hidden_states.std(dim=0, unbiased=False) + 1e-8
        
        cv_per_neuron = std_act / (torch.abs(mean_act) + 1e-8)
        cv_global = cv_per_neuron.mean().item()
        
        stats = {
            'cv_global': cv_global,
            'cv_per_neuron_mean': cv_per_neuron.mean().item(),
            'cv_per_neuron_std': cv_per_neuron.std().item(),
            'activation_sparsity': (hidden_states == 0).sum().item() / hidden_states.numel(),
            'activation_mean': hidden_states.mean().item(),
        }
        
        return cv_global, stats
    
    def report(self, epoch, spike_outputs=None, hidden_states=None):
        """生成诊断报告"""
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} - CV Diagnostics Report")
            print(f"{'='*80}\n")
        
        # Firing rate CV
        if spike_outputs is not None and spike_outputs.numel() > 0:
            fr_cv, fr_stats = self.compute_firing_rate_cv(spike_outputs)
            self.cv_history['firing_rate_cv'].append(fr_cv)
            
            if self.verbose:
                print("📊 Firing Rate CV (激活值倍数差异度):")
                print(f"  - Global CV: {fr_cv:.4f}")
                print(f"  - Mean firing rate: {fr_stats['mean_firing_rate']:.4f}")
                print(f"  - Sparsity: {fr_stats['sparsity']:.4f} (neurons with 0 fires)")
                print(f"  ✓ Higher CV is better - indicates heterogeneous firing patterns\n")
        
        # Weight CV  
        w_cv, layer_cvs = self.compute_weight_cv()
        self.cv_history['weight_cv'].append(w_cv)
        
        if self.verbose:
            print("📊 Weight CV (权重参数倍数差异度):")
            print(f"  - Average CV: {w_cv:.4f}")
            
            # 显示前3层
            for i, (name, stats) in enumerate(list(layer_cvs.items())[:3]):
                print(f"  - {name}: CV={stats['cv_mean']:.4f} (range: {stats['cv_min']:.4f}-{stats['cv_max']:.4f})")
            
            if len(layer_cvs) > 3:
                print(f"  ... and {len(layer_cvs)-3} more layers")
            
            print(f"  ✓ Higher weight CV suggests parameter diversity (Sparse Coding principle)\n")
        
        # Activation CV
        if hidden_states is not None:
            act_cv, act_stats = self.compute_activation_cv(hidden_states)
            self.cv_history['activation_cv'].append(act_cv)
            
            if self.verbose:
                print("📊 Activation CV (隐层神经元多样性):")
                print(f"  - Global CV: {act_cv:.4f}")
                print(f"  - Sparsity: {act_stats.get('activation_sparsity', 0):.4f}")
                print(f"  ✓ Higher activation CV indicates neuron specialization\n")
        
        self.cv_history['epoch'].append(epoch)
        
        if self.verbose:
            print("💡 解释:")
            print("  CV (Coefficient of Variation) = σ / μ")
            print("  - Low CV:  同质化（所有值相似）")
            print("  - High CV: 异质化（值差异大）")
            print("  ")
            print("  对于SNN:")
            print("  - 高Firing Rate CV: 不同神经元放电活跃度差异大 → 稀疏表征")
            print("  - 高Weight CV:      权重参数多样性强 → 更好的特征学习")
            print("  - 高Activation CV:  不同神经元激活差异大 → 特征专一化")
            print(f"\n{'='*80}\n")
    
    def plot_cv_trends(self, save_path=None):
        """绘制CV值的变化趋势"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not installed, skipping plot")
            return
        
        if not self.cv_history['epoch']:
            print("⚠️  No CV history to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Firing rate CV
        if self.cv_history['firing_rate_cv']:
            axes[0].plot(self.cv_history['epoch'], self.cv_history['firing_rate_cv'], 'b-o')
            axes[0].set_title('Firing Rate CV Trend')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('CV Value')
            axes[0].grid(True, alpha=0.3)
        
        # Weight CV
        if self.cv_history['weight_cv']:
            axes[1].plot(self.cv_history['epoch'], self.cv_history['weight_cv'], 'g-o')
            axes[1].set_title('Weight CV Trend')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('CV Value')
            axes[1].grid(True, alpha=0.3)
        
        # Activation CV
        if self.cv_history['activation_cv']:
            axes[2].plot(self.cv_history['epoch'], self.cv_history['activation_cv'], 'r-o')
            axes[2].set_title('Activation CV Trend')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('CV Value')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Plot saved to {save_path}")
        else:
            plt.show()
    
    def save_report(self, output_dir):
        """保存诊断报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV
        import csv
        
        csv_path = output_dir / "cv_diagnostics.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'firing_rate_cv', 'weight_cv', 'activation_cv'])
            
            for i, epoch in enumerate(self.cv_history['epoch']):
                row = [
                    epoch,
                    self.cv_history['firing_rate_cv'][i] if i < len(self.cv_history['firing_rate_cv']) else '',
                    self.cv_history['weight_cv'][i] if i < len(self.cv_history['weight_cv']) else '',
                    self.cv_history['activation_cv'][i] if i < len(self.cv_history['activation_cv']) else '',
                ]
                writer.writerow(row)
        
        print(f"✅ Report saved to {csv_path}")


# 使用示例
if __name__ == "__main__":
    import sys
    
    # 示例：创建一个简单的模型并诊断
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3)
            self.fc = nn.Linear(32*30*30, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleNet()
    diag = CVDiagnostics(model)
    
    # 示例spike outputs (T, B, C, H, W)
    spike_outputs = torch.randint(0, 2, (4, 2, 32, 30, 30)).float()
    
    # 示例hidden states (B, N)
    hidden_states = torch.randn(2, 256)
    
    print("示例诊断报告:")
    diag.report(epoch=0, spike_outputs=spike_outputs, hidden_states=hidden_states)
