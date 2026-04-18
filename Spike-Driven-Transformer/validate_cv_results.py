#!/usr/bin/env python3

"""
CV Loss Complete Experiment Validation

运行完整的CV Loss实验，并在完成后自动验证结果是否合格。
"""

import os
import glob
import pandas as pd
from pathlib import Path
import time
from tabulate import tabulate
import sys


class ExperimentValidator:
    """验证CV Loss实验结果的类"""
    
    def __init__(self, base_dir="./output_cv_experiments"):
        self.base_dir = base_dir
        self.baseline_threshold_top1 = 93.0  # 期望Baseline在原始300epoch能达到93%+
        self.cv_loss_acceptance_delta = 0.5  # CV Loss可接受的精度下降 ≤ 0.5%
        self.convergence_improvement = -2    # CV Loss应该收敛更快（-2 epoch以内可接受）
    
    def load_experiment(self, exp_dir):
        """加载单个实验的结果"""
        csv_files = glob.glob(f"{exp_dir}/**/summary.csv", recursive=True)
        
        if not csv_files:
            return None
        
        try:
            df = pd.read_csv(csv_files[0])
            if len(df) == 0:
                return None
            
            exp_name = Path(exp_dir).name
            last_row = df.iloc[-1]
            best_top1_idx = df['eval_top1'].idxmax()
            best_row = df.iloc[best_top1_idx]
            
            return {
                'name': exp_name,
                'path': exp_dir,
                'total_epochs': len(df) - 1,
                'final_epoch': int(last_row['epoch']),
                'final_train_loss': float(last_row['train_loss']),
                'final_eval_loss': float(last_row['eval_loss']),
                'final_top1': float(last_row['eval_top1']),
                'final_top5': float(last_row['eval_top5']),
                'best_top1': float(best_row['eval_top1']),
                'best_top1_epoch': int(best_row['epoch']),
                'best_top5': float(best_row['eval_top5']),
                'complete': len(df) > 300,  # Check if reached 300 epochs
                'dataframe': df
            }
        except Exception as e:
            print(f"ERROR loading {exp_dir}: {e}")
            return None
    
    def get_all_experiments(self):
        """获取所有实验"""
        if not os.path.exists(self.base_dir):
            return {}
        
        experiments = {
            'baseline': [],
            'cv-loss-005': [],
            'cv-loss-01': [],
            'cv-loss-02': []
        }
        
        exp_dirs = sorted(glob.glob(f"{self.base_dir}/*"))
        
        for exp_dir in sorted(exp_dirs, reverse=True)[:20]:  # 最多检查20个
            result = self.load_experiment(exp_dir)
            if result:
                exp_name = result['name'].lower()
                if 'baseline' in exp_name:
                    experiments['baseline'].append(result)
                elif '005' in exp_name or 'cv_loss_005' in exp_name:
                    experiments['cv-loss-005'].append(result)
                elif '02' in exp_name or 'cv_loss_02' in exp_name:
                    experiments['cv-loss-02'].append(result)
                else:  # Default to cv-loss-01
                    experiments['cv-loss-01'].append(result)
        
        return experiments
    
    def check_completion(self):
        """检查实验完成情况"""
        experiments = self.get_all_experiments()
        
        summary = {
            'baseline': {'count': len(experiments['baseline']), 'complete': 0},
            'cv-loss-005': {'count': len(experiments['cv-loss-005']), 'complete': 0},
            'cv-loss-01': {'count': len(experiments['cv-loss-01']), 'complete': 0},
            'cv-loss-02': {'count': len(experiments['cv-loss-02']), 'complete': 0},
        }
        
        for exp_type in summary:
            for result in experiments[exp_type]:
                if result['complete'] or result['total_epochs'] >= 299:
                    summary[exp_type]['complete'] += 1
        
        return experiments, summary
    
    def validate_results(self, experiments):
        """验证结果是否合格"""
        
        print("\n╔════════════════════════════════════════════════════════════════════════════╗")
        print("║            CV LOSS 完整实验结果验证                                       ║")
        print("╚════════════════════════════════════════════════════════════════════════════╝\n")
        
        # 检查Baseline
        print("📊 BASELINE 验证")
        print("=" * 80)
        
        if not experiments['baseline']:
            print("❌ 未找到Baseline实验")
            baseline_result = None
        else:
            baseline_result = experiments['baseline'][0]
            print(f"✅ 找到Baseline: {baseline_result['name']}")
            print(f"   - 训练进度: {baseline_result['total_epochs']}/300 epochs")
            print(f"   - 最佳Top-1: {baseline_result['best_top1']:.2f}% (@ Epoch {baseline_result['best_top1_epoch']})")
            print(f"   - 最佳Top-5: {baseline_result['best_top5']:.2f}%")
            print(f"   - 最终验证损失: {baseline_result['final_eval_loss']:.6f}")
            
            if baseline_result['best_top1'] >= 93.0:
                print(f"   ✅ 准确率合格 (≥93.0%)")
            else:
                print(f"   ⚠️  准确率偏低 ({baseline_result['best_top1']:.2f}% < 93.0%)")
        
        # 检查CV Loss变体
        print("\n📊 CV LOSS 变体验证")
        print("=" * 80)
        
        cv_variants = ['cv-loss-005', 'cv-loss-01', 'cv-loss-02']
        results_summary = []
        
        for variant in cv_variants:
            if not experiments[variant]:
                print(f"\n⚠️  {variant}: 未找到实验")
                results_summary.append({
                    'variant': variant,
                    'status': '❌ 未运行',
                    'top1': 'N/A',
                    'delta': 'N/A',
                    'verdict': '❌ 失效'
                })
                continue
            
            result = experiments[variant][0]
            print(f"\n✅ {variant}: {result['name']}")
            print(f"   - 训练进度: {result['total_epochs']}/300 epochs")
            print(f"   - 最佳Top-1: {result['best_top1']:.2f}% (@ Epoch {result['best_top1_epoch']})")
            print(f"   - 最佳Top-5: {result['best_top5']:.2f}%")
            print(f"   - 最终验证损失: {result['final_eval_loss']:.6f}")
            
            if baseline_result:
                delta = result['best_top1'] - baseline_result['best_top1']
                epoch_delta = result['best_top1_epoch'] - baseline_result['best_top1_epoch']
                
                print(f"   - 与Baseline对比:")
                print(f"     * Top-1差异: {delta:+.2f}% {'✅' if abs(delta) <= self.cv_loss_acceptance_delta else '❌'}")
                print(f"     * 收敛速度: {epoch_delta:+d} epochs {'✅' if epoch_delta <= self.convergence_improvement else '⚠️'}")
                
                # 判定
                passed = abs(delta) <= self.cv_loss_acceptance_delta
                verdict = "✅ 合格" if passed else "❌ 不合格"
                
                results_summary.append({
                    'variant': variant.replace('cv-loss-', 'λ='),
                    'status': f"✅完成" if result['total_epochs'] >= 299 else f"⏳{result['total_epochs']}/300",
                    'top1': f"{result['best_top1']:.2f}%",
                    'delta': f"{delta:+.2f}%",
                    'epochs': f"{result['best_top1_epoch']}@{baseline_result['best_top1_epoch']}",
                    'verdict': verdict
                })
            else:
                results_summary.append({
                    'variant': variant.replace('cv-loss-', 'λ='),
                    'status': f"✅完成" if result['total_epochs'] >= 299 else f"⏳{result['total_epochs']}/300",
                    'top1': f"{result['best_top1']:.2f}%",
                    'delta': 'N/A',
                    'epochs': f"{result['best_top1_epoch']}",
                    'verdict': '⚠️ 缺Baseline'
                })
        
        # 打印对比表格
        print("\n" + "=" * 80)
        print("📋 对比总结")
        print("=" * 80 + "\n")
        
        if baseline_result:
            new_summary = [
                {
                    'variant': 'BASELINE',
                    'top1': f"{baseline_result['best_top1']:.2f}%",
                    'convergence': f"Epoch {baseline_result['best_top1_epoch']}",
                    'eval_loss': f"{baseline_result['final_eval_loss']:.4f}",
                    'status': '✅ 参考'
                }
            ]
            
            for item in results_summary:
                new_summary.append({
                    'variant': item['variant'],
                    'top1': item['top1'],
                    'convergence': f"Δ{item.get('epochs', 'N/A')}",
                    'accuracy_delta': item['delta'],
                    'verdict': item['verdict']
                })
            
            headers = ['配置', 'Top-1准确率', '收敛速度', '精度差异', '评定']
            table_data = [
                [s.get('variant', ''), 
                 s.get('top1', ''),
                 s.get('convergence', ''),
                 s.get('accuracy_delta', ''),
                 s.get('verdict', '')]
                for s in new_summary
            ]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 总体评定
        print("\n" + "=" * 80)
        print("🎯 最终评定")
        print("=" * 80 + "\n")
        
        if not baseline_result:
            print("❌ Baseline未完成或未找到，无法进行完整评估")
            return False
        
        all_cv_passed = all(
            '✅' in item['verdict'] 
            for item in results_summary 
            if item['verdict'] != '⚠️ 缺Baseline'
        )
        
        if baseline_result['best_top1'] >= 93.0:
            print(f"✅ Baseline通过: Top-1 = {baseline_result['best_top1']:.2f}% ≥ 93.0%")
        else:
            print(f"⚠️  Baseline精度低: Top-1 = {baseline_result['best_top1']:.2f}% < 93.0%")
        
        if all_cv_passed:
            print(f"✅ CV Loss变体通过: 所有配置性能在可接受范围内")
            print(f"\n🎉 实验结果合格！CV Loss正则化有效。")
            return True
        else:
            print(f"❌ CV Loss变体未全部通过: 某些配置性能下降过多")
            print(f"\n📝 建议: 调整λ和权重参数后重新运行")
            return False
    
    def run_monitoring(self):
        """持续监控实验进度"""
        print("\n🔄 开始监控实验进度...\n")
        
        while True:
            experiments, summary = self.check_completion()
            
            # 清空屏幕显示
            print("\033[2J\033[H")  # ANSI clear screen
            
            print("═" * 80)
            print(" 实验进度监控 (每5秒刷新)")
            print("═" * 80 + "\n")
            
            progress_data = [
                ['Baseline', 
                 f"{summary['baseline']['complete']}/{summary['baseline']['count']}", 
                 '✅' if summary['baseline']['complete'] > 0 else '⏳'],
                ['CV Loss λ=0.005', 
                 f"{summary['cv-loss-005']['complete']}/{summary['cv-loss-005']['count']}", 
                 '✅' if summary['cv-loss-005']['complete'] > 0 else '⏳'],
                ['CV Loss λ=0.01', 
                 f"{summary['cv-loss-01']['complete']}/{summary['cv-loss-01']['count']}", 
                 '✅' if summary['cv-loss-01']['complete'] > 0 else '⏳'],
                ['CV Loss λ=0.02', 
                 f"{summary['cv-loss-02']['complete']}/{summary['cv-loss-02']['count']}", 
                 '✅' if summary['cv-loss-02']['complete'] > 0 else '⏳'],
            ]
            
            print(tabulate(progress_data, 
                          headers=['配置', '完成情况', '状态'],
                          tablefmt="grid"))
            
            # 详细进度
            print("\n📊 详细进度:\n")
            
            for exp_type in ['baseline', 'cv-loss-005', 'cv-loss-01', 'cv-loss-02']:
                if experiments[exp_type]:
                    result = experiments[exp_type][0]
                    progress = min(100, int(result['total_epochs'] / 300 * 100))
                    bar_length = 30
                    filled = int(bar_length * progress / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    print(f"{exp_type:<20} [{bar}] {result['total_epochs']}/300 ({progress}%)")
                    if result['best_top1'] > 0:
                        print(f"                      Best Top-1: {result['best_top1']:.2f}% @ Epoch {result['best_top1_epoch']}")
            
            # 检查是否全部完成
            all_done = (summary['baseline']['complete'] > 0 and 
                       summary['cv-loss-005']['complete'] > 0 and
                       summary['cv-loss-01']['complete'] > 0 and
                       summary['cv-loss-02']['complete'] > 0)
            
            if all_done:
                print("\n✅ 所有实验已完成！\n")
                return self.validate_results(experiments)
            
            print("\n⏳ 下一次检查: 5秒后...")
            time.sleep(5)


def main():
    validator = ExperimentValidator()
    
    # 检查当前完成状态
    experiments, summary = validator.check_completion()
    
    total_exps = sum(s['count'] for s in summary.values())
    total_completed = sum(s['complete'] for s in summary.values())
    
    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║            CV LOSS 完整实验验证系统                                       ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")
    
    print(f"📊 当前状态: {total_completed}/{total_exps} 个实验已完成\n")
    
    if total_completed == total_exps and total_exps > 0:
        print("✅ 所有实验已完成，开始验证结果...\n")
        passed = validator.validate_results(experiments)
        sys.exit(0 if passed else 1)
    else:
        print("⏳ 实验仍在进行中...\n")
        print("💡 提示: 按 Ctrl+C 可随时停止监控\n")
        
        try:
            validator.run_monitoring()
        except KeyboardInterrupt:
            print("\n\n⏸️  监控已停止")
            print("可以稍后运行以下命令检查结果:")
            print("  python validate_cv_results.py\n")


if __name__ == "__main__":
    main()
