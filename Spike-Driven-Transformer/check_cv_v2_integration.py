#!/usr/bin/env python3
"""
CV Loss v2 快速集成检查清单

检查所有文件是否都已正确修改和集成。
"""

import os
import sys
from pathlib import Path

def check_integration():
    """检查v2集成状态"""
    
    issues = []
    warnings = []
    success = []
    
    base_dir = Path(__file__).parent
    
    # 检查文件存在性
    files_to_check = {
        'criterion_v2.py': '新的多层CV损失实现',
        'criterion.py': '原始损失函数（应保留）',
        'train.py': '修改后支持v2参数',
        'cv_diagnostics.py': 'CV值诊断和监控工具',
        'SPARSE_CODING_STRATEGY.md': '理论解释文档',
    }
    
    print("=" * 80)
    print("CV Loss v2 集成检查")
    print("=" * 80 + "\n")
    
    for filename, description in files_to_check.items():
        filepath = base_dir / filename
        if filepath.exists():
            success.append(f"✅ {filename}: {description}")
        else:
            issues.append(f"❌ {filename} 未找到")
    
    # 检查train.py中的关键修改
    print("检查train.py的关键修改:\n")
    
    train_path = base_dir / 'train.py'
    with open(train_path) as f:
        content = f.read()
        
        # 检查导入
        if 'import criterion_v2' in content:
            success.append("✅ train.py: 导入criterion_v2")
        else:
            issues.append("❌ train.py: 缺少 import criterion_v2")
        
        # 检查新参数
        params = [
            ('--use-weight-cv', '权重CV参数'),
            ('--lambda-weight-cv', '权重CV强度参数'),
            ('--use-act-cv', '隐层激活CV参数'),
            ('--lambda-act-cv', '隐层激活CV强度参数'),
        ]
        
        for param, desc in params:
            if param in content:
                success.append(f"✅ train.py: 添加 {param}")
            else:
                issues.append(f"❌ train.py: 缺少 {param}")
        
        # 检查loss计算修改
        if 'HAS_CV_V2' in content:
            success.append("✅ train.py: 条件导入v2逻辑")
        else:
            issues.append("❌ train.py: 缺少v2导入逻辑")
        
        if 'criterion_v2.combined_loss' in content:
            success.append("✅ train.py: 调用criterion_v2.combined_loss")
        else:
            issues.append("❌ train.py: 未找到对v2的调用")
    
    # 检查criterion_v2.py
    print("\n检查criterion_v2.py的函数:\n")
    
    v2_path = base_dir / 'criterion_v2.py'
    if v2_path.exists():
        with open(v2_path) as f:
            v2_content = f.read()
            
            functions = {
                'firing_rate_cv_loss': '原始激活值CV损失',
                'weight_cv_loss': '权重参数CV损失（NEW）',
                'activation_cv_loss': '隐层激活CV损失（NEW）',
                'combined_loss': '组合损失函数',
            }
            
            for func_name, desc in functions.items():
                if f'def {func_name}' in v2_content:
                    success.append(f"✅ criterion_v2.py: 定义 {func_name}()")
                else:
                    issues.append(f"❌ criterion_v2.py: 缺少 {func_name}()")
    
    # 打印结果
    print("\n" + "=" * 80)
    print("检查结果")
    print("=" * 80 + "\n")
    
    categories = [
        ("✅ 成功项目", success),
        ("⚠️  警告", warnings),
        ("❌ 问题", issues),
    ]
    
    for title, items in categories:
        if items:
            print(f"{title}:")
            for item in items:
                print(f"  {item}")
            print()
    
    if issues:
        print(f"\n❌ 检查失败：发现 {len(issues)} 个问题\n")
        return False
    elif warnings:
        print(f"\n⚠️  检查通过，但有 {len(warnings)} 个警告\n")
        return True
    else:
        print(f"\n✅ 检查成功！所有文件和修改都已正确集成\n")
        return True


def show_usage():
    """显示使用说明"""
    
    print("\n" + "=" * 80)
    print("CV Loss v2 使用指南")
    print("=" * 80 + "\n")
    
    examples = [
        {
            'title': '1️⃣  仅使用激活值CV（原始方法）',
            'cmd': '''python train.py \\
  --use-cv-loss \\
  --lambda-cv 0.01 \\
  --cv-weight 1.0''',
            'note': '这是原来的实现，只约束输出激活值多样性'
        },
        {
            'title': '2️⃣  添加权重CV（推荐！）',
            'cmd': '''python train.py \\
  --use-cv-loss \\
  --lambda-cv 0.01 \\
  --use-weight-cv \\
  --lambda-weight-cv 0.001''',
            'note': '基于Sparse Coding原理，最大化权重参数多样性'
        },
        {
            'title': '3️⃣  完整三层CV正则化（最强）',
            'cmd': '''python train.py \\
  --use-cv-loss \\
  --lambda-cv 0.01 \\
  --use-weight-cv \\
  --lambda-weight-cv 0.001 \\
  --use-act-cv \\
  --lambda-act-cv 0.005''',
            'note': '同时优化激活值、权重、隐层的多样性'
        },
    ]
    
    for ex in examples:
        print(f"{ex['title']}")
        print(f"命令：")
        print(f"  {ex['cmd']}")
        print(f"说明：{ex['note']}\n")
    
    print("=" * 80 + "\n")


def show_monitoring():
    """显示监控指南"""
    
    print("\n" + "=" * 80)
    print("CV值监控和诊断")
    print("=" * 80 + "\n")
    
    print("在训练过程中监控CV值的变化：\n")
    
    print("选项1：实时进度监控")
    print("  python cv_training_status.py quick  # 快速测试")
    print("  python cv_training_status.py full   # 完整训练\n")
    
    print("选项2：制作诊断报告（在train.py中整合）")
    print("  from cv_diagnostics import CVDiagnostics")
    print("  diag = CVDiagnostics(model)")
    print("  diag.report(epoch=e, spike_outputs=outputs, hidden_states=acts)")
    print("  diag.plot_cv_trends('cv_trends.png')\n")
    
    print("预期观察到的变化：")
    print("  ✓ firing_rate_cv 逐步增加（激活值变得更稀疏异质）")
    print("  ✓ weight_cv 逐步增加（权重变得更多样化）")
    print("  ✓ activation_cv 逐步增加（隐层神经元更专一化）")
    print("  ✓ 这些伴随着训练损失下降和准确率提升\n")
    
    print("=" * 80 + "\n")


def main():
    """主函数"""
    
    if not check_integration():
        print("请检查以上问题并修正！\n")
        sys.exit(1)
    
    show_usage()
    show_monitoring()
    
    print("✨ CV Loss v2 已准备好！开始训练：")
    print("  bash test_cv_quick.sh   # 快速验证（20 epochs）")
    print("  bash test_cv_loss.sh    # 完整实验（300 epochs）\n")


if __name__ == '__main__':
    main()
