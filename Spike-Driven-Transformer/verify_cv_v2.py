#!/usr/bin/env python3
"""
criterion_v2.py 快速验证脚本

验证：
1. 所有function都能正确导入
2. CV loss能正确计算
3. 与原始criterion兼容
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

def test_imports():
    """测试导入"""
    print("=" * 80)
    print("1️⃣  测试导入")
    print("=" * 80 + "\n")
    
    try:
        import criterion
        print("✅ 成功导入 criterion (原始)")
    except Exception as e:
        print(f"❌ 导入criterion失败: {e}")
        return False
    
    try:
        import criterion_v2
        print("✅ 成功导入 criterion_v2 (改进版)")
    except Exception as e:
        print(f"❌ 导入criterion_v2失败: {e}")
        return False
    
    return True


def test_criterion_v2_functions():
    """测试criterion_v2的所有函数"""
    print("\n" + "=" * 80)
    print("2️⃣  测试criterion_v2的函数")
    print("=" * 80 + "\n")
    
    import criterion_v2
    
    # 创建虚拟数据
    B, T, C, H, W = 4, 4, 32, 32, 32
    spike_outputs = torch.randint(0, 2, (T, B, C, H, W)).float()
    
    # 创建简单的模型
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.Linear(64 * 26 * 26, 10)
    )
    
    tests_passed = 0
    tests_total = 0
    
    # 测试1: firing_rate_cv_loss
    tests_total += 1
    try:
        loss = criterion_v2.firing_rate_cv_loss(spike_outputs, lambda_cv=0.01)
        print(f"✅ firing_rate_cv_loss: {loss.item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ firing_rate_cv_loss失败: {e}")
    
    # 测试2: weight_cv_loss
    tests_total += 1
    try:
        loss = criterion_v2.weight_cv_loss(model, lambda_weight_cv=0.001)
        print(f"✅ weight_cv_loss: {loss.item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ weight_cv_loss失败: {e}")
    
    # 测试3: activation_cv_loss
    tests_total += 1
    try:
        hidden = torch.randn(B, 128)
        loss = criterion_v2.activation_cv_loss(hidden, lambda_act_cv=0.005)
        print(f"✅ activation_cv_loss: {loss.item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ activation_cv_loss失败: {e}")
    
    # 测试4: combined_loss
    tests_total += 1
    try:
        # TET格式: (T, B, C) 其中T是时间步，B是batch，C是类别
        T = 4
        output = torch.randn(T, B, 10)
        target = torch.randint(0, 10, (B,))
        loss_fn = nn.CrossEntropyLoss()
        
        loss = criterion_v2.combined_loss(
            output, target, loss_fn,
            use_cv_loss=True,
            use_weight_cv=True,
            use_act_cv=True,
            lambda_cv=0.01,
            lambda_weight_cv=0.001,
            lambda_act_cv=0.005,
            means=0.0,
            lamb=0.0,
            model=model,
            hidden_states=hidden
        )
        print(f"✅ combined_loss: {loss.item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ combined_loss失败: {e}")
    
    print(f"\n总计: {tests_passed}/{tests_total} 通过\n")
    return tests_passed == tests_total


def test_backward_compatibility():
    """测试向后兼容性"""
    print("=" * 80)
    print("3️⃣  测试向后兼容性")
    print("=" * 80 + "\n")
    
    import criterion_v2
    
    B = 4
    T = 4  # 时间步
    output = torch.randn(T, B, 10)  # TET格式: (T, B, C)
    target = torch.randint(0, 10, (B,))
    loss_fn = nn.CrossEntropyLoss()
    
    # 测试所有参数关闭的情况（应该只返回基础loss）
    tests_passed = 0
    tests_total = 0
    
    # 测试1: 无CV项
    tests_total += 1
    try:
        loss1 = criterion_v2.combined_loss(
            output, target, loss_fn,
            use_cv_loss=False,
            use_weight_cv=False,
            use_act_cv=False,
            means=0.0,
            lamb=0.0
        )
        
        # 用原始criterion比较
        import criterion as crit
        loss2 = crit.TET_loss(output, target, loss_fn, means=0.0, lamb=0.0)
        
        diff = abs(loss1.item() - loss2.item())
        if diff < 1e-5:
            print(f"✅ 无CV项: {loss1.item():.6f} ≈ {loss2.item():.6f} (差异: {diff:.2e})")
            tests_passed += 1
        else:
            print(f"❌ 无CV项: 差异过大 {diff:.2e}")
    except Exception as e:
        print(f"❌ 无CV项失败: {e}")
    
    # 测试2: 仅激活值CV（兼容v1）
    tests_total += 1
    try:
        loss = criterion_v2.combined_loss(
            output, target, loss_fn,
            use_cv_loss=True,
            use_weight_cv=False,
            use_act_cv=False,
            lambda_cv=0.01,
            means=0.0,
            lamb=0.0
        )
        print(f"✅ 仅激活值CV（v1兼容）: {loss.item():.6f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 仅激活值CV失败: {e}")
    
    print(f"\n总计: {tests_passed}/{tests_total} 通过\n")
    return tests_passed == tests_total


def test_cv_diagnostics():
    """测试CV诊断工具"""
    print("=" * 80)
    print("4️⃣  测试CV诊断工具")
    print("=" * 80 + "\n")
    
    try:
        from cv_diagnostics import CVDiagnostics
        
        # 创建模型
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.Linear(16 * 30 * 30, 10)
        )
        
        # 创建诊断工具
        diag = CVDiagnostics(model, verbose=False)
        
        # 测试数据
        spike_outputs = torch.randint(0, 2, (4, 2, 16, 30, 30)).float()
        hidden_states = torch.randn(2, 256)
        
        # 运行诊断
        diag.report(epoch=0, spike_outputs=spike_outputs, hidden_states=hidden_states)
        
        print("✅ CVDiagnostics 运行成功")
        print(f"   - firing_rate_cv 记录: {len(diag.cv_history['firing_rate_cv'])}")
        print(f"   - weight_cv 记录: {len(diag.cv_history['weight_cv'])}")
        print(f"   - activation_cv 记录: {len(diag.cv_history['activation_cv'])}")
        
        return True
    except Exception as e:
        print(f"❌ CVDiagnostics失败: {e}")
        return False


def main():
    """主函数"""
    
    # 改变到工作目录
    work_dir = Path(__file__).parent
    import sys
    sys.path.insert(0, str(work_dir))
    
    results = []
    
    # 运行所有测试
    results.append(("导入测试", test_imports()))
    
    if results[-1][1]:  # 只有导入成功才继续
        results.append(("函数测试", test_criterion_v2_functions()))
        results.append(("向后兼容性", test_backward_compatibility()))
        results.append(("诊断工具", test_cv_diagnostics()))
    
    # 打印总体结果
    print("\n" + "=" * 80)
    print("📊 验证总结")
    print("=" * 80 + "\n")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✨ 所有验证通过！criterion_v2 已准备就绪。")
        print("\n开始使用:")
        print("  python train.py \\")
        print("    --use-cv-loss --lambda-cv 0.01 \\")
        print("    --use-weight-cv --lambda-weight-cv 0.001")
        return 0
    else:
        print("❌ 某些验证失败。请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
