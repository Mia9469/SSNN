#!/bin/bash

# 快速测试脚本 - CV Loss v2 版本
# 使用改进的多层CV正则化
# 运行时间: 约 20-30 分钟（20 epochs）

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "${PROJECT_DIR}/../.venv/bin/python" ]; then
        PYTHON="${PROJECT_DIR}/../.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

echo "=================================="
echo "CV Loss v2 快速测试"
echo "=================================="
echo ""
echo "使用改进的权重CV损失正则化"
echo "基于Sparse Coding原理：最大化权重参数多样性"
echo ""

# 配置参数
EPOCHS=20
BATCH_SIZE=64
LR=0.0003
DATA_DIR="${PROJECT_DIR}/data"
CONFIG_DIR="${PROJECT_DIR}/conf/cifar10"

# 测试配置
CONFIGS=(
    "2_256_300E_t4.yml"  # 标准配置
)

# CV Loss参数（使用普通数组，兼容 macOS 默认 bash 3.2）
TEST_NAMES=("baseline" "cv_v1" "cv_v2" "cv_v2_full")
TEST_ARGS=(
    "--use-cv-loss --lambda-cv 0"
    "--use-cv-loss --lambda-cv 0.01 --cv-weight 1.0"
    "--use-cv-loss --lambda-cv 0.01 --use-weight-cv --lambda-weight-cv 0.001"
    "--use-cv-loss --lambda-cv 0.01 --use-weight-cv --lambda-weight-cv 0.001 --use-act-cv --lambda-act-cv 0.005"
)

# 运行测试
for config_file in "${CONFIGS[@]}"; do
    config_path="${CONFIG_DIR}/${config_file}"
    
    if [ ! -f "$config_path" ]; then
        echo "❌ 配置文件未找到: $config_path"
        exit 1
    fi
    
    echo "───────────────────────────────────"
    echo "配置: $config_file"
    echo "───────────────────────────────────"
    echo ""
    
    for idx in "${!TEST_NAMES[@]}"; do
        test_name="${TEST_NAMES[$idx]}"
        cv_args="${TEST_ARGS[$idx]}"
        
        echo "🔄 运行: $test_name"
        echo "   参数: $cv_args"
        
        # 创建输出目录
        output_dir="output_cv_tests/cv_v2_${test_name}_$(date +%Y%m%d-%H%M%S)"
        mkdir -p "${output_dir}"
        
        # 运行训练
        "$PYTHON" train.py \
            --config "$config_path" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LR \
            --output "$output_dir" \
            --no-prefetcher \
            --TET \
            --TET-means 1 \
            --TET-lamb 1 \
            $cv_args \
            2>&1 | tee "${output_dir}/test_${test_name}.log"
        
        # 提取最佳准确率
        if [ -f "${output_dir}/summary.csv" ]; then
            best_acc=$(tail -1 "${output_dir}/summary.csv" | cut -d',' -f6)
            echo "✅ $test_name 完成: 最佳准确率 = $best_acc%"
        fi
        
        echo ""
    done
done

echo "=================================="
echo "✅ 快速测试完成！"
echo "=================================="
echo ""
echo "结果对比："
echo "  1. baseline:   激活值CV = 0（无CV损失）"
echo "  2. cv_v1:      仅激活值CV（原始实现）"
echo "  3. cv_v2:      激活值CV + 权重CV（推荐！）"
echo "  4. cv_v2_full: 三层CV正则化（最强）"
echo ""
echo "预期结果："
echo "  - cv_v2 应该比 cv_v1 好 +0.3-0.5%"
echo "  - cv_v2_full 可能带来额外 +0.1-0.3% 的改进"
echo ""
echo "分析结果:"
"$PYTHON" -c "
import os
import csv
from pathlib import Path

results = {}
for d in Path('output_cv_tests').glob('cv_v2_*'):
    name = d.name.split('cv_v2_')[1].rsplit('_', 1)[0]
    csv_file = d / 'summary.csv'
    if csv_file.exists():
        with open(csv_file) as f:
            lines = list(csv.reader(f))
            if len(lines) > 1:
                last_row = lines[-1]
                acc = float(last_row[6]) if len(last_row) > 6 else None
                epoch = int(last_row[0]) if last_row else 0
                results[name] = (acc, epoch)

print('✅ 快速测试结果对比:')
for name in ['baseline', 'cv_v1', 'cv_v2', 'cv_v2_full']:
    if name in results:
        acc, epoch = results[name]
        print(f'   {name:12s}: {acc:.2f}% @ Epoch {epoch}')
    else:
        print(f'   {name:12s}: (未完成)')

# 计算改进
if 'cv_v1' in results and 'cv_v2' in results:
    diff = results['cv_v2'][0] - results['cv_v1'][0]
    print(f'\\n🎯 cv_v2 比 cv_v1 的改进: {diff:+.2f}%')
" 2>/dev/null || echo "（等待测试完成后自动分析）"
