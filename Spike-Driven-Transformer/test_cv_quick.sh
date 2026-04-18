#!/bin/bash

# ============================================================================
# Quick Test: CV Loss Functionality Verification
# ============================================================================
# This script runs a quick test with only 20 epochs to verify:
# 1. CV loss functions correctly compute
# 2. Training loop integrates without errors
# 3. Model converges with CV loss
#
# Runtime: ~5-10 minutes on GPU
# ============================================================================

set -e

# Ensure torchinfo is installed
echo "Checking dependencies..."
/opt/anaconda3/bin/python -c "import torchinfo" 2>/dev/null || {
    echo "Installing missing dependencies..."
    /opt/anaconda3/bin/python -m pip install torchinfo -q
}

echo "============================================================================"
echo "Quick Test: CV Loss Functionality"
echo "============================================================================"
echo ""
echo "This test runs 20 epochs (instead of 300) to verify CV loss works correctly."
echo "Estimated runtime: 5-10 minutes on GPU"
echo ""

# Configuration for quick test
CONFIG="conf/cifar10/2_256_300E_t4.yml"
MODEL="sdt"
SPIKE_MODE="lif"
WORKERS=2
QUICK_EPOCHS=20  # Quick test with 20 epochs only
BATCH_SIZE=64

# Output directory
OUTPUT_BASE="./output_cv_quick_test"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "${OUTPUT_BASE}"

# ============================================================================
# Test 1: Baseline (No CV Loss)  - Just to establish baseline
# ============================================================================
echo ""
echo "Test 1: Baseline Model (No CV Loss)"
echo "  Running for 20 epochs..."
echo ""

OUTPUT_DIR_1="${OUTPUT_BASE}/baseline_quick_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR_1}"

/opt/anaconda3/bin/python train.py \
    -c "${CONFIG}" \
    --model "${MODEL}" \
    --spike-mode "${SPIKE_MODE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR_1}" \
    --experiment "sdt-quick-baseline" \
    --epochs "${QUICK_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --no-prefetcher

echo "✓ Baseline test completed"
echo "  Results: ${OUTPUT_DIR_1}/train/*/summary.csv"
echo ""

# ============================================================================
# Test 2: CV Loss (lambda_cv=0.01)
# ============================================================================
echo ""
echo "Test 2: CV Loss Enabled (λ=0.01, w=1.0)"
echo "  Running for 20 epochs..."
echo ""

OUTPUT_DIR_2="${OUTPUT_BASE}/cv_loss_quick_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR_2}"

/opt/anaconda3/bin/python train.py \
    -c "${CONFIG}" \
    --model "${MODEL}" \
    --spike-mode "${SPIKE_MODE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR_2}" \
    --experiment "sdt-quick-cv-loss" \
    --use-cv-loss \
    --lambda-cv 0.01 \
    --cv-weight 1.0 \
    --epochs "${QUICK_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --no-prefetcher

echo "✓ CV Loss test completed"
echo "  Results: ${OUTPUT_DIR_2}/train/*/summary.csv"
echo ""

# ============================================================================
# Summary & Comparison
# ============================================================================
echo "============================================================================"
echo "Quick Test Completed!"
echo "============================================================================"
echo ""
echo "Baseline results:"
find "${OUTPUT_DIR_1}" -name "summary.csv" -exec echo "  File: {}" \; -exec tail -5 {} \;
echo ""
echo "CV Loss results:"
find "${OUTPUT_DIR_2}" -name "summary.csv" -exec echo "  File: {}" \; -exec tail -5 {} \;
echo ""
echo "Next: Compare the final accuracy (epoch 19) between baseline and CV loss."
echo "      Ideally, CV Loss should achieve comparable or better accuracy."
echo ""
echo "============================================================================"
