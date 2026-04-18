#!/bin/bash

# ============================================================================
# Spike-Driven Transformer: CV Loss Regularization vs Baseline Comparison
# ============================================================================
# This script runs parallel experiments to compare:
# 1. Baseline model (original TET loss only)
# 2. Model with CV loss regularization (sparse firing patterns)
# 3. Model with different CV loss strengths
#
# Date: March 21, 2026
# ============================================================================

set -e

# Ensure torchinfo is installed
echo "Checking dependencies..."
/opt/anaconda3/bin/python -c "import torchinfo" 2>/dev/null || {
    echo "Installing missing dependencies..."
    /opt/anaconda3/bin/python -m pip install torchinfo -q
}

# Configuration
DATASET="cifar10"
CONFIG="conf/cifar10/2_256_300E_t4.yml"
MODEL="sdt"
SPIKE_MODE="lif"
WORKERS=2
EPOCHS=300  # You can reduce for quick testing
BATCH_SIZE=64

# Output directories
BASE_OUTPUT="./output_cv_experiments"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Log file
LOG_FILE="${BASE_OUTPUT}/experiments_${TIMESTAMP}.log"
mkdir -p "${BASE_OUTPUT}"

echo "============================================================================" | tee -a "${LOG_FILE}"
echo "CV Loss Regularization Experiments" | tee -a "${LOG_FILE}"
echo "Start time: $(date)" | tee -a "${LOG_FILE}"
echo "============================================================================" | tee -a "${LOG_FILE}"

# ============================================================================
# Experiment 1: Baseline (No CV Loss)
# ============================================================================
echo "" | tee -a "${LOG_FILE}"
echo "----------- Experiment 1: Baseline Model (No CV Loss) -----------" | tee -a "${LOG_FILE}"
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  - Model: ${MODEL}" | tee -a "${LOG_FILE}"
echo "  - Dataset: ${DATASET}" | tee -a "${LOG_FILE}"
echo "  - Config: ${CONFIG}" | tee -a "${LOG_FILE}"
echo "  - Epochs: ${EPOCHS}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

OUTPUT_DIR_1="${BASE_OUTPUT}/baseline_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR_1}"

/opt/anaconda3/bin/python train.py \
    -c "${CONFIG}" \
    --model "${MODEL}" \
    --spike-mode "${SPIKE_MODE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR_1}" \
    --experiment "sdt-cifar10-baseline" \
    --epochs "${EPOCHS}" \
    --no-prefetcher \
    2>&1 | tee -a "${LOG_FILE}" "${OUTPUT_DIR_1}/train.log"

BASELINE_SUMMARY="${OUTPUT_DIR_1}/train/*/summary.csv"
echo "Baseline training completed. Results saved to: ${OUTPUT_DIR_1}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# ============================================================================
# Experiment 2: CV Loss (lambda_cv=0.01, cv_weight=1.0)
# ============================================================================
echo "----------- Experiment 2: CV Loss (λ=0.01, w=1.0) -----------" | tee -a "${LOG_FILE}"
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  - CV Loss: ENABLED" | tee -a "${LOG_FILE}"
echo "  - Lambda CV: 0.01" | tee -a "${LOG_FILE}"
echo "  - CV Weight: 1.0" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

OUTPUT_DIR_2="${BASE_OUTPUT}/cv_loss_01_w10_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR_2}"

/opt/anaconda3/bin/python train.py \
    -c "${CONFIG}" \
    --model "${MODEL}" \
    --spike-mode "${SPIKE_MODE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR_2}" \
    --experiment "sdt-cifar10-cv-loss-01" \
    --use-cv-loss \
    --lambda-cv 0.01 \
    --cv-weight 1.0 \
    --epochs "${EPOCHS}" \
    --no-prefetcher \
    2>&1 | tee -a "${LOG_FILE}" "${OUTPUT_DIR_2}/train.log"

echo "CV Loss (0.01) training completed. Results saved to: ${OUTPUT_DIR_2}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# ============================================================================
# Experiment 3: CV Loss (lambda_cv=0.005, cv_weight=1.0) - Weaker regularization
# ============================================================================
echo "----------- Experiment 3: CV Loss (λ=0.005, w=1.0) - Weaker -----------" | tee -a "${LOG_FILE}"
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  - CV Loss: ENABLED" | tee -a "${LOG_FILE}"
echo "  - Lambda CV: 0.005 (weaker regularization)" | tee -a "${LOG_FILE}"
echo "  - CV Weight: 1.0" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

OUTPUT_DIR_3="${BASE_OUTPUT}/cv_loss_005_w10_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR_3}"

/opt/anaconda3/bin/python train.py \
    -c "${CONFIG}" \
    --model "${MODEL}" \
    --spike-mode "${SPIKE_MODE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR_3}" \
    --experiment "sdt-cifar10-cv-loss-005" \
    --use-cv-loss \
    --lambda-cv 0.005 \
    --cv-weight 1.0 \
    --epochs "${EPOCHS}" \
    --no-prefetcher \
    2>&1 | tee -a "${LOG_FILE}" "${OUTPUT_DIR_3}/train.log"

echo "CV Loss (0.005) training completed. Results saved to: ${OUTPUT_DIR_3}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# ============================================================================
# Experiment 4: CV Loss (lambda_cv=0.02, cv_weight=1.0) - Stronger regularization
# ============================================================================
echo "----------- Experiment 4: CV Loss (λ=0.02, w=1.0) - Stronger -----------" | tee -a "${LOG_FILE}"
echo "Configuration:" | tee -a "${LOG_FILE}"
echo "  - CV Loss: ENABLED" | tee -a "${LOG_FILE}"
echo "  - Lambda CV: 0.02 (stronger regularization)" | tee -a "${LOG_FILE}"
echo "  - CV Weight: 1.0" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

OUTPUT_DIR_4="${BASE_OUTPUT}/cv_loss_02_w10_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR_4}"

/opt/anaconda3/bin/python train.py \
    -c "${CONFIG}" \
    --model "${MODEL}" \
    --spike-mode "${SPIKE_MODE}" \
    --workers "${WORKERS}" \
    --output "${OUTPUT_DIR_4}" \
    --experiment "sdt-cifar10-cv-loss-02" \
    --use-cv-loss \
    --lambda-cv 0.02 \
    --cv-weight 1.0 \
    --epochs "${EPOCHS}" \
    --no-prefetcher \
    2>&1 | tee -a "${LOG_FILE}" "${OUTPUT_DIR_4}/train.log"

echo "CV Loss (0.02) training completed. Results saved to: ${OUTPUT_DIR_4}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================" | tee -a "${LOG_FILE}"
echo "All experiments completed!" | tee -a "${LOG_FILE}"
echo "============================================================================" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Output directories:" | tee -a "${LOG_FILE}"
echo "  1. Baseline (no CV):        ${OUTPUT_DIR_1}" | tee -a "${LOG_FILE}"
echo "  2. CV Loss (λ=0.01, w=1.0):  ${OUTPUT_DIR_2}" | tee -a "${LOG_FILE}"
echo "  3. CV Loss (λ=0.005, w=1.0): ${OUTPUT_DIR_3}" | tee -a "${LOG_FILE}"
echo "  4. CV Loss (λ=0.02, w=1.0):  ${OUTPUT_DIR_4}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Master log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Next steps:" | tee -a "${LOG_FILE}"
echo "  1. Compare accuracy curves within each output directory's summary.csv" | tee -a "${LOG_FILE}"
echo "  2. Analyze firing rates using: python compare_cv_experiments.py" | tee -a "${LOG_FILE}"
echo "  3. Plot results: python plot_cv_comparison.py" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Completion time: $(date)" | tee -a "${LOG_FILE}"
echo "============================================================================" | tee -a "${LOG_FILE}"
