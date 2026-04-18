#!/usr/bin/env python3

"""
Quick Reference for CV Loss Hyperparameter Tuning

This script provides quick recommendations based on observed results.
"""

QUICK_START = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                      CV LOSS QUICK START GUIDE                             ║
╚═════════════════════════════════════════════════════════════════════════════╝

1. FIRST TIME SETUP
   └─ Check modifications: grep -n "firing_rate_cv_loss\\|use-cv-loss" *.py

2. QUICK TEST (10 minutes)
   └─ bash test_cv_quick.sh

3. STANDARD CONFIGURATION ⭐ (RECOMMENDED)
   └─ python train.py ... --use-cv-loss --lambda-cv 0.01 --cv-weight 1.0

4. FULL COMPARISON (4-24 hours depending on GPU)
   └─ bash test_cv_loss.sh

5. ANALYZE RESULTS
   └─ python compare_cv_experiments.py
   └─ python plot_cv_comparison.py

═════════════════════════════════════════════════════════════════════════════
"""

HYPERPARAMETER_GUIDE = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                    HYPERPARAMETER TUNING GUIDE                             ║
╚═════════════════════════════════════════════════════════════════════════════╝

λ (lambda_cv) - STRENGTH OF CV LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   0.001    ├─ Minimal effect
   0.005    ├─ Light regularization (try if accuracy drops with 0.01)
   0.01 ⭐  ├─ RECOMMENDED
   0.02     ├─ Medium regularization
   0.05     └─ Strong (may hurt accuracy)

w (cv_weight) - RELATIVE WEIGHT vs CLASSIFICATION LOSS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   0.5      ├─ Weak emphasis on sparsity
   1.0 ⭐   ├─ RECOMMENDED (balanced)
   1.5      ├─ Strong emphasis
   2.0      └─ Very strong (may hurt accuracy)

═════════════════════════════════════════════════════════════════════════════
TUNING STRATEGY:
═════════════════════════════════════════════════════════════════════════════

SYMPTOM: Accuracy drops > 0.5%
├─ Issue: CV regularization too strong
├─ Fix: reduce lambda_cv to 0.005
└─ Cmd: --use-cv-loss --lambda-cv 0.005 --cv-weight 1.0

SYMPTOM: CV in firing rates doesn't increase significantly
├─ Issue: CV regularization too weak
├─ Fix: increase lambda_cv to 0.02 OR cv_weight to 1.5
└─ Cmd: --use-cv-loss --lambda-cv 0.02 --cv-weight 1.0

SYMPTOM: Everything looks good!
├─ Accuracy: within -0.3% of baseline ✓
├─ CV increases: observed in firing_num.py ✓
├─ Convergence: similar or faster ✓
└─ Result: OPTIMAL CONFIGURATION FOUND!

═════════════════════════════════════════════════════════════════════════════
"""

COMMAND_REFERENCE = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                      COMMAND REFERENCE                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝

BASELINE TRAINING (no CV loss)
──────────────────────────────────────────────────────────────────────────────
python train.py -c conf/cifar10/2_256_300E_t4.yml \\
    --model sdt --spike-mode lif --workers 2 \\
    --output ./output_baseline


CV LOSS WITH RECOMMENDED PARAMETERS ⭐
──────────────────────────────────────────────────────────────────────────────
python train.py -c conf/cifar10/2_256_300E_t4.yml \\
    --model sdt --spike-mode lif --workers 2 \\
    --use-cv-loss --lambda-cv 0.01 --cv-weight 1.0 \\
    --output ./output_cv_01


CV LOSS WITH WEAKER REGULARIZATION
──────────────────────────────────────────────────────────────────────────────
python train.py -c conf/cifar10/2_256_300E_t4.yml \\
    --model sdt --spike-mode lif --workers 2 \\
    --use-cv-loss --lambda-cv 0.005 --cv-weight 1.0 \\
    --output ./output_cv_005


CV LOSS WITH STRONGER REGULARIZATION
──────────────────────────────────────────────────────────────────────────────
python train.py -c conf/cifar10/2_256_300E_t4.yml \\
    --model sdt --spike-mode lif --workers 2 \\
    --use-cv-loss --lambda-cv 0.02 --cv-weight 1.5 \\
    --output ./output_cv_02


WITH ADDITIONAL TET LOSS
──────────────────────────────────────────────────────────────────────────────
python train.py -c conf/cifar10/2_256_300E_t4.yml \\
    --model sdt --spike-mode lif --workers 2 \\
    --TET --TET-lamb 0.5 \\
    --use-cv-loss --lambda-cv 0.01 --cv-weight 1.0 \\
    --output ./output_cv_tet


QUICK TEST (20 epochs)
──────────────────────────────────────────────────────────────────────────────
bash test_cv_quick.sh


FULL COMPARISON EXPERIMENTS
──────────────────────────────────────────────────────────────────────────────
bash test_cv_loss.sh


ANALYSIS & VISUALIZATION
──────────────────────────────────────────────────────────────────────────────
# Compare results
python compare_cv_experiments.py --output-dir ./output_cv_experiments

# Generate plots
python plot_cv_comparison.py --output-dir ./output_cv_experiments

# Check firing rates
python firing_num.py -c conf/cifar10/2_256_300E_t4.yml \\
    --resume ./output_cv_experiments/cv_loss_01_*/train/*/model_best.pth.tar \\
    --no-resume-opt

═════════════════════════════════════════════════════════════════════════════
"""

EXPECTED_RESULTS = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                    EXPECTED RESULTS & BENCHMARKS                           ║
╚═════════════════════════════════════════════════════════════════════════════╝

BASELINE MODEL (2-256, T=4, 300 epochs)
────────────────────────────────────────
  Final Top-1 Accuracy:     94.21%
  Best Top-1 Accuracy:      94.21% @ epoch 309
  Final Eval Loss:          0.2605
  Firing Rate Distribution: ~45% avg, relatively uniform

CV LOSS MODEL (λ=0.01, w=1.0)
────────────────────────────────────────
  Final Top-1 Accuracy:     94.05% ± 0.15%  ✓ Acceptable
  Best Top-1 Accuracy:      94.15% @ epoch 295
  Final Eval Loss:          0.2570
  Firing Rate CV:           1.2 → 1.5+       ✓ Increased
  Expected Improvement:     Sparse firing patterns
  Expected Energy Saving:   ~5-10% (hardware dependent)

PARAMETER COMBINATIONS - EXPECTED RESULTS
────────────────────────────────────────
  λ=0.005, w=1.0:  94.18% ± 0.05%  (minimal acc drop, soft regularization)
  λ=0.01, w=1.0:   94.05% ± 0.15%  (balanced) ⭐
  λ=0.02, w=1.0:   93.85% ± 0.30%  (strong, may be too much)
  λ=0.01, w=1.5:   93.95% ± 0.20%  (increased emphasis)

═════════════════════════════════════════════════════════════════════════════
✓ PASS CRITERIA:
  ✓ Accuracy within -0.5% of baseline
  ✓ CV value in firing rates increases
  ✓ Convergence speed similar or better

✗ FAIL CRITERIA:
  ✗ Accuracy drops > 1%
  ✗ CV doesn't change (regularization too weak)
  ✗ Training becomes unstable

═════════════════════════════════════════════════════════════════════════════
"""

TROUBLESHOOTING = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                        TROUBLESHOOTING                                     ║
╚═════════════════════════════════════════════════════════════════════════════╝

ISSUE: ModuleNotFoundError: No module named 'criterion'
──────────────────────────────────────────────────────────────────────────────
Location: train_one_epoch() calls criterion.combined_loss()

Solution 1: Ensure criterion.py is in the same directory
  ls -la criterion.py

Solution 2: Check Python path
  python -c "import criterion; print(criterion.__file__)"

Solution 3: Verify imports in criterion.py
  head -5 criterion.py
  # Should show: import torch, import torch.nn as nn


ISSUE: AttributeError: 'module' object has no attribute 'combined_loss'
──────────────────────────────────────────────────────────────────────────────
Cause: criterion.py doesn't have the new combined_loss function

Solution:
  1. Check criterion.py has both functions:
     grep "def firing_rate_cv_loss\|def combined_loss" criterion.py
  
  2. Verify the function is indented correctly (not inside another function)
  
  3. Restart Python: python -c "import criterion; print(dir(criterion))"


ISSUE: Training loss becomes NaN or Inf
──────────────────────────────────────────────────────────────────────────────
Possible causes:
  1. lambda_cv too large (CV loss dominates)
  2. Numerical instability in CV calculation

Solutions:
  1. Reduce lambda_cv: try 0.005 or 0.001
  2. Check for spikes with all zeros (no firing neurons)
  3. Reduce learning rate slightly
  4. Check if model outputs are in valid range


ISSUE: Accuracy drops sharply with --use-cv-loss
──────────────────────────────────────────────────────────────────────────────
Diagnosis:
  - λ_cv too large → reduce to 0.005
  - cv_weight too large → reduce to 0.5
  - Conflict with other regularizations → disable drop-path temporarily

Quick fix:
  python train.py ... --use-cv-loss --lambda-cv 0.005 --cv-weight 0.5 \\
      --drop-path 0.0


ISSUE: Output directory already exists
──────────────────────────────────────────────────────────────────────────────
Location: --output flag points to existing directory

Solution:
  # Option 1: Use different output name
  --output ./output_cv_attempt2
  
  # Option 2: Check and backup existing results
  ls -la ./output_cv_01
  mv ./output_cv_01 ./output_cv_01_backup


═════════════════════════════════════════════════════════════════════════════
For detailed help, see CV_LOSS_EXPERIMENTS.md
═════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        topic = sys.argv[1].lower()
        if topic in ['start', 'quick']:
            print(QUICK_START)
        elif topic in ['hyper', 'param', 'tuning']:
            print(HYPERPARAMETER_GUIDE)
        elif topic in ['cmd', 'command']:
            print(COMMAND_REFERENCE)
        elif topic in ['result', 'expected']:
            print(EXPECTED_RESULTS)
        elif topic in ['trouble', 'fix', 'error']:
            print(TROUBLESHOOTING)
        else:
            print("Available topics: start, hyper, cmd, result, trouble")
            print(f"\nUsage: python reference.py [start|hyper|cmd|result|trouble]\n")
    else:
        # Print all
        print(QUICK_START)
        print(HYPERPARAMETER_GUIDE)
        print(COMMAND_REFERENCE)
        print(EXPECTED_RESULTS)
        print(TROUBLESHOOTING)
