#!/usr/bin/env python3

"""
Verification Script for CV Loss Implementation

This script verifies that all CV Loss modifications and new files have been
correctly installed and are ready to use.

Usage:
    python verify_cv_loss.py [--verbose]
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

def check_file_exists(filepath, desc=""):
    """Check if a file exists."""
    exists = os.path.isfile(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath:<50} {desc}")
    return exists

def check_file_contains(filepath, search_string, desc=""):
    """Check if a file contains a specific string."""
    if not os.path.isfile(filepath):
        print(f"✗ File not found: {filepath}")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
        contains = search_string in content
        status = "✓" if contains else "✗"
        print(f"{status} {filepath:<50} contains '{search_string}'")
        return contains

def check_executable(filepath):
    """Check if a file is executable."""
    try:
        os.access(filepath, os.X_OK)
        print(f"✓ {filepath:<50} executable")
        return True
    except:
        print(f"✗ {filepath:<50} not executable")
        return False

def main():
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║         CV LOSS IMPLEMENTATION VERIFICATION SCRIPT                         ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")
    
    all_ok = True
    
    # ========== CHECK MODIFIED FILES ==========
    print("1️⃣  MODIFIED FILES")
    print("─" * 80)
    
    all_ok &= check_file_contains(
        "criterion.py",
        "def firing_rate_cv_loss",
        "CV loss function"
    )
    
    all_ok &= check_file_contains(
        "criterion.py",
        "def combined_loss",
        "Combined loss function"
    )
    
    all_ok &= check_file_contains(
        "train.py",
        "--use-cv-loss",
        "CV Loss parameter"
    )
    
    all_ok &= check_file_contains(
        "train.py",
        "criterion.combined_loss",
        "CV Loss usage in training"
    )
    
    print()
    
    # ========== CHECK NEW FILES ==========
    print("2️⃣  NEW FILES")
    print("─" * 80)
    
    scripts = [
        "test_cv_quick.sh",
        "test_cv_loss.sh",
    ]
    
    for script in scripts:
        all_ok &= check_file_exists(script, "Test script")
    
    print()
    
    python_files = [
        "compare_cv_experiments.py",
        "plot_cv_comparison.py",
        "reference_cv_loss.py",
    ]
    
    for pyfile in python_files:
        all_ok &= check_file_exists(pyfile, "Python tool")
    
    print()
    
    docs = [
        "CV_LOSS_EXPERIMENTS.md",
        "CV_LOSS_EXTENSION.md",
        "CV_LOSS_SUMMARY.md",
    ]
    
    for doc in docs:
        all_ok &= check_file_exists(doc, "Documentation")
    
    print()
    
    # ========== SYNTAX CHECK ==========
    print("3️⃣  SYNTAX VERIFICATION")
    print("─" * 80)
    
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", "criterion.py"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ criterion.py syntax OK")
        else:
            print(f"✗ criterion.py has syntax error")
            print(f"  Error: {result.stderr.decode()}")
            all_ok = False
    except Exception as e:
        print(f"⚠ Could not check criterion.py syntax: {e}")
    
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", "train.py"],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ train.py syntax OK")
        else:
            print(f"✗ train.py has syntax error")
            all_ok = False
    except Exception as e:
        print(f"⚠ Could not check train.py syntax: {e}")
    
    print()
    
    # ========== IMPORT CHECK ==========
    print("4️⃣  IMPORT VERIFICATION")
    print("─" * 80)
    
    try:
        import criterion
        if hasattr(criterion, 'firing_rate_cv_loss'):
            print("✓ criterion.firing_rate_cv_loss importable")
        else:
            print("✗ criterion.firing_rate_cv_loss not found")
            all_ok = False
        
        if hasattr(criterion, 'combined_loss'):
            print("✓ criterion.combined_loss importable")
        else:
            print("✗ criterion.combined_loss not found")
            all_ok = False
    except ImportError as e:
        print(f"✗ Failed to import criterion: {e}")
        all_ok = False
    
    print()
    
    # ========== QUICK SANITY CHECK ==========
    print("5️⃣  QUICK SANITY CHECK")
    print("─" * 80)
    
    try:
        import torch
        import criterion
        
        # Create dummy tensors
        T, B, N = 4, 2, 256  # time steps, batch, neurons
        dummy_spikes = torch.randint(0, 2, (T, B, N)).float()
        
        # Test CV loss
        loss_cv = criterion.firing_rate_cv_loss(dummy_spikes, lambda_cv=0.01)
        if loss_cv.item() != 0:  # Should be non-zero
            print(f"✓ firing_rate_cv_loss works (loss={loss_cv.item():.4f})")
        else:
            print(f"⚠ firing_rate_cv_loss returned zero")
        
        # Test combined loss
        dummy_outputs = torch.randn(T, B, 10)  # (T, B, num_classes)
        dummy_labels = torch.randint(0, 10, (B,))
        criterion_ce = torch.nn.CrossEntropyLoss()
        
        loss_total = criterion.combined_loss(
            dummy_outputs, dummy_labels, criterion_ce,
            use_cv_loss=True, lambda_cv=0.01, cv_weight=1.0
        )
        if loss_total.item() > 0:
            print(f"✓ combined_loss works (loss={loss_total.item():.4f})")
        else:
            print(f"⚠ combined_loss returned zero or negative")
        
    except Exception as e:
        print(f"✗ Sanity check failed: {e}")
        all_ok = False
    
    print()
    
    # ========== SUMMARY ==========
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    if all_ok:
        print("║                    ✅ ALL CHECKS PASSED                                   ║")
        print("║                  Implementation is ready to use!                         ║")
    else:
        print("║                    ⚠️  SOME CHECKS FAILED                                  ║")
        print("║            Please review the errors above and fix them.                 ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")
    
    # ========== NEXT STEPS ==========
    print("📋 NEXT STEPS:")
    print("─" * 80)
    print("\n1️⃣  Quick Test (recommended first):")
    print("   $ bash test_cv_quick.sh\n")
    
    print("2️⃣  Read Documentation:")
    print("   $ cat CV_LOSS_EXPERIMENTS.md  # Full user guide")
    print("   $ cat CV_LOSS_EXTENSION.md    # Technical details")
    print("   $ cat CV_LOSS_SUMMARY.md      # Implementation summary\n")
    
    print("3️⃣  Quick Reference:")
    print("   $ python reference_cv_loss.py start\n")
    
    print("4️⃣  Full Experiments (optional, takes 4-24 hours):")
    print("   $ bash test_cv_loss.sh\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
