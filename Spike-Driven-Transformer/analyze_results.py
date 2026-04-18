#!/usr/bin/env python3

"""
CV Loss Experiment Results Analysis

分析CV Loss快速测试的结果，对比不同配置的性能。
"""

import os
import pandas as pd
import glob
from pathlib import Path
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')


def load_summary_csv(csv_path):
    """Load summary.csv and return as dataframe."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"ERROR loading {csv_path}: {e}")
        return None


def analyze_experiment(exp_dir):
    """Analyze a single experiment directory."""
    
    # Find summary.csv
    csv_files = glob.glob(f"{exp_dir}/**/summary.csv", recursive=True)
    
    if not csv_files:
        return None
    
    csv_path = csv_files[0]
    df = load_summary_csv(csv_path)
    
    if df is None or len(df) == 0:
        return None
    
    exp_name = Path(exp_dir).name
    
    # Get final results
    last_row = df.iloc[-1]
    best_top1_idx = df['eval_top1'].idxmax()
    best_row = df.iloc[best_top1_idx]
    
    return {
        'name': exp_name,
        'path': exp_dir,
        'epochs_completed': len(df) - 1,  # -1 for header
        'final_epoch': int(last_row['epoch']),
        'final_train_loss': float(last_row['train_loss']),
        'final_eval_loss': float(last_row['eval_loss']),
        'final_top1': float(last_row['eval_top1']),
        'final_top5': float(last_row['eval_top5']),
        'best_top1': float(best_row['eval_top1']),
        'best_top1_epoch': int(best_row['epoch']),
        'best_top5': float(best_row['eval_top5']),
        'dataframe': df
    }


def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║            CV LOSS EXPERIMENT RESULTS ANALYSIS                             ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")
    
    # Find experiments
    quick_test_dir = "./output_cv_quick_test"
    full_exp_dir = "./output_cv_experiments"
    
    experiments = {}
    
    # Scan quick tests
    if os.path.exists(quick_test_dir):
        print(f"📁 Scanning quick tests: {quick_test_dir}\n")
        
        quick_dirs = glob.glob(f"{quick_test_dir}/*")
        for exp_dir in sorted(quick_dirs, reverse=True):
            result = analyze_experiment(exp_dir)
            if result:
                exp_type = "baseline" if "baseline" in result['name'] else "cv-loss"
                if exp_type not in experiments:
                    experiments[exp_type] = []
                experiments[exp_type].append(result)
    
    # Scan full experiments
    if os.path.exists(full_exp_dir):
        print(f"📁 Scanning full experiments: {full_exp_dir}\n")
        
        full_dirs = glob.glob(f"{full_exp_dir}/*")
        for exp_dir in sorted(full_dirs, reverse=True):
            result = analyze_experiment(exp_dir)
            if result:
                exp_type = "baseline" if "baseline" in result['name'] else "cv-loss"
                if exp_type not in experiments:
                    experiments[exp_type] = []
                experiments[exp_type].append(result)
    
    if not experiments:
        print("❌ No experiment results found!")
        return
    
    # Display summary
    print("=" * 80)
    print(" EXPERIMENT SUMMARY")
    print("=" * 80 + "\n")
    
    all_results = []
    for exp_type in sorted(experiments.keys()):
        for result in experiments[exp_type]:
            all_results.append([
                result['name'][:30],
                f"Epochs {result['final_epoch']}",
                f"{result['final_train_loss']:.4f}",
                f"{result['final_eval_loss']:.4f}",
                f"{result['best_top1']:.2f}%",
                f"{result['best_top5']:.2f}%",
            ])
    
    headers = ["Experiment", "Progress", "Train Loss", "Eval Loss", "Best Top-1", "Best Top-5"]
    print(tabulate(all_results, headers=headers, tablefmt="grid"))
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print(" DETAILED RESULTS")
    print("=" * 80 + "\n")
    
    for exp_type in sorted(experiments.keys()):
        print(f"\n{'='*80}")
        print(f" {exp_type.upper()} EXPERIMENTS")
        print(f"{'='*80}\n")
        
        for result in experiments[exp_type]:
            print(f"📊 {result['name']}")
            print(f"  Path: {result['path']}")
            print(f"  Progress: Epoch {result['final_epoch']} completed")
            print(f"  ")
            print(f"  Final Metrics:")
            print(f"    - Train Loss: {result['final_train_loss']:.6f}")
            print(f"    - Eval Loss:  {result['final_eval_loss']:.6f}")
            print(f"    - Top-1 Acc:  {result['final_top1']:.2f}%")
            print(f"    - Top-5 Acc:  {result['final_top5']:.2f}%")
            print(f"  ")
            print(f"  Best Metrics:")
            print(f"    - Best Top-1: {result['best_top1']:.2f}% (@ Epoch {result['best_top1_epoch']})")
            print(f"    - Best Top-5: {result['best_top5']:.2f}%")
            print("")
    
    # Comparative analysis
    if "baseline" in experiments and "cv-loss" in experiments:
        print("\n" + "=" * 80)
        print(" COMPARATIVE ANALYSIS: BASELINE vs CV LOSS")
        print("=" * 80 + "\n")
        
        baseline = experiments["baseline"][0]
        cv_loss = experiments["cv-loss"][0]
        
        diff_top1 = cv_loss['best_top1'] - baseline['best_top1']
        diff_top5 = cv_loss['best_top5'] - baseline['best_top5']
        diff_train_loss = cv_loss['final_train_loss'] - baseline['final_train_loss']
        diff_eval_loss = cv_loss['final_eval_loss'] - baseline['final_eval_loss']
        
        comparison_data = [
            ["Best Top-1 (%)", f"{baseline['best_top1']:.2f}", f"{cv_loss['best_top1']:.2f}", 
             f"{diff_top1:+.2f}" + (" ✅" if diff_top1 >= -0.5 else " ❌")],
            ["Best Top-5 (%)", f"{baseline['best_top5']:.2f}", f"{cv_loss['best_top5']:.2f}", 
             f"{diff_top5:+.2f}" + (" ✅" if diff_top5 >= -0.5 else " ❌")],
            ["Final Train Loss", f"{baseline['final_train_loss']:.4f}", f"{cv_loss['final_train_loss']:.4f}", 
             f"{diff_train_loss:+.4f}"],
            ["Final Eval Loss", f"{baseline['final_eval_loss']:.4f}", f"{cv_loss['final_eval_loss']:.4f}", 
             f"{diff_eval_loss:+.4f}" + (" ✅" if diff_eval_loss < 0 else " ⚠️")],
            ["Convergence Epoch", f"{baseline['best_top1_epoch']}", f"{cv_loss['best_top1_epoch']}", 
             f"{cv_loss['best_top1_epoch'] - baseline['best_top1_epoch']:+d}"],
        ]
        
        comp_headers = ["Metric", "Baseline", "CV Loss", "Difference"]
        print(tabulate(comparison_data, headers=comp_headers, tablefmt="grid"))
        
        print("\n📈 INTERPRETATION:")
        print("")
        print(f"1. Accuracy Impact:")
        if diff_top1 >= -0.5:
            print(f"   ✅ CV Loss maintains acceptable accuracy ({diff_top1:+.2f}% difference)")
        else:
            print(f"   ❌ CV Loss causes significant accuracy drop ({diff_top1:+.2f}%)")
        
        print(f"\n2. Convergence:")
        if cv_loss['best_top1_epoch'] <= baseline['best_top1_epoch']:
            print(f"   ✅ CV Loss converges faster (@ epoch {cv_loss['best_top1_epoch']} vs {baseline['best_top1_epoch']})")
        else:
            print(f"   ⚠️  CV Loss takes longer to converge (@ epoch {cv_loss['best_top1_epoch']} vs {baseline['best_top1_epoch']})")
        
        print(f"\n3. Regularization Effect:")
        if diff_eval_loss < 0:
            print(f"   ✅ CV Loss improves generalization (eval loss reduced by {-diff_eval_loss:.4f})")
        else:
            print(f"   ⚠️  CV Loss slightly increases eval loss ({diff_eval_loss:+.4f})")
        
        print(f"\n4. Overall Assessment:")
        if diff_top1 >= -0.5 and diff_eval_loss < 0:
            print("   🎯 CV Loss is BENEFICIAL - maintains accuracy with better generalization")
        elif diff_top1 >= -0.5:
            print("   ✅ CV Loss is ACCEPTABLE - maintains accuracy levels")
        else:
            print("   ⚠️  CV Loss needs tuning - accuracy drop detected")
    
    else:
        print("\n⚠️  Need both Baseline and CV Loss experiments for comparison")
        if "baseline" in experiments:
            print(f"   - Found {len(experiments['baseline'])} baseline experiment(s)")
        if "cv-loss" in experiments:
            print(f"   - Found {len(experiments['cv-loss'])} CV Loss experiment(s)")
    
    print("\n" + "=" * 80)
    print("\n💡 NEXT STEPS:")
    print("   1. If results are good, run full experiments: bash test_cv_loss.sh")
    print("   2. If accuracy drops, adjust --lambda-cv parameter (try 0.005 or 0.02)")
    print("   3. Run visualization: python plot_cv_comparison.py")
    print("\n")


if __name__ == "__main__":
    main()
