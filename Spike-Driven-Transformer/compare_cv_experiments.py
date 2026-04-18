#!/usr/bin/env python3

"""
Compare CV Loss Experiments Results

This script analyzes and compares results from different CV loss experiments,
generating comparison tables and plots.

Usage:
    python compare_cv_experiments.py [--output-dir ./output_cv_experiments]
"""

import os
import glob
import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime

def find_summary_csv(base_path):
    """Recursively find summary.csv files in experiment directory."""
    pattern = os.path.join(base_path, "**", "summary.csv")
    files = glob.glob(pattern, recursive=True)
    return files

def load_experiment_results(output_dir):
    """Load all experiment results from output directory."""
    experiments = {}
    
    # Find all experiment subdirectories
    exp_dirs = [d for d in glob.glob(os.path.join(output_dir, "*")) 
                if os.path.isdir(d) and any(char.isdigit() for char in d)]
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        
        # Find summary.csv
        csv_files = find_summary_csv(exp_dir)
        
        if csv_files:
            csv_path = csv_files[0]
            try:
                df = pd.read_csv(csv_path)
                experiments[exp_name] = {
                    'path': exp_dir,
                    'csv': csv_path,
                    'data': df
                }
                print(f"✓ Loaded: {exp_name}")
            except Exception as e:
                print(f"✗ Failed to load {exp_name}: {e}")
        else:
            print(f"⚠ No summary.csv found in {exp_name}")
    
    return experiments

def extract_key_metrics(experiments):
    """Extract key metrics from each experiment."""
    metrics = {}
    
    for exp_name, exp_data in experiments.items():
        df = exp_data['data']
        
        if len(df) == 0:
            continue
        
        # Get last epoch results
        last_epoch = df.iloc[-1]
        # Get best results
        best_idx = df['eval_top1'].idxmax()
        best_row = df.iloc[best_idx]
        
        metrics[exp_name] = {
            'final_top1': last_epoch['eval_top1'],
            'final_top5': last_epoch['eval_top5'],
            'final_train_loss': last_epoch['train_loss'],
            'final_eval_loss': last_epoch['eval_loss'],
            'best_top1': best_row['eval_top1'],
            'best_top1_epoch': best_row['epoch'],
            'best_top5': best_row['eval_top5'],
            'num_epochs': len(df),
        }
    
    return metrics

def print_comparison_table(metrics):
    """Print comparison table of key metrics."""
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*100 + "\n")
    
    # Create comparison dataframe
    comparison_data = []
    for exp_name, exp_metrics in metrics.items():
        comparison_data.append({
            'Experiment': exp_name.replace('_', ' '),
            'Final Top-1 (%)': f"{exp_metrics['final_top1']:.2f}",
            'Best Top-1 (%)': f"{exp_metrics['best_top1']:.2f}",
            '@Epoch': int(exp_metrics['best_top1_epoch']),
            'Final Top-5 (%)': f"{exp_metrics['final_top5']:.2f}",
            'Final Train Loss': f"{exp_metrics['final_train_loss']:.4f}",
            'Final Eval Loss': f"{exp_metrics['final_eval_loss']:.4f}",
            'Epochs': int(exp_metrics['num_epochs']),
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("\n" + "="*100 + "\n")
    
    return df_comparison

def analyze_cv_differences(metrics):
    """Analyze differences between baseline and CV loss variants."""
    
    print("="*100)
    print("CV LOSS IMPACT ANALYSIS")
    print("="*100 + "\n")
    
    # Find baseline
    baseline_names = [k for k in metrics.keys() if 'baseline' in k.lower()]
    if baseline_names:
        baseline_name = baseline_names[0]
        baseline_best = metrics[baseline_name]['best_top1']
        
        print(f"Baseline: {baseline_name}")
        print(f"Baseline Best Top-1: {baseline_best:.2f}%\n")
        
        print("Comparison with CV Loss Variants:")
        print("-" * 100)
        print(f"{'CV Variant':<40} {'Best Top-1':<15} {'Δ from Baseline':<20} {'Δ %':<10}")
        print("-" * 100)
        
        for exp_name, exp_metrics in metrics.items():
            if 'baseline' not in exp_name.lower():
                best_acc = exp_metrics['best_top1']
                delta = best_acc - baseline_best
                delta_pct = (delta / baseline_best) * 100
                
                status = "✓" if delta >= -0.5 else "✗"
                print(f"{exp_name:<40} {best_acc:<15.2f} {delta:+.2f}%{'':<14} {delta_pct:+.2f}%      {status}")
        
        print("-" * 100)
        print("\nInterpretation:")
        print("  ✓ Δ >= -0.5%: Acceptable (CV loss maintains or improves accuracy)")
        print("  ✗ Δ < -0.5%:  Significant drop (may need parameter tuning)")
        print("\n" + "="*100 + "\n")

def save_results_summary(metrics, output_file):
    """Save detailed results to file."""
    
    with open(output_file, 'w') as f:
        f.write("CV LOSS EXPERIMENTS - DETAILED RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for exp_name, exp_metrics in sorted(metrics.items()):
            f.write(f"\n{exp_name}\n")
            f.write("-"*80 + "\n")
            for key, value in sorted(exp_metrics.items()):
                if isinstance(value, float):
                    f.write(f"  {key:<25}: {value:.4f}\n")
                else:
                    f.write(f"  {key:<25}: {value}\n")
    
    print(f"Results summary saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare CV Loss Experiments Results"
    )
    parser.add_argument(
        "--output-dir",
        default="./output_cv_experiments",
        help="Base output directory containing experiment results (default: ./output_cv_experiments)"
    )
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save detailed summary to file"
    )
    
    args = parser.parse_args()
    
    # Load experiments
    print(f"\nScanning directory: {args.output_dir}\n")
    print("Loading experiments...")
    experiments = load_experiment_results(args.output_dir)
    
    if not experiments:
        print(f"\n✗ No experiments found in {args.output_dir}")
        print("  Make sure to run: bash test_cv_loss.sh")
        return
    
    print(f"\n✓ Loaded {len(experiments)} experiments\n")
    
    # Extract metrics
    metrics = extract_key_metrics(experiments)
    
    # Print comparison
    df_comparison = print_comparison_table(metrics)
    
    # Analyze differences
    analyze_cv_differences(metrics)
    
    # Save summary if requested
    if args.save_summary:
        summary_file = os.path.join(args.output_dir, "comparison_summary.txt")
        save_results_summary(metrics, summary_file)
    
    # Print recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 100)
    print("\n1. ACCURACY COMPARISON:")
    print("   - Check if CV loss variants achieve comparable accuracy (within ±0.5%)")
    print("   - If accuracy improves, CV loss pattern regularization is beneficial")
    print("   - If accuracy drops >1%, reduce lambda_cv or cv_weight\n")
    
    print("2. CONVERGENCE SPEED:")
    print("   - Compare epoch numbers where best accuracy is achieved")
    print("   - CV loss may help with faster convergence\n")
    
    print("3. FIRING RATE DISTRIBUTION:")
    print("   - Run firing_num.py on best models to compare firing patterns")
    print("   - CV loss should show higher CV in firing rates\n")
    
    print("4. NEXT STEPS:")
    print("   - python compare_cv_experiments.py --save-summary")
    print("   - python plot_cv_comparison.py  # To generate plots")
    print("   - Check firing rates: python firing_num.py --resume [model_path]\n")

if __name__ == "__main__":
    main()
