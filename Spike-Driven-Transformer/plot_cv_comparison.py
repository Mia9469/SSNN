#!/usr/bin/env python3

"""
Generate Plots Comparing CV Loss Experiments

This script creates visualization plots for comparing different CV loss experiments.

Usage:
    python plot_cv_comparison.py [--output-dir ./output_cv_experiments]
"""

import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_summary_csv(base_path):
    """Recursively find summary.csv files in experiment directory."""
    pattern = os.path.join(base_path, "**", "summary.csv")
    files = glob.glob(pattern, recursive=True)
    return files

def load_all_experiments(output_dir):
    """Load all experiment data."""
    experiments = {}
    
    exp_dirs = [d for d in glob.glob(os.path.join(output_dir, "*")) 
                if os.path.isdir(d) and any(char.isdigit() for char in d)]
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        csv_files = find_summary_csv(exp_dir)
        
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                experiments[exp_name] = df
                print(f"✓ Loaded: {exp_name} ({len(df)} epochs)")
            except Exception as e:
                print(f"✗ Failed to load {exp_name}: {e}")
    
    return experiments

def create_accuracy_plot(experiments, output_dir):
    """Create training curves comparison plot."""
    
    plt.figure(figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for (exp_name, df), color in zip(experiments.items(), colors):
        label = exp_name.replace('_', ' ').replace('-', ' ')
        plt.plot(df['epoch'], df['eval_top1'], 
                label=label, linewidth=2, marker='o', 
                markersize=3, color=color, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy Comparison: Baseline vs CV Loss Variants', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_loss_plot(experiments, output_dir):
    """Create training loss comparison plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Training loss
    for (exp_name, df), color in zip(experiments.items(), colors):
        label = exp_name.replace('_', ' ').replace('-', ' ')
        ax1.plot(df['epoch'], df['train_loss'], 
                label=label, linewidth=2, marker='o', 
                markersize=3, color=color, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Evaluation loss
    for (exp_name, df), color in zip(experiments.items(), colors):
        label = exp_name.replace('_', ' ').replace('-', ' ')
        ax2.plot(df['epoch'], df['eval_loss'], 
                label=label, linewidth=2, marker='o', 
                markersize=3, color=color, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Evaluation Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Evaluation Loss Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "loss_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_final_metrics_bar_plot(experiments, output_dir):
    """Create bar plot of final metrics."""
    
    names = []
    final_acc = []
    best_acc = []
    best_epochs = []
    
    for exp_name, df in experiments.items():
        names.append(exp_name.replace('_', ' ').replace('-', ' '))
        final_acc.append(df['eval_top1'].iloc[-1])
        best_idx = df['eval_top1'].idxmax()
        best_acc.append(df['eval_top1'].iloc[best_idx])
        best_epochs.append(df['epoch'].iloc[best_idx])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    x = np.arange(len(names))
    width = 0.35
    
    # Final vs Best accuracy
    ax1.bar(x - width/2, final_acc, width, label='Final Accuracy', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, best_acc, width, label='Best Accuracy', alpha=0.8, color='orange')
    
    ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Final vs Best Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Epoch at best accuracy
    colors_bar = plt.cm.Set3(np.linspace(0, 1, len(names)))
    ax2.bar(names, best_epochs, alpha=0.8, color=colors_bar)
    ax2.set_ylabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_title('Epoch Achieving Best Accuracy', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "final_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_convergence_speed_plot(experiments, output_dir):
    """Plot convergence speed (epochs to reach certain accuracy thresholds)."""
    
    thresholds = [90, 92, 93, 94, 95]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(thresholds))
    width = 0.15
    
    for i, (exp_name, df) in enumerate(experiments.items()):
        epochs_to_threshold = []
        
        for threshold in thresholds:
            reaching = df[df['eval_top1'] >= threshold]
            if len(reaching) > 0:
                epochs_to_threshold.append(reaching['epoch'].iloc[0])
            else:
                epochs_to_threshold.append(np.nan)
        
        offset = (i - len(experiments)/2 + 0.5) * width
        label = exp_name.replace('_', ' ').replace('-', ' ')
        
        ax.bar(x + offset, epochs_to_threshold, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Accuracy Threshold (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Epochs Required', fontsize=11, fontweight='bold')
    ax.set_title('Convergence Speed: Epochs to Reach Accuracy Thresholds', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}%' for t in thresholds])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "convergence_speed.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Generate Plots for CV Loss Experiments"
    )
    parser.add_argument(
        "--output-dir",
        default="./output_cv_experiments",
        help="Base output directory containing experiment results"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=['accuracy', 'loss', 'metrics', 'convergence', 'all'],
        default=['all'],
        help="Which plots to generate (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load experiments
    print(f"\nScanning: {args.output_dir}\n")
    experiments = load_all_experiments(args.output_dir)
    
    if not experiments:
        print(f"✗ No experiments found in {args.output_dir}")
        return
    
    print(f"\n✓ Loaded {len(experiments)} experiments\n")
    
    # Create plots
    plot_types = args.plots if 'all' not in args.plots else ['accuracy', 'loss', 'metrics', 'convergence']
    
    print("Generating plots...")
    
    if 'accuracy' in plot_types:
        create_accuracy_plot(experiments, args.output_dir)
    
    if 'loss' in plot_types:
        create_loss_plot(experiments, args.output_dir)
    
    if 'metrics' in plot_types:
        create_final_metrics_bar_plot(experiments, args.output_dir)
    
    if 'convergence' in plot_types:
        create_convergence_speed_plot(experiments, args.output_dir)
    
    print("\n" + "="*80)
    print("Plot generation complete!")
    print("="*80)
    print("\nGenerated files:")
    plot_files = glob.glob(os.path.join(args.output_dir, "*.png"))
    for f in sorted(plot_files):
        print(f"  - {os.path.basename(f)}")
    print()

if __name__ == "__main__":
    main()
