#!/usr/bin/env python3

"""
Quick Status Check for CV Loss Training

Provides real-time status of ongoing CV Loss training experiments.
"""

import os
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def get_latest_log_entries(log_file, num_lines=5):
    """Get last N lines from log file."""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            return lines[-num_lines:] if lines else []
    except FileNotFoundError:
        return []


def extract_epoch_info(log_line):
    """Extract epoch and batch info from training log line."""
    if "Train:" not in log_line:
        return None
    
    try:
        import re
        # Pattern: "Train: 0 [   0/781 (  0%)]"
        match = re.search(r"Train: (\d+) \[.*?/(\d+) \((.*?)%\)\]", log_line)
        if match:
            epoch = match.group(1)
            total = match.group(2)
            percent = match.group(3).strip()
            return {"epoch": epoch, "total": total, "percent": percent}
    except:
        pass
    
    return None


def extract_loss(log_line):
    """Extract loss value from log line."""
    try:
        import re
        match = re.search(r"Loss:\s+([\d.]+)", log_line)
        if match:
            return float(match.group(1))
    except:
        pass
    return None


def check_training_status(base_dir, test_type="quick"):
    """Check status of training experiments."""
    
    status = defaultdict(lambda: {"status": "unknown", "epoch": "N/A", "loss": "N/A", "complete": False})
    
    if test_type == "quick":
        search_patterns = [
            f"{base_dir}/output_cv_quick_test/*/*/*.log",  # legacy quick tests
            f"{base_dir}/output_cv_tests/*/*.log",          # v2 quick tests
        ]
    else:
        search_patterns = [f"{base_dir}/output_cv_experiments/*/*/*.log"]

    log_files = []
    for pattern in search_patterns:
        log_files.extend(glob.glob(pattern))
    
    for log_file in sorted(log_files, reverse=True)[:8]:  # Check last 8 runs
        exp_dir = os.path.dirname(log_file)
        exp_name = Path(exp_dir).parent.name
        
        lines = get_latest_log_entries(log_file, 10)
        
        if not lines:
            status[exp_name]["status"] = "not started"
            continue
        
        # Find latest training line
        for line in reversed(lines):
            epoch_info = extract_epoch_info(line)
            if epoch_info:
                status[exp_name]["status"] = "training"
                status[exp_name]["epoch"] = (
                    f"Epoch {epoch_info['epoch']} (batch progress: {epoch_info['percent']}%)"
                )
                loss = extract_loss(line)
                if loss:
                    status[exp_name]["loss"] = f"{loss:.4f}"
                break
        
        # Check if complete
        full_content = "".join(lines)
        if "Validation complete" in full_content or "Training complete" in full_content:
            status[exp_name]["status"] = "completed"
            status[exp_name]["complete"] = True
    
    return status


def main():
    import sys
    
    test_type = sys.argv[1] if len(sys.argv) > 1 else "quick"
    base_dir = "/Users/mia469/Library/Mobile Documents/com~apple~CloudDocs/GitHub/SSNN/Spike-Driven-Transformer"
    
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  CV LOSS TRAINING STATUS - {test_type.upper()} TESTS" + " " * (60 - len(test_type)))
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print("")
    
    status = check_training_status(base_dir, test_type)
    
    if not status:
        print("ℹ️  No training runs found yet")
        if test_type == "quick":
            print(f"   Expected locations:")
            print(f"   - {base_dir}/output_cv_quick_test/")
            print(f"   - {base_dir}/output_cv_tests/")
        else:
            print(f"   Expected location: {base_dir}/output_cv_experiments/")
        print("")
        return
    
    # Display summary
    completed_count = sum(1 for s in status.values() if s["complete"])
    training_count = sum(1 for s in status.values() if s["status"] == "training")
    
    print(f"📊 Summary: {completed_count} completed | {training_count} training | {len(status)} total")
    print("")
    print("Experiment Status:")
    print("─" * 80)
    
    for exp_name in sorted(status.keys(), reverse=True):
        info = status[exp_name]
        status_symbol = "✅" if info["complete"] else "⏳" if info["status"] == "training" else "⏸️"
        
        print(f"{status_symbol} {exp_name:<40} {info['status']:<12}")
        print(f"   {info['epoch']:<40} Loss: {info['loss']}")
        print("")
    
    print("─" * 80)
    
    # Provide usage hints
    if test_type == "quick":
        print("\n💡 Quick Tests Status:")
        print("   - Should complete in ~10-15 minutes on CPU")
        print("   - Each test runs 20 epochs")
        print("   - Batch progress 100% means current epoch finished, not full experiment")
        print("   - Monitor with: tail -f output_cv_tests/*/test_*.log")
    else:
        print("\n💡 Full Experiments Status:")
        print("   - May take 4-24 hours depending on hardware")
        print("   - Each experiment runs 300 epochs")
        print("   - Monitor with: tail -f output_cv_experiments/*/sdt-*/sdt.log")
    
    print(f"\n⏰ Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")


if __name__ == "__main__":
    main()
