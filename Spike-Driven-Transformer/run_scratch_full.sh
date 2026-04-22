#!/bin/bash
# Desktop / GPU machine runner for full from-scratch training.
#
# Usage:
#   chmod +x run_scratch_full.sh
#   ./run_scratch_full.sh
#
# Assumes: Python 3.9/3.10 venv active, CUDA GPU available.

set -euo pipefail
cd "$(dirname "$0")"

echo "=== Checking environment ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU/MPS\"}')"
python3 -c "import timm, spikingjelly; print(f'timm: {timm.__version__}')"

echo ""
echo "=== Overriding config for full training ==="
python3 - <<'PYEOF'
import re
path = "scratch_train_compare.py"
with open(path) as f:
    content = f.read()

# Full-training overrides
replacements = [
    (r"N_EPOCHS\s*=\s*\d+", "N_EPOCHS        = 300"),
    (r"N_TRAIN_SAMPLES\s*=\s*\d+", "N_TRAIN_SAMPLES = 50000"),
    (r"EVAL_EVERY\s*=\s*\d+", "EVAL_EVERY      = 10"),
    (r"WARMUP_EP\s*=\s*\d+", "WARMUP_EP       = 20"),
]
for pat, rep in replacements:
    content = re.sub(pat, rep, content)

with open(path, "w") as f:
    f.write(content)
print("Config updated in scratch_train_compare.py")
PYEOF

echo ""
echo "=== Launching training ==="
rm -f scratch_compare_results.json   # force rerun

LOG="scratch_compare_$(date +%Y%m%d-%H%M%S).log"
nohup python3 -u scratch_train_compare.py > "$LOG" 2>&1 &
PID=$!
echo $PID > scratch.pid
echo "Started  PID=$PID  Log=$LOG"
echo ""
echo "Watch progress:  tail -f $LOG"
echo "Stop training:   kill $PID"
