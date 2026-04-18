#!/bin/bash

# ============================================================================
# CV Loss Training Monitor
# ============================================================================
# This script monitors the progress of CV loss training experiments
# 
# Usage:
#   ./monitor_cv_training.sh                # Monitor all recent tests
#   ./monitor_cv_training.sh quick          # Monitor quick tests only
#   ./monitor_cv_training.sh full           # Monitor full experimental suite
# ============================================================================

MODE="${1:-all}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║          CV LOSS TRAINING MONITOR                                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Find latest training runs
echo "📊 Latest Training Runs:"
echo "─" * 80
echo ""

if [[ "$MODE" == "quick" || "$MODE" == "all" ]]; then
    echo "🏃 Quick Tests (20 epochs):"
    echo ""
    
    QUICK_DIRS=$(find ./output_cv_quick_test -maxdepth 2 -name "sdt.log" 2>/dev/null | sort -r | head -4)
    
    if [ -z "$QUICK_DIRS" ]; then
        echo "  No quick test results found"
    else
        for LOG_FILE in $QUICK_DIRS; do
            DIR=$(dirname "$LOG_FILE")
            BASENAME=$(basename "$DIR")
            PARENT=$(basename "$(dirname "$DIR")")
            
            echo "  Directory: $PARENT"
            echo "  ─────────────────────────────────────────"
            
            # Count completed batches in current epoch
            LAST_BATCH=$(tail -20 "$LOG_FILE" | grep "Train:" | tail -1)
            if [ ! -z "$LAST_BATCH" ]; then
                echo "  Status: $(echo "$LAST_BATCH" | sed 's/^.*Train:/Train:/' | head -c 80)"
            fi
            
            # Show loss trend
            LOSS_TREND=$(tail -5 "$LOG_FILE" | grep "Loss:" | tail -1 | grep -oE "Loss: [0-9.]+" | head -1)
            if [ ! -z "$LOSS_TREND" ]; then
                echo "  $LOSS_TREND"
            fi
            
            # Show last timestamp
            LAST_TIME=$(tail -1 "$LOG_FILE" | grep -oE "^[0-9-]+ [0-9:,]+")
            if [ ! -z "$LAST_TIME" ]; then
                echo "  Last update: $LAST_TIME"
            fi
            
            echo ""
        done
    fi
fi

if [[ "$MODE" == "full" || "$MODE" == "all" ]]; then
    echo ""
    echo "🔬 Full Experiments (300 epochs):"
    echo ""
    
    FULL_DIRS=$(find ./output_cv_experiments -maxdepth 2 -name "sdt.log" 2>/dev/null | sort -r | head -4)
    
    if [ -z "$FULL_DIRS" ]; then
        echo "  No full experiment results found yet"
    else
        for LOG_FILE in $FULL_DIRS; do
            DIR=$(dirname "$LOG_FILE")
            BASENAME=$(basename "$DIR")
            PARENT=$(basename "$(dirname "$DIR")")
            
            echo "  Experiment: $PARENT"
            echo "  ─────────────────────────────────────────"
            
            # Show current epoch and progress
            EPOCH_LINE=$(tail -50 "$LOG_FILE" | grep "Train:" | tail -1)
            if [ ! -z "$EPOCH_LINE" ]; then
                EPOCH=$(echo "$EPOCH_LINE" | grep -oE "^[0-9]+" | head -1)
                PROGRESS=$(echo "$EPOCH_LINE" | grep -oE "\([^)]*%\)" | head -1)
                echo "  Epoch/Progress: $EPOCH $PROGRESS"
                echo "  Latest: $(echo "$EPOCH_LINE" | sed 's/^.*Loss:/Loss:/' | head -c 100)"
            fi
            
            # Check if complete
            COMPLETION=$(grep "Validation complete" "$LOG_FILE" | tail -1)
            if [ ! -z "$COMPLETION" ]; then
                echo "  ✅ Experiment completed!"
            fi
            
            echo ""
        done
    fi
fi

echo ""
echo "💡 Tips:"
echo "  - Quick tests should complete in ~10-15 minutes on CPU"
echo "  - Full experiments take ~4-24 hours depending on hardware"
echo "  - Log files are located in output_*/*/sdt-*/sdt.log"
echo "  - View real-time: tail -f output_cv_quick_test/*/sdt-*/sdt.log"
echo ""
echo "Status as of: $(date)"
echo ""
