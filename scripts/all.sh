#!/bin/bash
# Run all training scripts ONE BY ONE (sequential, not parallel)

set -e

SCRIPT_DIR="/mcsr_project/scripts"
LOG_FILE="/mcsr_project/scripts/all_train.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting sequential training..."
log "=========================================="

# Function to run one training job
run_train() {
    local model=$1
    local dataset=$2
    local scale=$3
    local batch_size=${4:-1}

    log ">>> Training $model on $dataset x$scale (bs=$batch_size)..."

    python main.py \
        --model "$model" \
        --dataset "$dataset" \
        --upscale "$scale" \
        --epochs 300 \
        --batch_size "$batch_size" \
        --log 1

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log "!!! FAILED: $model $dataset x$scale (exit code: $exit_code)"
        return $exit_code
    fi
    log ">>> DONE: $model $dataset x$scale"
}

# Clear log
> "$LOG_FILE"

# ===== 1. McASSR =====
log ""
log "===== [1/7] McASSR ====="
run_train "McASSR" "BraTSReg" 2 1
run_train "McASSR" "BraTSReg" 4 1
run_train "McASSR" "BraTSReg" 8 1
run_train "McASSR" "IXI" 2 1
run_train "McASSR" "IXI" 4 1
run_train "McASSR" "IXI" 8 1

# ===== 2. A2CDic =====
log ""
log "===== [2/7] A2CDic ====="
run_train "A2CDic" "BraTSReg" 2 1
run_train "A2CDic" "BraTSReg" 4 1
run_train "A2CDic" "BraTSReg" 8 1
run_train "A2CDic" "IXI" 2 1
run_train "A2CDic" "IXI" 4 1
run_train "A2CDic" "IXI" 8 1

# ===== 3. CDDPE =====
log ""
log "===== [3/7] CDDPE ====="
run_train "CDDPE" "BraTSReg" 2 1
run_train "CDDPE" "BraTSReg" 4 1
run_train "CDDPE" "BraTSReg" 8 1
run_train "CDDPE" "IXI" 2 1
run_train "CDDPE" "IXI" 4 1
run_train "CDDPE" "IXI" 8 1

# ===== 4. DINet =====
log ""
log "===== [4/7] DINet ====="
run_train "DINet" "BraTSReg" 2 1
run_train "DINet" "BraTSReg" 4 1
run_train "DINet" "BraTSReg" 8 1
run_train "DINet" "IXI" 2 1
run_train "DINet" "IXI" 4 1
run_train "DINet" "IXI" 8 1

# ===== 5. MTrans =====
log ""
log "===== [5/7] MTrans ====="
run_train "MTrans" "BraTSReg" 2 1
run_train "MTrans" "BraTSReg" 4 1
run_train "MTrans" "BraTSReg" 8 1
run_train "MTrans" "IXI" 2 1
run_train "MTrans" "IXI" 4 1
run_train "MTrans" "IXI" 8 1

# ===== 6. SANet =====
log ""
log "===== [6/7] SANet ====="
run_train "SANet" "BraTSReg" 2 1
run_train "SANet" "BraTSReg" 4 1
run_train "SANet" "BraTSReg" 8 1
run_train "SANet" "IXI" 2 1
run_train "SANet" "IXI" 4 1
run_train "SANet" "IXI" 8 1

# ===== 7. Restormer =====
log ""
log "===== [7/7] Restormer ====="
run_train "Restormer" "BraTSReg" 2 1
run_train "Restormer" "BraTSReg" 4 1
run_train "Restormer" "BraTSReg" 8 1
run_train "Restormer" "IXI" 2 1
run_train "Restormer" "IXI" 4 1
run_train "Restormer" "IXI" 8 1

log ""
log "=========================================="
log "ALL TRAINING COMPLETE!"
log "Log saved to: $LOG_FILE"
log "=========================================="
