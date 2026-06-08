#!/bin/bash
# Restormer training script
# Usage: bash scripts/restormer_train.sh [scale] [max_steps]

SCALE=${1:-4}
MAX_STEPS=${2:-300000}
MODEL="Restormer"
DATASET="BraTSReg"

EXP_NAME="${DATASET}-${MODEL}x${SCALE}"
EXP_DIR="./logs/${EXP_NAME}"

echo "============================================================"
echo "MCSR Training: ${EXP_NAME}"
echo "============================================================"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/checkpoints" "${EXP_DIR}/model_src"

# Run training
nohup python main.py \
    --model $MODEL \
    --dataset $DATASET \
    --upscale $SCALE \
    --epochs $MAX_STEPS \
    > "${EXP_DIR}/training.log" 2>&1 &

PID=$!
echo $PID > "${EXP_DIR}/train.pid"

echo "Training started!"
echo "  PID: $PID"
echo "  Log: ${EXP_DIR}/training.log"
echo "  Checkpoints: ${EXP_DIR}/checkpoints/"