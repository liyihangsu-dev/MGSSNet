#!/bin/bash
# DINO-Guided CMS training script
# Usage: bash scripts/dino_cms_train.sh [max_steps]

MAX_STEPS=${1:-300000}
MODEL="dino_cms"
DATASET="BraTSReg"
SCALE=${SCALE:-4}

EXP_NAME="${DATASET}-${MODEL}-s${SCALE}"
EXP_DIR="./logs/${EXP_NAME}"

echo "============================================================"
echo "MCSR Training: ${EXP_NAME}"
echo "============================================================"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/checkpoints" "${EXP_DIR}/model_src"

# Run training
nohup python main.py \
    --model $MODEL \
    --dataset $DATASET \
    --downsample_k $SCALE \
    --max_steps $MAX_STEPS \
    --num_workers 4 \
    --log \
    > "${EXP_DIR}/training.log" 2>&1 &

PID=$!
echo $PID > "${EXP_DIR}/train.pid"

echo "Training started!"
echo "  PID: $PID"
echo "  Log: ${EXP_DIR}/training.log"
echo "  Checkpoints: ${EXP_DIR}/checkpoints/"