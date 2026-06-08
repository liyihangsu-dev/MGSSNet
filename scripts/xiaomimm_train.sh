#!/bin/bash
# XiaomiMM training script

screen -dmS "train_xiaomimm_x2" bash -c "python main.py --model XiaomiMM --dataset BraTSReg --upscale 2 --epochs 300 --log 1; exec bash"
screen -dmS "train_xiaomimm_x4" bash -c "python main.py --model XiaomiMM --dataset BraTSReg --upscale 4 --epochs 300 --log 1; exec bash"
screen -dmS "train_xiaomimm_x8" bash -c "python main.py --model XiaomiMM --dataset BraTSReg --upscale 8 --epochs 300 --log 1; exec bash"

echo "XiaomiMM x2/x4/x8 training started in screen"