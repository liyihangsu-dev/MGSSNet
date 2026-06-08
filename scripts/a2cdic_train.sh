#!/bin/bash
# A2CDic training script (TMI 2025)

# BraTSReg
screen -dmS "train_a2cdic_brats_x2" bash -c "python main.py --model A2CDic --dataset BraTSReg --upscale 2 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_a2cdic_brats_x4" bash -c "python main.py --model A2CDic --dataset BraTSReg --upscale 4 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_a2cdic_brats_x8" bash -c "python main.py --model A2CDic --dataset BraTSReg --upscale 8 --epochs 300 --batch_size 1 --log 1; exec bash"

# IXI
screen -dmS "train_a2cdic_ixi_x2" bash -c "python main.py --model A2CDic --dataset IXI --upscale 2 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_a2cdic_ixi_x4" bash -c "python main.py --model A2CDic --dataset IXI --upscale 4 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_a2cdic_ixi_x8" bash -c "python main.py --model A2CDic --dataset IXI --upscale 8 --epochs 300 --batch_size 1 --log 1; exec bash"

echo "A2CDic training started for BraTSReg and IXI at x2/x4/x8"