#!/bin/bash
# MTrans training script (TMI 2022)

# BraTSReg
screen -dmS "train_mtrans_brats_x2" bash -c "python main.py --model MTrans --dataset BraTSReg --upscale 2 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_mtrans_brats_x4" bash -c "python main.py --model MTrans --dataset BraTSReg --upscale 4 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_mtrans_brats_x8" bash -c "python main.py --model MTrans --dataset BraTSReg --upscale 8 --epochs 300 --batch_size 1 --log 1; exec bash"

# IXI
screen -dmS "train_mtrans_ixi_x2" bash -c "python main.py --model MTrans --dataset IXI --upscale 2 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_mtrans_ixi_x4" bash -c "python main.py --model MTrans --dataset IXI --upscale 4 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_mtrans_ixi_x8" bash -c "python main.py --model MTrans --dataset IXI --upscale 8 --epochs 300 --batch_size 1 --log 1; exec bash"

echo "MTrans training started for BraTSReg and IXI at x2/x4/x8"