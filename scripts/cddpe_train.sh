#!/bin/bash
# CDDPE training script (AAAI 2026)

# BraTSReg
screen -dmS "train_cddpe_brats_x2" bash -c "python main.py --model CDDPE --dataset BraTSReg --upscale 2 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_cddpe_brats_x4" bash -c "python main.py --model CDDPE --dataset BraTSReg --upscale 4 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_cddpe_brats_x8" bash -c "python main.py --model CDDPE --dataset BraTSReg --upscale 8 --epochs 300 --batch_size 1 --log 1; exec bash"

# IXI
screen -dmS "train_cddpe_ixi_x2" bash -c "python main.py --model CDDPE --dataset IXI --upscale 2 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_cddpe_ixi_x4" bash -c "python main.py --model CDDPE --dataset IXI --upscale 4 --epochs 300 --batch_size 1 --log 1; exec bash"
screen -dmS "train_cddpe_ixi_x8" bash -c "python main.py --model CDDPE --dataset IXI --upscale 8 --epochs 300 --batch_size 1 --log 1; exec bash"

echo "CDDPE training started for BraTSReg and IXI at x2/x4/x8"