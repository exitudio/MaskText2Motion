#!/bin/sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-GPT/experiments/
# screen -S temp ~/git/MaskText2Motion/T2M-GPT/experiments/speedtest.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-GPT
conda activate T2M-GPT
export CUDA_VISIBLE_DEVICES=7

python3 speedtest.py
sleep 500