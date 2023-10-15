#!/bin/sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen -S temp /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/speedtest.sh

. /home/epinyoan/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/MaskText2Motion/T2M-BD
conda activate T2M-GPT
export CUDA_VISIBLE_DEVICES=0

python3 speedtest.py
sleep 500