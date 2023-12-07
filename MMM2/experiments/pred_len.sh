#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch eval_gpt.sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen ~/git/MaskText2Motion/T2M-BD/experiments/pred_len.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-BD
conda activate T2M-GPT
export CUDA_VISIBLE_DEVICES=2,3

python3 length_predictor.py
sleep 500