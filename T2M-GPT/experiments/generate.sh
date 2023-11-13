#!/bin/sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-GPT/experiments/
# screen -S temp ~/git/MaskText2Motion/T2M-GPT/experiments/generate.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-GPT
conda activate T2M-GPT
export CUDA_VISIBLE_DEVICES=7

python3 speedtest.py \
    'a person walks forward then turns completely around and does a cartwheel.' \
    /home/epinyoan/git/MaskText2Motion/T2M-GPT/output/npy/a_person_walks_forward_then_turns_completely_around_and_does_a_cartwheel4

sleep 500