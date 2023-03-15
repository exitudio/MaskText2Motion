#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-GPT/experiments/
# sbatch train_vq.sh
# screen -S temp ~/git/MaskText2Motion/T2M-GPT/experiments/train_vq.sh

#SBATCH --job-name=1GPU
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-GPT
conda activate T2M-GPT
name='2_VQVAE_1GPU'
dataset_name='kit'
debug='f'
# export CUDA_VISIBLE_DEVICES=3
python3 train_vq.py \
    --batch-size 256 \
    --lr 2e-4 \
    --total-iter 300000 \
    --lr-scheduler 200000 \
    --nb-code 512 \
    --down-t 2 \
    --depth 3 \
    --dilation-growth-rate 3 \
    --out-dir output \
    --dataname ${dataset_name} \
    --vq-act relu \
    --quantizer ema_reset \
    --loss-vel 0.5 \
    --recons-loss l1_smooth \
    --exp-name ${name}

sleep 500