#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch train_vq.sh

# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments
# screen -S temp ~/git/MaskText2Motion/T2M-BD/experiments/train_vq.sh

#SBATCH --job-name=1GPU
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-BD
conda activate T2M-GPT
name='16_VQVAE_upperlower_notShareCB'
dataset_name='t2m'
debug='f'
# export CUDA_VISIBLE_DEVICES=3
python3 train_vq.py \
    --batch-size 256 \
    --lr 2e-4 \
    --total-iter 300000 \
    --lr-scheduler 200000 \
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

# --sep-uplow \
sleep 500