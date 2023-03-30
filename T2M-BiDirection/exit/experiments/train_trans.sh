#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BiDirection/exit/experiments/
# sbatch train_trans.sh
# screen -S temp ~/git/MaskText2Motion/T2M-BiDirection/exit/experiments/train_trans.sh

#SBATCH --job-name=trans
#SBATCH --partition=GPU
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-BiDirection
conda activate T2M-GPT
name='1_TRANS'
dataset_name='kit'
vq_name='VQT_6_cbEmb32x8192_fixQMultiGPU'
debug='f'
# export CUDA_VISIBLE_DEVICES=3
CUDA_LAUNCH_BLOCKING=1
python3 train_t2m_trans.py  \
    --exp-name ${name} \
    --batch-size 128 \
    --num-layers 8 \
    --embed-dim-gpt 256 \
    --nb-code 8192 \
    --code-dim 32 \
    --n-head-gpt 16 \
    --block-size 196 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth output/${vq_name}/net_last.pth \
    --vq-name ${vq_name} \
    --out-dir output_GPT_Final \
    --total-iter 300000 \
    --lr-scheduler 150000 \
    --lr 0.0001 \
    --dataname ${dataset_name} \
    --down-t 2 \
    --depth 3 \
    --quantizer ema_reset \
    --eval-iter 10000 \
    --pkeep 0.5 \
    --dilation-growth-rate 3 \
    --vq-act relu
sleep 500