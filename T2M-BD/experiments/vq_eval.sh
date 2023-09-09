#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch eval_gpt.sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen ~/git/MaskText2Motion/T2M-BD/experiments/vq_eval.sh

#SBATCH --job-name=eval
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
name='HML3D_VQVAE_eval_26_VQVAE_batch512' # TEMP
dataset_name='t2m'
debug='f'
export CUDA_VISIBLE_DEVICES=6
# export CUDA_LAUNCH_BLOCKING=1

python3 VQ_eval.py \
--batch-size 256 \
--lr 2e-4 \
--total-iter 300000 \
--lr-scheduler 200000 \
--nb-code 8192 \
--code-dim 32 \
--down-t 2 \
--depth 3 \
--dilation-growth-rate 3 \
--out-dir output \
--dataname t2m \
--vq-act relu \
--quantizer ema_reset \
--loss-vel 0.5 \
--recons-loss l1_smooth \
--exp-name ${name} \
--dataname ${dataset_name} \
--resume-pth /data/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-09-05-17-09-05_26_VQVAE_batch512/net_last.pth
# --sep-uplow


sleep 11500