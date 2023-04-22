#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch eval_gpt.sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen -L -Logfile HML3D_VQVAE_official_last ~/git/MaskText2Motion/T2M-BD/experiments/eval.sh

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
name='HML3D_eval_gpt' # TEMP
dataset_name='t2m'
debug='f'
export CUDA_VISIBLE_DEVICES=3
# export CUDA_LAUNCH_BLOCKING=1

python3 GPT_eval_multi.py  \
    --exp-name ${name} \
    --batch-size 128 \
    --num-layers 8 \
    --embed-dim-gpt 512 \
    --nb-code 512 \
    --n-head-gpt 16 \
    --block-size 52 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth pretrained/VQVAE/net_last.pth \
    --vq-name VQVAE \
    --out-dir output \
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
    --vq-act relu \
    --resume-trans output/t2m/2023-04-17-21-53-21_HML3D_17_lossAllToken_vqLast_small/net_best_fid.pth
sleep 500