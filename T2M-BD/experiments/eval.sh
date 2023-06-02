#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch eval_gpt.sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen -L -Logfile eval_HML3D_32_endInput_noEndOutput_51Block_Small ~/git/MaskText2Motion/T2M-BD/experiments/eval.sh

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
name='eval_HML3D_32_endInput_noEndOutput_51Block_Small' # TEMP
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
    --block-size 51 \
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
    --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-05-31-11-03-15_HML3D_32_endInput_noEndOutput_51Block_Small/net_best_fid.pth
sleep 500