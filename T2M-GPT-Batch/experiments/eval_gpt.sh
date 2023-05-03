#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-GPT-Batch/experiments/
# sbatch eval_gpt.sh
# cd /home/epinyoan/git/MaskText2Motion/T2M-GPT-Batch/experiments/
# screen -L -Logfile HML3D_eval_gpt ~/git/MaskText2Motion/T2M-GPT-Batch/experiments/eval_gpt.sh

#SBATCH --job-name=HML3D_eval_gpt
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-GPT-Batch
conda activate T2M-GPT
name='HML3D_eval_gpt' # TEMP
dataset_name='t2m'
vq_name='HML3D_VQVAE_official_last'
debug='f'
export CUDA_VISIBLE_DEVICES=5
# export CUDA_LAUNCH_BLOCKING=1

python3 GPT_eval_multi.py  \
    --exp-name ${name} \
    --batch-size 128 \
    --num-layers 9 \
    --embed-dim-gpt 1024 \
    --nb-code 512 \
    --n-head-gpt 16 \
    --block-size 51 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth output/${vq_name}/net_last.pth \
    --vq-name ${vq_name} \
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
    --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-GPT-Batch/output/2023-04-19-16-52-09_HML3D_8_officialPretrainLast/net_best_fid.pth
sleep 500