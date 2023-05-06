#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-GPT-Batch/bidirec_autoregress/
# sbatch train_trans.sh
# screen -S temp ~/git/MaskText2Motion/T2M-GPT-Batch/bidirec_autoregress/train_trans.sh

#SBATCH --job-name=HML3D_3_BidirAutoreg_big_sameTxtEmb
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-GPT-Batch/
conda activate T2M-GPT
name='HML3D_3_BidirAutoreg_big_sameTxtEmb' # TEMP
dataset_name='t2m'
vq_name='HML3D_VQVAE_official_last'
debug='f'
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
MULTI_BATCH=4

python3 train_bidirec_autoregress.py  \
    --exp-name ${name} \
    --batch-size $((128*MULTI_BATCH)) \
    --num-layers 9 \
    --embed-dim-gpt 1024 \
    --nb-code 512 \
    --n-head-gpt 16 \
    --block-size 49 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth output/${vq_name}/net_last.pth \
    --vq-name ${vq_name} \
    --out-dir output \
    --total-iter $((300000/MULTI_BATCH)) \
    --lr-scheduler $((150000/MULTI_BATCH)) \
    --lr 0.0001 \
    --dataname ${dataset_name} \
    --down-t 2 \
    --depth 3 \
    --quantizer ema_reset \
    --eval-iter $((10000/MULTI_BATCH)) \
    --pkeep 0.5 \
    --dilation-growth-rate 3 \
    --vq-act relu
sleep 500

# original setting
# --batch-size $((128*num_gpu)) \
# --num-layers 9 \
# --embed-dim-gpt 1024 \
# --n-head-gpt 16 \
# --total-iter $((300000/num_gpu)) \
# --lr-scheduler $((150000/num_gpu)) \
# --eval-iter $((10000/num_gpu)) \