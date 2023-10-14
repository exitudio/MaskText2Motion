#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/
# sbatch train_trans.sh

# cd /home/epinyoan/git/MaskText2Motion/
# screen -L -Logfile HML3D_45_crsAtt1lyr_40breset -S temp ~/git/MaskText2Motion/T2M-BD/experiments/train_trans.sh

#SBATCH --job-name=HML3D_43_token1stStage_cdim8192_32_lr0.0001_mask0.1-1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. /home/epinyoan/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/MaskText2Motion/T2M-BD
conda activate T2M-GPT
name='HML3D_45_crsAtt1lyr_40breset' # TEMP
dataset_name='t2m'
vq_name='2023-10-07-10-29-34_23_VQVAE_40batchResetNRandom_8192_32'
debug='f'
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
# --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-BD/output/2023-04-08-08-16-27_2_train_withEval/net_last.pth
MULTI_BATCH=4

python3 train_t2m_trans.py  \
    --exp-name ${name} \
    --batch-size $((128*MULTI_BATCH)) \
    --num-layers 9 \
    --num-local-layer 1 \
    --embed-dim-gpt 1024 \
    --nb-code 8192 \
    --code-dim 32 \
    --n-head-gpt 16 \
    --block-size 51 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --vq-name ${vq_name} \
    --out-dir output/${dataset_name} \
    --total-iter $((300000/MULTI_BATCH)) \
    --lr-scheduler $((150000/MULTI_BATCH)) \
    --lr 0.0001 \
    --dataname ${dataset_name} \
    --down-t 2 \
    --depth 3 \
    --quantizer ema_reset \
    --eval-iter $((20000/MULTI_BATCH)) \
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