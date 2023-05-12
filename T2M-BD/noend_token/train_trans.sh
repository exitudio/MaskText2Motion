#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/noend_token/
# sbatch train_trans.sh

# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/noend_token/
# screen -L -Logfile HML3D_27_noEndToken -S temp ~/git/MaskText2Motion/T2M-BD/noend_token/train_trans.sh

#SBATCH --job-name=HML3D_27_noEndToken
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/T2M-BD
conda activate T2M-GPT
name='HML3D_27_noEndToken' # TEMP
dataset_name='t2m'
vq_name='VQVAE_official_last'
debug='f'
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
# --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-BD/output/2023-04-08-08-16-27_2_train_withEval/net_last.pth
MULTI_BATCH=4

python3 noend_token_train_t2m_trans.py  \
    --exp-name ${name} \
    --batch-size $((128*MULTI_BATCH)) \
    --num-layers 9 \
    --embed-dim-gpt 1024 \
    --nb-code 512 \
    --n-head-gpt 16 \
    --block-size 50 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth output/${dataset_name}/${vq_name}/net_last.pth \
    --vq-name ${vq_name} \
    --out-dir output/${dataset_name} \
    --total-iter 1 \
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