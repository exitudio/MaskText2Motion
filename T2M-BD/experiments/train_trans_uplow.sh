#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# sbatch train_trans.sh

# cd /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/
# screen -L -Logfile HML3D_37_upperLower_sepMask_Bilando_18Lyr_CB8192x32_RandUpperLowerSep_pkeep.9 -S temp ~/git/MaskText2Motion/T2M-BD/experiments/train_trans_uplow.sh

#SBATCH --job-name=HML3D_31_CFG
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
name='HML3D_37_upperLower_sepMask_Bilando_18Lyr_CB8192x32_RandUpperLowerSep_temp0.9' # TEMP
dataset_name='t2m'
vq_name='2023-08-08-00-29-40_16_VQVAE_upperlower_notShareCB_20batchResetNRandom_8192x32'
debug='f'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_LAUNCH_BLOCKING=1
# --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-BD/output/2023-04-08-08-16-27_2_train_withEval/net_last.pth
MULTI_BATCH=4

python3 train_t2m_trans_uplow.py  \
    --exp-name ${name} \
    --batch-size $((128*MULTI_BATCH)) \
    --num-layers 9 \
    --embed-dim-gpt 1024 \
    --nb-code 8192 \
    --code-dim 32 \
    --n-head-gpt 16 \
    --block-size 51 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth output/vq/${vq_name}/net_last.pth \
    --vq-name ${vq_name} \
    --out-dir output/${dataset_name} \
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