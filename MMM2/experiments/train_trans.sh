#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/
# sbatch train_trans.sh

# cd /home/epinyoan/git/MaskText2Motion/
# screen -L -Logfile HML3D_0_LFQ_cmmit1_entp.5_div1_cdim15__binaryLoss_fixMaskIdDup_fixWeightLoss -S temp ~/git/MaskText2Motion/MMM2/experiments/train_trans.sh

#SBATCH --job-name=HML3D_0_LFQ_cmmit1_entp.5_div1_cdim15__binaryLoss_fixMaskIdDup_fixWeightLoss
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. /home/epinyoan/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/MaskText2Motion/MMM2
conda activate T2M-GPT
name='HML3D_0_LFQ_cmmit1_entp.5_div1_cdim15__binaryLoss_fixMaskIdDup_fixWeightLoss' # TEMP
dataset_name='t2m'
vq_name='2023-12-09-10-44-31_0_VQVAE_LFQ_cmmit1_entp.5_div1_cdim15'
debug='f'
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1
# --resume-trans /home/epinyoan/git/MaskText2Motion/MMM2/output/2023-04-08-08-16-27_2_train_withEval/net_last.pth
MULTI_BATCH=4

python3 train_t2m_trans.py  \
    --exp-name ${name} \
    --batch-size $((128*MULTI_BATCH)) \
    --num-layers 9 \
    --num-local-layer 1 \
    --embed-dim-gpt 1024 \
    --nb-code 32768 \
    --code-dim 15 \
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