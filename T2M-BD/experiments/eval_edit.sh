#!/bin/sh
# cd /home/epinyoan/git/MaskText2Motion
# screen -L -Logfile EvalEDIT_HML3D_44_crsAtt1lyr_mask0.5-1_20bRest_2 /home/epinyoan/git/MaskText2Motion/T2M-BD/experiments/eval_edit.sh

. /home/epinyoan/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/MaskText2Motion/T2M-BD
conda activate T2M-GPT
name='EvalEDIT_HML3D_44_crsAtt1lyr_mask0.5-1_20bRest_2' # TEMP
dataset_name='t2m'
debug='f'
export CUDA_VISIBLE_DEVICES=2
# export CUDA_LAUNCH_BLOCKING=1

python3 eval_edit.py  \
    --exp-name ${name} \
    --batch-size 128 \
    --num-layers 9 \
    --num-local-layer 1 \
    --embed-dim-gpt 1024 \
    --nb-code 8192 \
    --code-dim 32 \
    --n-head-gpt 16 \
    --block-size 51 \
    --ff-rate 4 \
    --drop-out-rate 0.1 \
    --resume-pth output/vq/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32/net_last.pth \
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
    --resume-trans /home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-12-10-11-15_HML3D_45_crsAtt1lyr_40breset_WRONG_THIS_20BRESET/net_last.pth
sleep 500