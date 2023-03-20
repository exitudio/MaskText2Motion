#!/bin/bash

# screen -S temp ~/git/MaskText2Motion/MotionDiffuse/text2motion/mask_model/experiments/eval.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate motiondiffuse

cd /home/epinyoan/git/MaskText2Motion/MotionDiffuse/text2motion

vq_path='/home/epinyoan/git/MaskText2Motion/MotionDiffuse/text2motion/checkpoints/kit/2023-03-04-20-39-40_13_vqgan_poseformer_1DisScale/'
transformer_path='/home/epinyoan/git/MaskText2Motion/MotionDiffuse/text2motion/checkpoints/kit/2023-03-10-10-22-39_5_transformer_textCond_fixZeroCond_13_vqgan_poseformer_1DisScale/'
gpu_id=0,1
export CUDA_VISIBLE_DEVICES=0,1
# code needs to be modified to support multiple GPUs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u mask_model/eval/evaluation.py ${vq_path} ${transformer_path} ${gpu_id}

sleep 500