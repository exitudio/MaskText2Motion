#!/bin/bash

# screen -S temp ~/git/MotionDiffuse/text2motion/experiments/eval.sh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate motiondiffuse

cd /home/epinyoan/git/MotionDiffuse/text2motion

name='test5'
dataset_name='kit'
debug='f'
# code needs to be modified to support multiple GPUs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/evaluation.py checkpoints/${dataset_name}/${name}/${debug}

sleep 500