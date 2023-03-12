#!/bin/sh
# cd /users/epinyoan/git/MotionDiffuse/text2motion/experiments/
# sbatch train.sh
# screen -S temp ~/git/MotionDiffuse/text2motion/experiments/train.sh

#SBATCH --job-name=job
#SBATCH --partition=GPU
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MotionDiffuse/text2motion
conda activate motiondiffuse
name='3_SimMIM_kit'
dataset_name='kit'
debug='t'
# export CUDA_VISIBLE_DEVICES=2,3
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -u tools/train.py \
    --batch_size 128 \
    --times 50 \
    --num_epochs 50 \
    --dataset_name ${dataset_name} \
    --num_layers 8 \
    --diffusion_steps 1000 \
    --corrupt diffusion \
    --data_parallel \
    --project MD_1 \
    --name ${name} \
    --debug ${debug} \
    --gpu_id 0 1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/evaluation.py checkpoints/${dataset_name}/${name}/${debug}

sleep 500