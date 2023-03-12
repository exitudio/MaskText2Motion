#!/bin/sh
# cd /users/epinyoan/git/MaskText2Motion/MotionDiffuse/text2motion/experiments/
# sbatch train_transformer.sh
# screen -S temp ~/git/MaskText2Motion/MotionDiffuse/text2motion/experiments/train_transformer.sh

#SBATCH --job-name=job
#SBATCH --partition=Leo
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=64gb
#SBATCH --time=47:30:00
#SBATCH --output=%x.%j.out

. ~/miniconda3/etc/profile.d/conda.sh
cd ~/git/MaskText2Motion/MotionDiffuse/text2motion
conda activate motiondiffuse
name_save='12_vqgan_poseformer'
name='1_transformer'
dataset_name='kit'
debug='f'
export CUDA_VISIBLE_DEVICES=0,1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -u mask_model/train_transformer.py \
    --num_epochs 300 \
    --dataset_name ${dataset_name} \
    --data_parallel \
    --project MD_vqtransformer \
    --name_save ${name_save} \
    --name ${name} \
    --debug ${debug} \
    --gpu_id 0 1

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -u tools/evaluation.py checkpoints/${dataset_name}/${name}/${debug}

sleep 500