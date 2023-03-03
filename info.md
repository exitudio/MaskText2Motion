copy pretrain weights from 'https://drive.google.com/uc?id=1vzBZ2rNCQWBQpYvC6hpyJfR3iK1O_FEG' which I got if from Colab: https://colab.research.google.com/drive/1Dp6VsZp2ozKuu9ccMmsDjyij_vXfCYb3?usp=sharing#scrollTo=3WWX0dF9pH-W

The weights they specified in the info it's not compateble with the model ()
```
self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
KeyError: 'encoder'
```

# Install
- When install, using  ~/miniconda3/envs/motiondiffuse/bin/pip install
- Pytorch3D needs to install from source, conda install doesn't work. [Instruction](pip install -r requirements.txt)
```
~/miniconda3/envs/motiondiffuse/bin/pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
- if mmcv fail can install like this https://mmcv.readthedocs.io/en/latest/get_started/installation.html
```
pip install -U openmim
mim install mmcv-full
```





run visualize
```
cd /home/epinyoan/git/MotionDiffuse/text2motion
conda activate motiondiffuse
python -u ./tools/visualization.py --opt_path checkpoints/t2m/t2m_motiondiffuse/opt.txt --text "a man backflip" --motion_length 2     --result_path "test_sample.gif"
```

visualize KIT
```
python -u tools/visualization.py     --opt_path checkpoints/kit/kit_motiondiffuse/opt.txt     --text "A person throwing a left jab"     --motion_length 60     --result_path "test_sample.gif"     --gpu_id 0
```

generate_motion
```
cd /home/epinyoan/git/MotionDiffuse/text2motion/study
conda activate motiondiffuse
python generate_motion.py
```

# Run Training
```
conda activate motiondiffuse
cd text2motion
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -u tools/train.py \
    --name kit_baseline_dp_2gpu_8layers_1000 \
    --batch_size 128 \
    --times 50 \
    --num_epochs 50 \
    --dataset_name kit \
    --num_layers 8 \
    --diffusion_steps 1000 \
    --data_parallel \
    --gpu_id 0 1 2 3
```

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
jupyter notebook
```