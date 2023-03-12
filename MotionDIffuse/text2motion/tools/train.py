import os
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionTransformer, MaskMotionTransformer, SpatioTemporalMotionTransformer
from trainers import DDPMTrainer
from datasets import Text2MotionDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import torch
import torch.distributed as dist
from models.GaitMixer import SpatioTemporalTransformer
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric


def build_models(opt, dim_pose):
    # encoder = SpatioTemporalTransformer(
    #         num_frames=opt.max_motion_length,
    #         num_joints=opt.joints_num,
    #         # num_layers=opt.num_layers,
    #         # latent_dim=opt.latent_dim,
    #         # no_clip=opt.no_clip,
    #         # no_eff=opt.no_eff
    # )
    # return encoder
    if opt.corrupt == 'diffusion':
        encoder = MotionTransformer(
            input_feats=dim_pose,
            num_frames=opt.max_motion_length,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff)
    elif opt.corrupt == 'mask':
        encoder = MaskMotionTransformer(
            input_feats=dim_pose,
            num_frames=opt.max_motion_length,
            num_layers=opt.num_layers,
            latent_dim=opt.latent_dim,
            no_clip=opt.no_clip,
            no_eff=opt.no_eff)
    else:
        raise NotImplementedError(f"unknown corrupt type: {opt.corrupt}")
    return encoder


if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()
    rank, world_size = get_dist_info()

    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if opt.dataset_name == 't2m':
        opt.data_root = './data/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        radius = 4
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = './data/KIT-ML'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        radius = 240 * 8
        fps = 12.5
        dim_pose = 251
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain

    else:
        raise KeyError('Dataset Does Not Exist')

    if opt.debug:
        opt.num_epochs = 1
        opt.diffusion_steps = 20

    dim_word = 300
    mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
    std = np.load(pjoin(opt.data_root, 'Std.npy'))

    train_split_file = pjoin(opt.data_root, 'train.txt')

    encoder = build_models(opt, dim_pose)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    elif opt.data_parallel:
        encoder = MMDataParallel(
            encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
    else:
        encoder = encoder.cuda()

    trainer = DDPMTrainer(opt, encoder)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)
    trainer.train(train_dataset)

    # Visualize
    opt.is_train = False
    trainer.eval_mode()
    # for debugging
    # trainer.load('/home/epinyoan/git/MotionDiffuse/text2motion/checkpoints/kit/temp2/latest.tar')
    with torch.no_grad():
        caption = ["a person is jumping"]
        m_lens = torch.LongTensor([60]).cuda()
        pred_motions = trainer.generate(caption, m_lens, dim_pose)
        motion = pred_motions[0].cpu().numpy()
        motion = motion * std + mean
        title = caption[0] + " #%d" % motion.shape[0]
        joint = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num).numpy()
        if opt.dataset_name == 'kit':
            joint = joint/1000

        plot_3d_motion(f'{opt.save_root}/{title}.gif', kinematic_chain, joint, title=title, fps=20)