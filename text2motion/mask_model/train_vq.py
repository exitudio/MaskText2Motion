import os
from os.path import join as pjoin

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *

from trainers import DDPMTrainer
from datasets import Text2MotionDataset

from mmcv.runner import get_dist_info, init_dist
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import torch
import torch.distributed as dist
from models.GaitMixer import SpatioTemporalTransformer
from utils.plot_script import plot_3d_motion
from utils.motion_process import recover_from_ric

from tqdm import tqdm

from option import get_opt
from motiontransformer import MotionTransformerOnly, MotionTransformerOnly2, generate_src_mask
from mask_model.quantize import VectorQuantizer2
from torch.utils.data import DataLoader
from datasets import build_dataloader
from mask_model.util import hinge_d_loss, vanilla_d_loss, MeanMask
from utils.logs import UnifyLog
from study.mylib import visualize_2motions

if __name__ == '__main__':
    opt, dim_pose, kinematic_chain, mean, std, train_split_file = get_opt()
    # [TODO] check "Text2MotionDataset" from Text2Motion, there are multiple version (V2, ...)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, opt.times)
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    latent_dim = 256
    codebook_dim = 32
    encoder = MotionTransformerOnly2(input_feats=dim_pose, 
                                    output_feats=codebook_dim, 
                                    latent_dim=latent_dim, 
                                    num_layers=8)
    decoder = MotionTransformerOnly2(input_feats=codebook_dim, 
                                    output_feats=dim_pose, 
                                    latent_dim=latent_dim, 
                                    num_layers=8)
    discriminator = MotionTransformerOnly2(input_feats=dim_pose, 
                                    output_feats=1, 
                                    latent_dim=latent_dim, 
                                    num_layers=4)
    quantize = VectorQuantizer2(n_e = 8192,
                                e_dim = codebook_dim)
    unify_log = UnifyLog(opt, encoder)
    if opt.data_parallel:
        encoder = MMDataParallel(encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        decoder = MMDataParallel(decoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        discriminator = MMDataParallel(discriminator.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        quantize = MMDataParallel(quantize.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=opt.batch_size,
        drop_last=True,
        workers_per_gpu=4,
        shuffle=True,
        dist=opt.distributed,
        num_gpus=len(opt.gpu_id))
    
    # [TODO] VQGAN uses lr = 4.5e-6
    opt_ae = torch.optim.Adam(list(encoder.parameters())+
                                list(decoder.parameters())+
                                list(quantize.parameters()),
                                lr=opt.lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(discriminator.parameters(),
                                lr=opt.lr, betas=(0.5, 0.9))
    
    cur_epoch = 0
    encoder.train(), decoder.train(), discriminator.train(), quantize.train()
    num_batch = len(train_loader)
    dis_start_epoch = 50
    print('num batch:', num_batch)


    for epoch in tqdm(range(cur_epoch, opt.num_epochs), desc="Epoch", position=0):
        for i, batch_data in enumerate(tqdm(train_loader, desc=" Num batch", position=1)):
            caption, motions, m_lens = batch_data
            motions = motions.detach().to(opt.device).float()
            B, T = motions.shape[:2]
            length = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(opt.device)
            src_mask = generate_src_mask(T, length).to(motions.device).unsqueeze(-1)
            B, N, _ = motions.shape
            num_heads = 8
            src_mask_attn = src_mask.view(B, 1, 1, N).repeat(1, num_heads, N, 1)
            mean_mask = MeanMask(src_mask, dim_pose)

            z = encoder(motions, src_mask=src_mask_attn)
            z_q = quantize(z) * src_mask
            qloss = mean_mask.mean((z_q.detach()-z)**2 * src_mask) + 0.25 * \
                            mean_mask.mean((z_q - z.detach()) ** 2 * src_mask)
            # preserve gradients
            z_q = z + (z_q - z).detach()
            recon = decoder(z_q, src_mask=src_mask_attn)

            ##### GAN loss
            # [TODO] mask the shorter frames for gan loss
            # [TODO] skip perceptual loss (LPIPS)
            rec_loss = mean_mask.mean((motions - recon) ** 2 * src_mask)
            loss = rec_loss + qloss
            
            # [TODO] clarify: discriminator of VQGAN output only 1x30x30 from input 3x256x256
            # [TODO] g_loss is negative. Probably normal??
            g_loss = 0
            if epoch > dis_start_epoch:
                logits_fake = discriminator(recon, src_mask=src_mask_attn)
                g_loss = -mean_mask.mean(logits_fake)
                d_weight = .5 # [TODO] skip calculate_adaptive_weight
                disc_factor = 1 # [TODO] adopt_weight
                loss += d_weight * disc_factor * g_loss 
            
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
            
            ##### Discriminator loss
            # [TODO] mask the shorter frames for dis loss
            d_loss = 0
            if epoch > dis_start_epoch:
                logits_real = discriminator(motions.detach(), src_mask=src_mask, length=length) # [TODO] Can we use the same logits_real from GAN loss??? 
                logits_fake = discriminator(recon.detach(), src_mask=src_mask, length=length)
                d_loss = disc_factor * vanilla_d_loss(logits_real, logits_fake)

                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

            unify_log.log({'rec_loss:':rec_loss, 
                           'g_loss':g_loss, 
                           'qloss':qloss, 
                           'Dis loss': d_loss
                           }, step=epoch*num_batch + i)

        motion1 = motions[0].detach().cpu().numpy()
        motion2 = recon[0].detach().cpu().numpy()
        visualize_2motions(motion1, motion2, std, mean, opt.dataset_name, length[0], 
                           save_path=f'{opt.save_root}/epoch_{epoch}.html')
        unify_log.save_model(encoder, 'encoder.pth')
        unify_log.save_model(quantize, 'quantize.pth')
        unify_log.save_model(decoder, 'decoder.pth')

    # # Visualize
    # opt.is_train = False
    # trainer.eval_mode()
    # # for debugging
    # # trainer.load('/home/epinyoan/git/MotionDiffuse/text2motion/checkpoints/kit/temp2/latest.tar')
    # with torch.no_grad():
    #     caption = ["a person is jumping"]
    #     m_lens = torch.LongTensor([60]).cuda()
    #     pred_motions = trainer.generate(caption, m_lens, dim_pose)
    #     motion = pred_motions[0].cpu().numpy()
    #     motion = motion * std + mean
    #     title = caption[0] + " #%d" % motion.shape[0]
    #     joint = recover_from_ric(torch.from_numpy(motion).float(), opt.joints_num).numpy()
    #     if opt.dataset_name == 'kit':
    #         joint = joint/1000

    #     plot_3d_motion(f'{opt.save_root}/{title}.gif', kinematic_chain, joint, title=title, fps=20)