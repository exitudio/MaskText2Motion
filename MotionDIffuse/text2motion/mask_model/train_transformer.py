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

from mask_model.option import get_opt
from mask_model.motiontransformer import MotionTransformerOnly, MotionTransformerOnly2, generate_src_mask
from mask_model.quantize import VectorQuantizer2
from torch.utils.data import DataLoader
from datasets import build_dataloader
from mask_model.util import hinge_d_loss, vanilla_d_loss, MeanMask
from utils.logs import UnifyLog
from study.mylib import visualize_2motions
import sys
import glob
import argparse
from options.base_options import str2bool, init_save_folder

from mask_model.mingpt import GPT
from mask_model.util import configure_optimizers, generate_samples, get_model, get_latest_folder_by_name
import torch.nn.functional as F
import torch.nn as nn
import clip

class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        clip_dim = 512
        clip_version = 'ViT-B/32'
        self.embed_text = nn.Linear(clip_dim, output_dim)

        # load_and_freeze_clip 
        self.clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            self.clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

    def forward(self, raw_text, cond_mask_prob=.1):
        device = next(self.parameters()).device
        default_context_length = 77
        context_length = 20 + 2 # start_token + 20 + end_token
        assert context_length < default_context_length
        texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
        zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
        texts = torch.cat([texts, zero_pad], dim=1)

        # [TODO] MD: uses all seq, MDM uses only the highest token https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py#L343
        texts = self.clip_model.encode_text(texts).float()
    
        mask = torch.bernoulli(torch.ones(texts.shape[0], device=device) * cond_mask_prob).view(texts.shape[0], 1)  # 1-> use null_cond, 0-> use real cond
        texts = texts * (1. - mask)
        return self.embed_text(texts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_epochs', type=int, default=300, help='num epoch')
    parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
    parser.add_argument("--data_parallel", action="store_true", help="Whether to use DP training")
    parser.add_argument("--project", default="project_name")
    parser.add_argument('--name_save', type=str, default="test", help='name_save of this trial')
    parser.add_argument('--name', type=str, default="test", help='Name of this trial')
    parser.add_argument('--debug', type=str2bool, nargs='?', default=True)
    parser.add_argument("--gpu_id", type=int, nargs='+', default=(-1), help='GPU id')
    opt = parser.parse_args()

    save_path = get_latest_folder_by_name(f"checkpoints/{opt.dataset_name}", opt.name_save)

    # mock from train opt
    opt.checkpoints_dir = './checkpoints'
    opt.is_train = True
    opt.feat_bias = 25
    opt.batch_size = 128
    opt.distributed = False
    opt.lr = 4.5e-06
    opt.device = torch.device("cuda")
    opt.name = f'{opt.name}_{opt.name_save}'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    init_save_folder(opt)
    torch.cuda.set_device(opt.gpu_id[0])

    


    opt, dim_pose, kinematic_chain, mean, std, train_split_file = get_opt(opt)
    # [TODO] check "Text2MotionDataset" from Text2Motion, there are multiple version (V2, ...)
    train_dataset = Text2MotionDataset(opt, mean, std, train_split_file, times=50)
    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    latent_dim = 256
    codebook_dim = 32
    generator_dim = 256 # [TODO] decrease generator dim
    text_emb_dim = 512
    encoder = MotionTransformerOnly2(input_feats=dim_pose, 
                                    output_feats=codebook_dim, 
                                    latent_dim=latent_dim, 
                                    num_layers=8)
    decoder = MotionTransformerOnly2(input_feats=codebook_dim, 
                                    output_feats=dim_pose, 
                                    latent_dim=latent_dim, 
                                    num_layers=8)
    # discriminator = MotionTransformerOnly2(input_feats=dim_pose, 
    #                                 output_feats=1, 
    #                                 latent_dim=latent_dim, 
    #                                 num_layers=4)
    quantize = VectorQuantizer2(n_e = 8192,
                                e_dim = codebook_dim)
    textEncoder = TextEncoder(output_dim=text_emb_dim).to(opt.device)
    # [INFO] VQGAN params: GPT(vocab_size=1024, block_size=512, n_layer=24, n_head=16, n_embd=1024)
    transformer = GPT(vocab_size=8192, block_size=196+int(text_emb_dim/generator_dim), 
                      n_layer=8, n_head=8, n_embd=generator_dim)
    encoder.load_state_dict(torch.load(save_path+'/encoder.pth'))
    decoder.load_state_dict(torch.load(save_path+'/decoder.pth'))
    quantize.load_state_dict(torch.load(save_path+'/quantize.pth'))


    unify_log = UnifyLog(opt, encoder)
    if opt.data_parallel:
        encoder = MMDataParallel(encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        decoder = MMDataParallel(decoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        # discriminator = MMDataParallel(discriminator.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        quantize = MMDataParallel(quantize.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
        transformer = MMDataParallel(transformer.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=opt.batch_size,
        drop_last=True,
        workers_per_gpu=4,
        shuffle=True,
        dist=opt.distributed,
        num_gpus=len(opt.gpu_id))
    
    lr =  len(opt.gpu_id) * opt.batch_size * opt.lr
    optim = configure_optimizers(get_model(transformer), lr)
    
    cur_epoch = 0
    encoder.eval(), quantize.eval(), transformer.train() # decoder.eval(), 
    num_batch = len(train_loader)
    print('num batch:', num_batch)


    for epoch in tqdm(range(cur_epoch, opt.num_epochs), desc="Epoch", position=0):
        for i, batch_data in enumerate(tqdm(train_loader, desc=" Num batch", position=1)):
            caption, motions, m_lens = batch_data
            motions = motions.detach().to(opt.device).float()
            B, T = motions.shape[:2]
            length = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(opt.device)
            src_mask = generate_src_mask(T, length).to(motions.device).unsqueeze(-1)
            num_heads = 8
            src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, num_heads, T, 1)
            mean_mask = MeanMask(src_mask, dim_pose)

            z = encoder(motions, src_mask=src_mask_attn)
            z_q, indices = quantize(z)
            z_indices = indices.view(z_q.shape[0], -1)

            c_indices = torch.zeros(B, 1, dtype=z_indices.dtype).to(z_indices.device)
            cz_indices = torch.cat((c_indices, z_indices), dim=1)

            # text
            text_emb = textEncoder(caption, cond_mask_prob=0.1)
            text_emb = text_emb.view(text_emb.shape[0], -1, generator_dim)
            logits, _ = transformer(z_indices, text_emb)
            logits = logits[:, text_emb.shape[1]-1:-1]

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i%200==0:
                unify_log.log({'transformer_loss:':loss }, step=epoch*num_batch + i)

        text_emb = textEncoder(caption[-1:], cond_mask_prob=0)
        text_emb = text_emb.view(text_emb.shape[0], -1, generator_dim)
        gen_indices = generate_samples(transformer, opt.device, text_emb, steps=m_lens[-1])
        x = get_model(quantize).embedding(gen_indices.reshape(-1))
        x = x.reshape(*gen_indices.shape, -1)
        x = decoder(x)
        x = x[0].detach().cpu().numpy()
        visualize_2motions(x, std, mean, opt.dataset_name, x.shape[0], 
                            save_path=f'{opt.save_root}/epoch_{epoch}_{caption[-1]}_{m_lens[-1]}.html')
        unify_log.save_model(transformer, 'transformer.pth')
        unify_log.save_model(textEncoder, 'textEncoder.pth')