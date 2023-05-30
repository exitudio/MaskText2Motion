import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
from tqdm import tqdm
from exit.utils import generate_src_mask
import math

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'vq/{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_VQ.cycle(train_loader)

val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)

##### ---- Network ---- #####
if args.window_size == -1:
    from models.vqvae_trans import Encoder, Decoder, vqvae_wrapper
    from models.quantize_cnn import QuantizeEMAReset, QuantizeEMAReset_mask
    motion_dim = 251 if args.dataname == 'kit' else 263
    encoder = Encoder(motion_dim=motion_dim)
    encoder = torch.nn.DataParallel(encoder)
    encoder.cuda()
    decoder = Decoder(motion_dim=motion_dim)
    decoder = torch.nn.DataParallel(decoder)
    decoder.cuda()
    patch_size = 4
    quantizer = QuantizeEMAReset_mask(args.nb_code, args.code_dim, args)
    quantizer.cuda()
    vqvae = vqvae_wrapper(patch_size, encoder, decoder, quantizer)
    vqvae.train()
else:
    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm)
    net.train()
    net.cuda()


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(list(encoder.parameters()) + list(quantizer.parameters()) + list(decoder.parameters()), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  

Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in tqdm(range(1, args.warm_up_iter)):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
    if args.window_size == -1:
        gt_motion, m_len = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
        m_len = m_len.cuda()
        pred_motion, x_d, loss_commit, perplexity = vqvae(gt_motion, m_len)
    else:
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
        pred_motion, loss_commit, perplexity = net(gt_motion)
   
    patch_size = 4
    m_token_len = torch.ceil(m_len/patch_size)
    max_token_len = math.ceil(gt_motion.shape[1]/patch_size)
    seq_mask = generate_src_mask(gt_motion.shape[1], m_len)
    seq_token_mask = generate_src_mask(max_token_len, m_token_len)

    # motion loss
    loss_motion = Loss(pred_motion, gt_motion).mean(-1)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion).mean(-1)

    weights = seq_mask / (seq_mask.sum(1).unsqueeze(-1) * seq_mask.shape[0])
    weights = torch.masked_select(weights, seq_mask)
    loss_motion = torch.masked_select(loss_motion, seq_mask)
    loss_vel = torch.masked_select(loss_vel, seq_mask)

    # token loss
    loss_commit = loss_commit.mean(-1)
    token_weights = seq_token_mask / (seq_token_mask.sum(1).unsqueeze(-1) * seq_token_mask.shape[0])
    token_weights = torch.masked_select(token_weights, seq_token_mask)
    # loss_commit = torch.masked_select(loss_commit, seq_token_mask)

    # all loss
    loss_motion = (loss_motion*weights).sum()
    loss_vel = (loss_vel*weights).sum()
    loss_commit = (loss_commit * token_weights).sum()
    loss = loss_motion + args.loss_vel * loss_vel + args.commit * loss_commit
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, vqvae, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper)

for nb_iter in tqdm(range(1, args.total_iter + 1)):
    
    if args.window_size == -1:
        gt_motion, m_len = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
        m_len = m_len.cuda()
        pred_motion, x_d, loss_commit, perplexity = vqvae(gt_motion, m_len)
    else:
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
        pred_motion, loss_commit, perplexity = net(gt_motion)
    
    patch_size = 4
    m_token_len = torch.ceil(m_len/patch_size)
    max_token_len = math.ceil(gt_motion.shape[1]/patch_size)
    seq_mask = generate_src_mask(gt_motion.shape[1], m_len)
    seq_token_mask = generate_src_mask(max_token_len, m_token_len)

    # motion loss
    loss_motion = Loss(pred_motion, gt_motion).mean(-1)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion).mean(-1)

    weights = seq_mask / (seq_mask.sum(1).unsqueeze(-1) * seq_mask.shape[0])
    weights = torch.masked_select(weights, seq_mask)
    loss_motion = torch.masked_select(loss_motion, seq_mask)
    loss_vel = torch.masked_select(loss_vel, seq_mask)


    # token loss
    loss_commit = loss_commit.mean(-1)
    token_weights = seq_token_mask / (seq_token_mask.sum(1).unsqueeze(-1) * seq_token_mask.shape[0])
    token_weights = torch.masked_select(token_weights, seq_token_mask)
    # loss_commit = torch.masked_select(loss_commit, seq_token_mask)

    # all loss
    loss_motion = (loss_motion*weights).sum()
    loss_vel = (loss_vel*weights).sum()
    loss_commit = (loss_commit * token_weights).sum()
    loss = loss_motion + args.loss_vel * loss_vel + args.commit * loss_commit
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if nb_iter % args.eval_iter==0 :
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_vqvae(args.out_dir, val_loader, vqvae, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper)
        