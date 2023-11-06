import torch
import clip
import models.vqvae as vqvae
import models.t2m_trans as trans
import numpy as np


##### ---- CLIP ---- #####
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# https://github.com/openai/CLIP/issues/111
class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        with torch.no_grad():
            word_emb = self.model.token_embedding(text).type(self.model.dtype)
            word_emb = word_emb + self.model.positional_embedding.type(self.model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.model.transformer(word_emb)
            word_emb = self.model.ln_final(word_emb).permute(1, 0, 2).float()
            enctxt = self.model.encode_text(text).float()
        return enctxt, word_emb
clip_model = TextCLIP(clip_model)

def get_vqvae(args):
    return vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate)

def get_maskdecoder(args, vqvae):
    return trans.Text2Motion_Transformer(vqvae,
                                num_vq=args.nb_code, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                num_local_layer=args.num_local_layer, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)

class InstantMotion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        class Temp:
            def __init__(self):
                print('mock:: opt')
        args = Temp()
        args.resume_pth = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32/net_last.pth'
        args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-12-10-11-15_HML3D_45_crsAtt1lyr_40breset_WRONG_THIS_20BRESET/net_last.pth'


        args.dataname = args.dataset_name = 't2m'
        args.nb_code = 8192
        args.code_dim = 32
        args.output_emb_width = 512
        args.down_t = 2
        args.stride_t = 2
        args.width = 512
        args.depth = 3
        args.dilation_growth_rate = 3
        args.quantizer = 'ema_reset'
        args.mu = 0.99
        args.clip_dim = 512

        
        args.embed_dim_gpt = 1024
        args.block_size = 51
        args.num_layers = 9
        args.num_local_layer = 1
        args.n_head_gpt = 16
        args.drop_out_rate = 0.1
        args.ff_rate = 4

        self.vqvae = get_vqvae(args)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
        self.vqvae.eval()
        self.vqvae.cuda()

        self.maskdecoder = get_maskdecoder(args, self.vqvae)
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)
        self.maskdecoder.train()
        self.maskdecoder.cuda()

    def forward(self, text, lengths=-1):
        b = len(text)
        feat_clip_text = clip.tokenize(text, truncate=True).cuda()
        feat_clip_text, word_emb = clip_model(feat_clip_text)
        index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, rand_pos=True)

        m_token_length = torch.ceil((lengths)/4).int()
        pred_pose_all = torch.zeros((b, 196, 263)).cuda()
        for k in range(b):
            pred_pose = self.vqvae(index_motion[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_all[k:k+1, :int(lengths[k].item())] = pred_pose
        return pred_pose_all
    
    def inbetween(self, base_pose, m_length, start_f, end_f, inbetween_text):
        bs, seq = base_pose.shape[:2]
        tokens = -1*torch.ones((bs, 50), dtype=torch.long).cuda()
        m_token_length = torch.ceil((m_length)/4).int()
        end_token = torch.ceil((end_f)/4).int()

        for k in range(bs):
            # start tokens
            index_motion = self.vqvae(base_pose[k:k+1, :start_f[k]].cuda(), type='encode')
            tokens[k,:index_motion.shape[1]] = index_motion[0]

            # end tokens
            index_motion = self.vqvae(base_pose[k:k+1, end_f[k] :m_length[k]].cuda(), type='encode')
            tokens[k, end_token[k] :m_token_length[k]] = index_motion[0]

        text = clip.tokenize(inbetween_text, truncate=True).cuda()
        feat_clip_text, word_emb_clip = clip_model(text)

        mask_id = self.maskdecoder.num_vq + 2
        tokens[tokens==-1] = mask_id
        inpaint_index = self.maskdecoder(feat_clip_text, word_emb_clip, type="sample", m_length=m_length.cuda(), token_cond=tokens)

        pred_pose = torch.zeros((bs, seq, base_pose.shape[-1])).cuda()
        for k in range(bs):
            pred_pose = self.vqvae(inpaint_index[k:k+1, :m_token_length[k]], type='decode')
            pred_pose[k:k+1, :int(m_length[k].item())] = pred_pose
        return pred_pose
    
    def long_range(self, text, lengths):
        # import datetime
        # start_time = datetime.datetime.now()
        b = len(text)
        feat_clip_text = clip.tokenize(text, truncate=True).cuda()
        feat_clip_text, word_emb = clip_model(feat_clip_text)
        index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, rand_pos=False)

        m_token_length = torch.ceil((lengths)/4).int()
        
        half_token_length = (m_token_length/2).int()
        idx_full_len = half_token_length >= 24
        half_token_length[idx_full_len] = half_token_length[idx_full_len] - 1

        mask_id = self.maskdecoder.num_vq + 2
        tokens = -1*torch.ones((b-1, 50), dtype=torch.long).cuda()
        transition_train_length = []
        
        for i in range(b-1):
            left_end = half_token_length[i]
            right_start = left_end + 2
            end = right_start + half_token_length[i+1]

            tokens[i, :left_end] = index_motion[i, m_token_length[i]-left_end: m_token_length[i]]
            tokens[i, left_end:right_start] = mask_id
            tokens[i, right_start:end] = index_motion[i+1, :half_token_length[i+1]]
            transition_train_length.append(end)
        transition_train_length = torch.tensor(transition_train_length).to(index_motion.device)
        text = clip.tokenize(text[:-1], truncate=True).cuda()
        feat_clip_text, word_emb_clip = clip_model(text)
        inpaint_index = self.maskdecoder(feat_clip_text, word_emb_clip, type="sample", m_length=transition_train_length*4, token_cond=tokens, max_steps=1)
        all_tokens = []
        for i in range(b-1):
            all_tokens.append(index_motion[i, :m_token_length[i]])
            all_tokens.append(inpaint_index[i, tokens[i] == mask_id])
        all_tokens.append(index_motion[-1, :m_token_length[-1]])
        all_tokens = torch.cat(all_tokens).unsqueeze(0)
        pred_pose = self.vqvae(all_tokens, type='decode')
        # end_time = datetime.datetime.now()
        # diff_time = (end_time-start_time).total_seconds()
        # print('render time:', diff_time, 'seconds')
        return pred_pose

        