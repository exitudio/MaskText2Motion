import torch
import clip
import models.vqvae as vqvae
from models.vqvae_sep import VQVAE_SEP
import models.t2m_trans as trans
import models.t2m_trans_uplow as trans_uplow
import numpy as np
import math


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

def get_vqvae(args, is_upper_edit):
    if not is_upper_edit:
        return vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                            args.nb_code,
                            args.code_dim,
                            args.output_emb_width,
                            args.down_t,
                            args.stride_t,
                            args.width,
                            args.depth,
                            args.dilation_growth_rate)
    else:
        return VQVAE_SEP(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        moment={'mean': torch.from_numpy(args.mean).cuda().float(), 
                            'std': torch.from_numpy(args.std).cuda().float()},
                        sep_decoder=True)

def get_maskdecoder(args, vqvae, is_upper_edit):
    tranformer = trans if not is_upper_edit else trans_uplow
    return tranformer.Text2Motion_Transformer(vqvae,
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
    def __init__(self, is_upper_edit=False, extra_args=None):
        super().__init__()
        self.is_upper_edit = is_upper_edit

        class Temp:
            def __init__(self, extra_args):
                print('mock:: opt')
                if extra_args is not None:
                    for i in extra_args:
                        self.__dict__[i] = extra_args[i]
        args = Temp(extra_args)
        if not is_upper_edit:
            args.resume_pth = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-07-19-04-17-17_12_VQVAE_20batchResetNRandom_8192_32/net_last.pth'
            # w/ txt
            # args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-12-10-11-15_HML3D_45_crsAtt1lyr_40breset_WRONG_THIS_20BRESET/net_last.pth'
            # w/o txt
            args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-15-09-13-42_HML3D_46_crsAtt1lyr_20breset_.1DrpTxt/net_last.pth'
            args.num_local_layer = 1
        else:
            args.resume_pth = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/vq/2023-10-04-07-27-56_29_VQVAE_uplow_sepDec_moveUpperDown/net_last.pth'
            # with Txt
            # args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-29-11-24-16_HML3D_44_upperEdit_transMaskLower_moveUpperDown_1crsAttn_noRandLen/net_last.pth'
            # with Txt + ranLen
            # args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-22-11-12-22_HML3D_44_upperEdit_transMaskLower_randLen_moveUpperDown_1crsAttn/net_last.pth'
            # no txt
            # args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-11-01-23-04-58_HML3D_44_upperEdit_transMaskLower_moveUpperDown_1crsAttn_noRandLen_dropTxt.1/net_last.pth'
            # args.num_local_layer = 1

            # txt randLen localLayer0
            args.resume_trans = '/home/epinyoan/git/MaskText2Motion/T2M-BD/output/t2m/2023-10-05-09-21-54_HML3D_43_upperEdit_transMaskLower_randLen_moveUpperDown/net_last.pth'
            args.num_local_layer = 0

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
        args.n_head_gpt = 16
        args.drop_out_rate = 0.1
        args.ff_rate = 4

        self.vqvae = get_vqvae(args, is_upper_edit)
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        self.vqvae.load_state_dict(ckpt['net'], strict=True)
        if is_upper_edit:
            class VQVAE_WRAPPER(torch.nn.Module):
                def __init__(self, vqvae) :
                    super().__init__()
                    self.vqvae = vqvae
                    
                def forward(self, *args, **kwargs):
                    return self.vqvae(*args, **kwargs)
            self.vqvae = VQVAE_WRAPPER(self.vqvae)
        self.vqvae.eval()
        self.vqvae.cuda()

        self.maskdecoder = get_maskdecoder(args, self.vqvae, is_upper_edit)
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        self.maskdecoder.load_state_dict(ckpt['trans'], strict=True)
        self.maskdecoder.train()
        self.maskdecoder.cuda()

    def forward(self, text, lengths=-1, rand_pos=True):
        b = len(text)
        feat_clip_text = clip.tokenize(text, truncate=True).cuda()
        feat_clip_text, word_emb = clip_model(feat_clip_text)
        index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, rand_pos=rand_pos, if_test=False)

        m_token_length = torch.ceil((lengths)/4).int()
        pred_pose_all = torch.zeros((b, 196, 263)).cuda()
        for k in range(b):
            pred_pose = self.vqvae(index_motion[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_all[k:k+1, :int(lengths[k].item())] = pred_pose
        return pred_pose_all
    
    def anytime_paint(self, base_pose, m_length, when_paint, inpaint_text):
        bs, seq = base_pose.shape[:2]
        tokens = -1*torch.ones((bs, int(seq/4)), dtype=torch.long).cuda()
        m_token_length = torch.ceil((m_length)/4).int()

        for i in range(bs):
            index_motion = self.vqvae(base_pose[i:i+1, :m_length[i]].cuda(), type='encode')
            token_len = index_motion.shape[1]
            condition = ~when_paint[i, :token_len] # for the case base_pose is shorter than 196
            tokens[i, :token_len][condition] = index_motion[0, condition]
        tokens = torch.cat([tokens, -1 * torch.ones(bs, 1, dtype=int).cuda()], dim=1)
            
        text = clip.tokenize(inpaint_text, truncate=True).cuda()
        feat_clip_text, word_emb_clip = clip_model(text)

        mask_id = self.maskdecoder.num_vq + 2
        tokens[tokens==-1] = mask_id
        inpaint_index = self.maskdecoder(feat_clip_text, word_emb_clip, type="sample", m_length=m_length.cuda(), token_cond=tokens)

        pred_pose_eval = torch.zeros((bs, seq, base_pose.shape[-1])).cuda()
        for k in range(bs):
            pred_pose = self.vqvae(inpaint_index[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_eval[k:k+1, :int(m_length[k].item())] = pred_pose
        return pred_pose_eval
    
    def inbetween(self, base_pose, m_length, start_f, end_f, inbetween_text):
        bs, seq = base_pose.shape[:2]
        token_len = torch.ceil((m_length)/4).int()
        when_paint = torch.zeros((base_pose.shape[0], token_len), dtype=bool)
        for i in range(bs):
            when_paint[i, int(start_f[i]/4):int(end_f[i]/4)] = True
        return self.anytime_paint(base_pose, m_length, when_paint, inbetween_text)
    
    def inbetween_eval(self, base_pose, m_length, start_f, end_f, inbetween_text):
        bs, seq = base_pose.shape[:2]
        tokens = -1*torch.ones((bs, 50), dtype=torch.long).cuda()
        m_token_length = torch.ceil((m_length)/4).int()
        start_t = torch.round((start_f)/4).int()
        end_t = torch.round((end_f)/4).int()

        for k in range(bs):
            index_motion = self.vqvae(base_pose[k:k+1, :m_length[k]].cuda(), type='encode')
            tokens[k, :start_t[k]] = index_motion[0][:start_t[k]]
            tokens[k, end_t[k]:m_token_length[k]] = index_motion[0][end_t[k]:m_token_length[k]]

            # # start tokens
            # index_motion = self.vqvae(base_pose[k:k+1, :start_f[k]].cuda(), type='encode')
            # tokens[k,:index_motion.shape[1]] = index_motion[0]

            # # end tokens
            # index_motion = self.vqvae(base_pose[k:k+1, end_f[k] :m_length[k]].cuda(), type='encode')
            # tokens[k, end_t[k] :m_token_length[k]] = index_motion[0]

        text = clip.tokenize(inbetween_text, truncate=True).cuda()
        feat_clip_text, word_emb_clip = clip_model(text)

        mask_id = self.maskdecoder.num_vq + 2
        tokens[tokens==-1] = mask_id
        inpaint_index = self.maskdecoder(feat_clip_text, word_emb_clip, type="sample", m_length=m_length.cuda(), token_cond=tokens)

        pred_pose_eval = torch.zeros((bs, seq, base_pose.shape[-1])).cuda()
        for k in range(bs):
            pred_pose = self.vqvae(inpaint_index[k:k+1, :m_token_length[k]], type='decode')
            pred_pose_eval[k:k+1, :int(m_length[k].item())] = pred_pose
        return pred_pose_eval
    
    def long_range(self, text, lengths, num_transition_token=2, output='concat'):
        # assert num_transition_token%2==0
        # import datetime
        # start_time = datetime.datetime.now()
        b = len(text)
        feat_clip_text = clip.tokenize(text, truncate=True).cuda()
        feat_clip_text, word_emb = clip_model(feat_clip_text)
        index_motion = self.maskdecoder(feat_clip_text, word_emb, type="sample", m_length=lengths, rand_pos=False)

        m_token_length = torch.ceil((lengths)/4).int()
        if output == 'eval':
            frame_length = m_token_length * 4
            m_token_length = m_token_length.clone()
            m_token_length = m_token_length - 2*num_transition_token
            m_token_length[[0,-1]] += num_transition_token # first and last have transition only half
        
        half_token_length = (m_token_length/2).int()
        idx_full_len = half_token_length >= 24
        half_token_length[idx_full_len] = half_token_length[idx_full_len] - 1

        mask_id = self.maskdecoder.num_vq + 2
        tokens = -1*torch.ones((b-1, 50), dtype=torch.long).cuda()
        transition_train_length = []
        
        for i in range(b-1):
            if output == 'concat':
                i_index_motion = index_motion[i]
                i1_index_motion = index_motion[i+1]
            if output == 'eval':
                if i == 0:
                    i_index_motion = index_motion[i, :m_token_length[i]]
                else:
                    i_index_motion = index_motion[i, num_transition_token:m_token_length[i] + num_transition_token]
                if i == b-1:
                    i1_index_motion = index_motion[i+1, :m_token_length[i+1]]
                else:
                    i1_index_motion = index_motion[i+1, 
                                                num_transition_token:m_token_length[i+1] + num_transition_token]
            left_end = half_token_length[i]
            right_start = left_end + num_transition_token
            end = right_start + half_token_length[i+1]

            tokens[i, :left_end] = i_index_motion[m_token_length[i]-left_end: m_token_length[i]]
            tokens[i, left_end:right_start] = mask_id
            tokens[i, right_start:end] = i1_index_motion[:half_token_length[i+1]]
            transition_train_length.append(end)
        transition_train_length = torch.tensor(transition_train_length).to(index_motion.device)
        text = clip.tokenize(text[:-1], truncate=True).cuda()
        feat_clip_text, word_emb_clip = clip_model(text)
        inpaint_index = self.maskdecoder(feat_clip_text, word_emb_clip, type="sample", m_length=transition_train_length*4, token_cond=tokens, max_steps=1)
        
        if output == 'concat':
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
        elif output == 'eval':
            all_tokens = []
            for i in range(b):
                motion_token = index_motion[i, :m_token_length[i]]
                if i == 0:
                    first_current_trans_tok = inpaint_index[i, tokens[i] == mask_id]
                    all_tokens.append(motion_token)
                    all_tokens.append(first_current_trans_tok)
                else:
                    if i < b-1:
                        first_current_trans_tok = inpaint_index[i, tokens[i] == mask_id]
                        all_tokens.append(motion_token)
                        all_tokens.append(first_current_trans_tok)
                    else:
                        all_tokens.append(motion_token)
            all_tokens = torch.cat(all_tokens)
            pred_pose_concat = self.vqvae(all_tokens.unsqueeze(0), type='decode')
            
            trans_frame = num_transition_token*4
            pred_pose = torch.zeros((b, 196, 263)).cuda()
            current_point = 0
            for i in range(b):
                if i == 0:
                    start_f = torch.tensor(0)
                    end_f = frame_length[i]
                else:
                    start_f = current_point - trans_frame
                    end_f = start_f + frame_length[i]
                current_point = end_f
                pred_pose[i, :frame_length[i]] = pred_pose_concat[0, start_f: end_f]
            return pred_pose


            # pred_pose = torch.zeros((b, 196, 263)).cuda()
            # for i in range(b):
            #     motion_token = index_motion[i, :m_token_length[i]]
            #     if i == 0:
            #         first_current_trans_tok = inpaint_index[i, tokens[i] == mask_id]
            #         token = torch.cat((motion_token, first_current_trans_tok))
            #     else:
            #         last_prev_trans_tok = inpaint_index[i-1, tokens[i-1] == mask_id]
            #         if i < b-1:
            #             first_current_trans_tok = inpaint_index[i, tokens[i] == mask_id]
            #             token = torch.cat((last_prev_trans_tok, motion_token, first_current_trans_tok))
            #         else:
            #             token = torch.cat((last_prev_trans_tok, motion_token,))
            #     __pred_pose = self.vqvae(token.unsqueeze(0), type='decode')
            #     pred_pose[i, :__pred_pose.shape[1]] = __pred_pose[0]
            # return pred_pose
    
    def upper_edit(self, pose, m_length, upper_text, lower_mask=None):
        pose = pose.clone().cuda().float() # bs, nb_joints, joints_dim, seq_len
        m_tokens_len = torch.ceil((m_length)/4)
        bs, seq = pose.shape[:2]
        max_motion_length = int(seq/4) + 1
        mot_end_idx = self.vqvae.vqvae.num_code
        mot_pad_idx = self.vqvae.vqvae.num_code + 1
        mask_id = self.vqvae.vqvae.num_code + 2
        target_lower = []
        for k in range(bs):
            target = self.vqvae(pose[k:k+1, :m_length[k]], type='encode')
            if m_tokens_len[k]+1 < max_motion_length:
                target = torch.cat([target, 
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx, 
                                    torch.ones((1, max_motion_length-1-m_tokens_len[k].int().item(), 2), dtype=int, device=target.device) * mot_pad_idx], axis=1)
            else:
                target = torch.cat([target, 
                                    torch.ones((1, 1, 2), dtype=int, device=target.device) * mot_end_idx], axis=1)
            target_lower.append(target[..., 1])
        target_lower = torch.cat(target_lower, axis=0)

        ### lower mask ###
        if lower_mask is not None:
            lower_mask = torch.cat([lower_mask, torch.zeros(bs, 1, dtype=int)], dim=1).bool()
            target_lower_masked = target_lower.clone()
            target_lower_masked[lower_mask] = mask_id
            select_end = target_lower == mot_end_idx
            target_lower_masked[select_end] = target_lower[select_end]
        else:
            target_lower_masked = target_lower
        ##################

        pred_len = m_length.cuda()
        pred_tok_len = m_tokens_len
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()

        # __upper_text__ = ['A man punches with right hand.'] * 32
        text = clip.tokenize(upper_text, truncate=True).cuda()
        feat_clip_text, word_emb_clip = clip_model(text)
        # index_motion = trans_encoder(feat_clip_text, idx_lower=target_lower_masked, word_emb=word_emb_clip, type="sample", m_length=pred_len, rand_pos=True, CFG=-1)
        index_motion = self.maskdecoder(feat_clip_text, target_lower_masked, word_emb_clip, type="sample", m_length=pred_len, rand_pos=True)
        for i in range(bs):
            all_tokens = torch.cat([
                index_motion[i:i+1, :int(pred_tok_len[i].item()), None],
                target_lower[i:i+1, :int(pred_tok_len[i].item()), None]
            ], axis=-1)
            pred_pose = self.vqvae(all_tokens, type='decode')
            pred_pose_eval[i:i+1, :int(pred_len[i].item())] = pred_pose

        return pred_pose_eval