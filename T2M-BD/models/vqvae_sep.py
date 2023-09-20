import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
import torch
from utils.humanml_utils import HML_UPPER_BODY_MASK, HML_LOWER_BODY_MASK

class VQVAE_SEP(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 sep_decoder=False):
        super().__init__()
        if args.dataname == 'kit':
            self.nb_joints = 21
            output_dim = 251
            upper_dim = 120        
            lower_dim = 131  
        else:
            self.nb_joints = 22
            output_dim = 263
            upper_dim = 156        
            lower_dim = 107 
        self.code_dim = code_dim
        # self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        
        # self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.sep_decoder = sep_decoder
        if self.sep_decoder:
            self.decoder_upper = Decoder(upper_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
            self.decoder_lower = Decoder(lower_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        
        else:
            self.decoder = Decoder(output_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        


        self.num_code = nb_code

        self.encoder_upper = Encoder(upper_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_lower = Encoder(lower_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer_upper = QuantizeEMAReset(nb_code, int(code_dim/2), args)
        self.quantizer_lower = QuantizeEMAReset(nb_code, int(code_dim/2), args)

    def rand_emb_idx(self, x_quantized, quantizer, idx_noise):
        # x_quantized = x_quantized.detach()
        x_quantized = x_quantized.permute(0,2,1)
        mask = torch.bernoulli(idx_noise * torch.ones((*x_quantized.shape[:2], 1),
                                                device=x_quantized.device))
        r_indices = torch.randint(int(self.num_code/2), x_quantized.shape[:2], device=x_quantized.device)
        r_emb = quantizer.dequantize(r_indices)
        x_quantized = mask * r_emb + (1-mask) * x_quantized
        x_quantized = x_quantized.permute(0,2,1)
        return x_quantized
    
    def forward(self, x, *args, type='full', **kwargs):
        '''type=[full, encode, decode]'''
        if type=='full':
            upper_emb = get_part_mask(HML_UPPER_BODY_MASK, x)
            lower_emb = get_part_mask(HML_LOWER_BODY_MASK, x)
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, loss_upper, perplexity = self.quantizer_upper(upper_emb)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, loss_lower, perplexity = self.quantizer_lower(lower_emb)
            loss = loss_upper + loss_lower

            if 'idx_noise' in kwargs and kwargs['idx_noise'] > 0:
                upper_emb = self.rand_emb_idx(upper_emb, self.quantizer_upper, kwargs['idx_noise'])
                lower_emb = self.rand_emb_idx(lower_emb, self.quantizer_lower, kwargs['idx_noise'])


            # x_in = self.preprocess(x)
            # x_encoder = self.encoder(x_in)
        
            # ## quantization
            # x_quantized, loss, perplexity  = self.quantizer(x_encoder)

            ## decoder
            if self.sep_decoder:
                x_decoder_upper = self.decoder_upper(upper_emb)
                x_decoder_upper = self.postprocess(x_decoder_upper)
                x_decoder_lower = self.decoder_lower(lower_emb)
                x_decoder_lower = self.postprocess(x_decoder_lower)
                x_out = merge_upper_lower(x_decoder_upper, x_decoder_lower)
            else:
                x_quantized = torch.cat([upper_emb, lower_emb], dim=1)
                x_decoder = self.decoder(x_quantized)
                x_out = self.postprocess(x_decoder)
            
            return x_out, loss, perplexity
        elif type=='encode':
            N, T, _ = x.shape
            upper_emb = get_part_mask(HML_UPPER_BODY_MASK, x)
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb = self.postprocess(upper_emb)
            upper_emb = upper_emb.reshape(-1, upper_emb.shape[-1])
            upper_code_idx = self.quantizer_upper.quantize(upper_emb)
            upper_code_idx = upper_code_idx.view(N, -1)

            lower_emb = get_part_mask(HML_LOWER_BODY_MASK, x)
            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb = self.postprocess(lower_emb)
            lower_emb = lower_emb.reshape(-1, lower_emb.shape[-1])
            lower_code_idx = self.quantizer_lower.quantize(lower_emb)
            lower_code_idx = lower_code_idx.view(N, -1)

            code_idx = torch.cat([upper_code_idx.unsqueeze(-1), lower_code_idx.unsqueeze(-1)], dim=-1)
            return code_idx

        elif type=='decode':
            if self.sep_decoder:
                x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
                x_d_upper = x_d_upper.permute(0, 2, 1).contiguous()
                x_d_upper = self.decoder_upper(x_d_upper)
                x_d_upper = self.postprocess(x_d_upper)

                x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
                x_d_lower = x_d_lower.permute(0, 2, 1).contiguous()
                x_d_lower = self.decoder_lower(x_d_lower)
                x_d_lower = self.postprocess(x_d_lower)
            
                x_out = merge_upper_lower(x_d_upper, x_d_lower)
                return x_out
            else:
                x_d_upper = self.quantizer_upper.dequantize(x[..., 0])
                x_d_lower = self.quantizer_lower.dequantize(x[..., 1])
                x_d = torch.cat([x_d_upper, x_d_lower], dim=-1)
                x_d = x_d.permute(0, 2, 1).contiguous()
                x_decoder = self.decoder(x_d)
                x_out = self.postprocess(x_decoder)
                return x_out

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x
    
    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


def get_part_mask(PART, motion):
    mask_part = torch.tensor(PART, dtype=torch.bool, device=motion.device)
    mask_part = mask_part.repeat(*motion.shape[:2], 1)
    emb_part = motion[mask_part].reshape(*motion.shape[:2], -1)
    return emb_part

def set_part_mask(PART, motion, part_emb):
    mask_part = torch.tensor(PART, dtype=torch.bool, device=motion.device)
    mask_part = mask_part.repeat(*motion.shape[:2], 1)
    motion[mask_part] = part_emb.reshape(-1)
    return motion

def merge_upper_lower(upper_emb, lower_emb):
    motion = torch.empty(*upper_emb.shape[:2], 263).to(upper_emb.device)
    motion = set_part_mask(HML_UPPER_BODY_MASK, motion, upper_emb)
    motion = set_part_mask(HML_LOWER_BODY_MASK, motion, lower_emb)
    return motion

def upper_lower_sep(motion, joints_num):
    # root
    _root = motion[..., :4] # root

    # position
    start_indx = 1 + 2 + 1
    end_indx = start_indx + (joints_num - 1) * 3
    positions = motion[..., start_indx:end_indx]
    positions = positions.view(*motion.shape[:2], (joints_num - 1), 3)

    # 6drot
    start_indx = end_indx
    end_indx = start_indx + (joints_num - 1) * 6
    _6d_rot = motion[..., start_indx:end_indx]
    _6d_rot = _6d_rot.view(*motion.shape[:2], (joints_num - 1), 6)

    # joint_velo
    start_indx = end_indx
    end_indx = start_indx + joints_num * 3
    joint_velo = motion[..., start_indx:end_indx]
    joint_velo = joint_velo.view(*motion.shape[:2], joints_num, 3)

    # foot_contact
    foot_contact = motion[..., end_indx:]

    ################################################################################################
    #### Lower Body
    if joints_num == 22:
        lower_body = torch.tensor([0,1,2,4,5,7,8,10,11])
    else:
        lower_body = torch.tensor([0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    lower_body_exclude_root = lower_body[1:] - 1

    LOW_positions = positions[:,:, lower_body_exclude_root].view(*motion.shape[:2], -1)
    LOW_6d_rot = _6d_rot[:,:, lower_body_exclude_root].view(*motion.shape[:2], -1)
    LOW_joint_velo = joint_velo[:,:, lower_body].view(*motion.shape[:2], -1)
    lower_emb = torch.cat([_root, LOW_positions, LOW_6d_rot, LOW_joint_velo, foot_contact], dim=-1)

    #### Upper Body
    if joints_num == 22:
        upper_body = torch.tensor([3,6,9,12,13,14,15,16,17,18,19,20,21])
    else:
        upper_body = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    upper_body_exclude_root = upper_body - 1

    UP_positions = positions[:,:, upper_body_exclude_root].view(*motion.shape[:2], -1)
    UP_6d_rot = _6d_rot[:,:, upper_body_exclude_root].view(*motion.shape[:2], -1)
    UP_joint_velo = joint_velo[:,:, upper_body].view(*motion.shape[:2], -1)
    upper_emb = torch.cat([UP_positions, UP_6d_rot, UP_joint_velo], dim=-1)

    return upper_emb, lower_emb