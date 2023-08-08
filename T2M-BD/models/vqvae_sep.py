import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.t2m_trans import Decoder_Transformer, Encoder_Transformer
from exit.utils import generate_src_mask
import torch

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
                 norm=None):
        super().__init__()
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        output_dim = 251 if args.dataname == 'kit' else 263
        # self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        
        # self.encoder = Encoder(output_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(output_dim, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)        

        upper_dim = 156        
        lower_dim = 107     
        self.encoder_upper = Encoder(upper_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.encoder_lower = Encoder(lower_dim, int(code_dim/2), down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.quantizer_upper = QuantizeEMAReset(nb_code, int(code_dim/2), args)
        self.quantizer_lower = QuantizeEMAReset(nb_code, int(code_dim/2), args)

    def forward(self, x, type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            upper_emb, lower_emb = upper_lower_sep(x)
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb, loss_upper, perplexity = self.quantizer_upper(upper_emb)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb, loss_lower, perplexity = self.quantizer_lower(lower_emb)
            loss = loss_upper + loss_lower

            x_quantized = torch.cat([upper_emb, lower_emb], dim=1)

            # x_in = self.preprocess(x)
            # x_encoder = self.encoder(x_in)
        
            # ## quantization
            # x_quantized, loss, perplexity  = self.quantizer(x_encoder)

            ## decoder
            x_decoder = self.decoder(x_quantized)
            x_out = self.postprocess(x_decoder)
            
            return x_out, loss, perplexity
        elif type=='encode':
            N, T, _ = x.shape
            upper_emb, lower_emb = upper_lower_sep(x)
            
            upper_emb = self.preprocess(upper_emb)
            upper_emb = self.encoder_upper(upper_emb)
            upper_emb = self.postprocess(upper_emb)
            upper_emb = upper_emb.reshape(-1, upper_emb.shape[-1])
            upper_code_idx = self.quantizer_upper.quantize(upper_emb)
            upper_code_idx = upper_code_idx.view(N, -1)

            lower_emb = self.preprocess(lower_emb)
            lower_emb = self.encoder_lower(lower_emb)
            lower_emb = self.postprocess(lower_emb)
            lower_emb = lower_emb.reshape(-1, lower_emb.shape[-1])
            lower_code_idx = self.quantizer_lower.quantize(lower_emb)
            lower_code_idx = lower_code_idx.view(N, -1)

            code_idx = torch.cat([upper_code_idx.unsqueeze(-1), lower_code_idx.unsqueeze(-1)], dim=-1)
            return code_idx

        elif type=='decode':
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

def upper_lower_sep(motion):
    joints_num = 22

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
    lower_body = torch.tensor([0,1,2,4,5,7,8,10,11])
    lower_body_exclude_root = lower_body[1:] - 1

    LOW_positions = positions[:,:, lower_body_exclude_root].view(*motion.shape[:2], -1)
    LOW_6d_rot = _6d_rot[:,:, lower_body_exclude_root].view(*motion.shape[:2], -1)
    LOW_joint_velo = joint_velo[:,:, lower_body].view(*motion.shape[:2], -1)
    lower_emb = torch.cat([_root, LOW_positions, LOW_6d_rot, LOW_joint_velo, foot_contact], dim=-1)

    #### Upper Body
    upper_body = torch.tensor([3,6,9,12,13,14,15,16,17,18,19,20,21])
    upper_body_exclude_root = upper_body - 1

    UP_positions = positions[:,:, upper_body_exclude_root].view(*motion.shape[:2], -1)
    UP_6d_rot = _6d_rot[:,:, upper_body_exclude_root].view(*motion.shape[:2], -1)
    UP_joint_velo = joint_velo[:,:, upper_body].view(*motion.shape[:2], -1)
    upper_emb = torch.cat([UP_positions, UP_6d_rot, UP_joint_velo], dim=-1)

    return upper_emb, lower_emb