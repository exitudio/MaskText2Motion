import torch.nn as nn
from models.quantize_cnn import QuantizeEMAReset
from exit.motiontransformer import MotionTransformerOnly2, generate_src_mask

class VQVAETransformer(nn.Module):
    def __init__(self,
                 args):
        super().__init__()
        self.dim_pose = 251 if args.dataname == 'kit' else 263
        code_dim = args.code_dim
        self.num_heads = 8
        self.encoder = MotionTransformerOnly2(input_feats=self.dim_pose, 
                                            output_feats=code_dim, 
                                            latent_dim=256, 
                                            num_layers=4)
        self.decoder = MotionTransformerOnly2(input_feats=code_dim, 
                                            output_feats=self.dim_pose, 
                                            latent_dim=256, 
                                            num_layers=4)
        self.quantizer = QuantizeEMAReset(args.nb_code, code_dim, args)

    def forward(self, x, src_mask, type='full'):
        '''type=[full, encode, decode]'''
        if type=='full':
            return self.full_forward(x, src_mask)
        elif type=='encode':
            return self.forward_encode(x, src_mask)
        elif type=='decode':
            return self.forward_decode(x, src_mask)
        else:
            raise ValueError(f'Unknown "{type}" type')

    def full_forward(self, x, src_mask):
        B, T, C = x.shape
        src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, self.num_heads, T, 1)

        z = self.encoder(x, src_mask=src_mask_attn) * src_mask

        ########## Lifted from quantizer ##########
        _z = z.view(-1, z.shape[-1])
        if self.training and not self.quantizer.init:
            self.quantizer.init_codebook(_z)
        code_idx = self.quantizer.quantize(_z)
        z_q = self.quantizer.dequantize(code_idx)
        # Update embeddings
        # [TODO] perplexity may not correctly calculated due to DP
        if self.training:
            perplexity = self.quantizer.update_codebook(_z, code_idx, src_mask)
        else : 
            perplexity = self.quantizer.compute_perplexity(code_idx)
        z_q = z_q.view(B, T, -1) * src_mask
        z_q = z + (z_q - z).detach()
        z_q = z_q * src_mask
        ##################################################

        x_recon = self.decoder(z_q, src_mask=src_mask_attn) * src_mask
        return x_recon, perplexity, z, z_q
    
    def forward_encode(self, x, src_mask):
        B, T, C = x.shape
        src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, self.num_heads, T, 1)

        z = self.encoder(x, src_mask=src_mask_attn) * src_mask
        _z = z.view(-1, z.shape[-1])
        code_idx = self.quantizer.quantize(_z)
        code_idx = code_idx.view(B, T)
        return code_idx

    def forward_decode(self, code_idx, src_mask):
        B, T = code_idx.shape
        src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, self.num_heads, T, 1)
        
        z_q = self.quantizer.dequantize(code_idx)
        z_q = z_q.view(B, T, -1) * src_mask
        x_recon = self.decoder(z_q, src_mask=src_mask_attn) * src_mask
        return x_recon