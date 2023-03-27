import torch.nn as nn
from models.quantize_cnn import QuantizeEMAReset
from exit.motiontransformer import MotionTransformerOnly2, generate_src_mask

class VQVAETransformer(nn.Module):
    def __init__(self,
                 args):
        super().__init__()
        self.dim_pose = 251 if args.dataname == 'kit' else 263

        self.encoder = MotionTransformerOnly2(input_feats=self.dim_pose, 
                                            output_feats=args.code_dim, 
                                            latent_dim=256, 
                                            num_layers=4)
        self.decoder = MotionTransformerOnly2(input_feats=args.code_dim, 
                                            output_feats=self.dim_pose, 
                                            latent_dim=256, 
                                            num_layers=4)
        self.quantizer = QuantizeEMAReset(args.nb_code, args.code_dim, args)

    def forward(self, x, src_mask):
        B, T, C = x.shape
        num_heads = 8
        src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, num_heads, T, 1)

        z = self.encoder(x, src_mask=src_mask_attn) * src_mask

        ########## Lifted from quantizer ##########
        _z = z.view(-1, z.shape[-1])
        if self.training and not self.quantizer.init:
            self.quantizer.init_codebook(_z)
        code_idx = self.quantizer.quantize(_z)
        z_q = self.quantizer.dequantize(code_idx)
        # Update embeddings
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

    def encode(self, x, src_mask):
        B, T, C = x.shape
        num_heads = 8
        src_mask_attn = src_mask.view(B, 1, 1, T).repeat(1, num_heads, T, 1)

        z = self.encoder(x, src_mask=src_mask_attn) * src_mask
        _z = z.view(-1, z.shape[-1])
        code_idx = self.quantizer.quantize(_z)
        return code_idx
