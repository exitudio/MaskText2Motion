import torch.nn as nn
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from exit.utils import get_model, generate_src_mask
from models.skip_transformer import PositionEmbeddingLearned1D, TransformerEncoderLayer, SkipTransformerEncoder
import torch
import models.pos_encoding as pos_encoding
from models.t2m_trans import Block
import math

class CoderBase(nn.Module):
    def __init__(self, motion_dim = 251, 
                 embed_dim = 512,
                 block_size = 49,
                 n_head = 16,
                 drop_out_rate = .1,
                 num_layers = 3,
                 patch_size = 4):
        super().__init__()
        
        self.block_size = block_size
        self.patch_size = patch_size
        self.n_head = n_head

        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, _length):
        x = self.pos_embed(x)
        src_mask = generate_src_mask(self.block_size, _length)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        for block in self.blocks:
            x = block(x, src_mask)
        return x

class Encoder(CoderBase):
    def __init__(self, motion_dim = 251, 
                 embed_dim = 512,
                 block_size = 49,
                 n_head = 16,
                 drop_out_rate = .1,
                 num_layers = 3,
                 patch_size = 4):
        super().__init__(motion_dim, embed_dim, block_size, n_head, drop_out_rate, num_layers, patch_size)
        self.to_patch_embedding = nn.Conv1d(motion_dim, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.apply(self._init_weights)

    def forward(self, x, m_len):
        x = x.permute(0, 2, 1)
        x = self.to_patch_embedding(x).permute(0, 2, 1)
        x = super().forward(x, m_len)
        return x
    
class Decoder(CoderBase):
    def __init__(self, motion_dim = 251, 
                 embed_dim = 512,
                 block_size = 49,
                 n_head = 16,
                 drop_out_rate = .1,
                 num_layers = 3,
                 patch_size = 4):
        super().__init__(motion_dim, embed_dim, block_size, n_head, drop_out_rate, num_layers, patch_size)

        # follow this: https://github.com/thuanz123/enhancing-transformers/blob/main/enhancing/modules/stage1/layers.py#L204
        self.head = torch.nn.ConvTranspose1d(embed_dim, motion_dim, kernel_size=patch_size, stride=patch_size)
        # self.head = nn.Linear(embed_dim, motion_dim, bias=False)
        self.apply(self._init_weights)

    def forward(self, x, m_token_len):
        x = super().forward(x, m_token_len).permute(0,2,1)
        x = self.head(x).permute(0,2,1)
        return x

def vqvae_wrapper(patch_size, encoder, decoder, quantizer):
    def vqvae(motion, m_len):
        src_mask = generate_src_mask(motion.shape[1], m_len)
        m_token_len = torch.ceil(m_len/patch_size)
        max_token_len = math.ceil(motion.shape[1]/patch_size)
        seq_token_mask = generate_src_mask(max_token_len, m_token_len)

        x_e = encoder(motion, m_len)
        x_d, loss_commit, perplexity  = quantizer(x_e, seq_token_mask)
        m_token_len = torch.ceil(m_len/patch_size)
        pred_motion = decoder(x_d, m_token_len) * src_mask.unsqueeze(-1)

        return pred_motion, x_d, loss_commit, perplexity
    vqvae.eval = lambda: encoder.eval(); decoder.eval(); quantizer.eval()
    vqvae.train = lambda: encoder.train(); decoder.train(); quantizer.train()
    return vqvae


class VQVAE_Transformer(nn.Module):
    """
    https://github.com/Qiyuan-Ge/PaintMind/blob/main/paintmind/stage1/layers.py#L81
    https://github.com/thuanz123/enhancing-transformers/blob/main/enhancing/modules/stage1/layers.py#L168
    https://github.com/lucidrains/parti-pytorch/blob/main/parti_pytorch/vit_vqgan.py#L316
    https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py#L374
    """
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512):
        super().__init__()
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        motion_dim = 251 if args.dataname == 'kit' else 263
        self.patch_size = 4
        self.encoder = Encoder(motion_dim=motion_dim)
        self.decoder = Decoder(motion_dim=motion_dim)
        self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)

    def forward(self, *args, type='full', **kwargs):
        '''type=[full, encode, decode]'''
        if type=='full':
            return self.forward_full(*args, **kwargs)
        elif type=='encode':
            return self.forward_encode(*args, **kwargs)
        elif type=='decode':
            return self.forward_decode(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')

    def forward_full(self, x, m_len):
        m_token_len = torch.ceil(m_len/self.patch_size)
        x_e = self.encoder(x, m_len).permute(0,2,1)
        x_d, commit_loss, perplexity  = self.quantizer(x_e)
        x_d = x_d.permute(0,2,1)
        x = self.decoder(x_d, m_token_len)
        return x, commit_loss, perplexity
    
    def forward_encode(self, x, m_len):
        m_token_len = torch.ceil(m_len/self.patch_size)
        x = self.encoder(x, m_token_len)
        return x
    
    def forward_decode(self, x, m_token_len):
        x = self.decoder(x, m_token_len)
        return x

        