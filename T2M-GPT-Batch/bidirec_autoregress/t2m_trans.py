import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from exit.utils import generate_src_mask
from bidirec_autoregress.util import get_bidirec_input

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.n_head = n_head
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if type=='forward':
            return self.forward_function(*args, **kwargs)
        elif type=='sample':
            return self.sample(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')

    def forward_function(self, idxs, clip_feature, m_tokens_len, max_len):
        src_mask = get_attn_mask(m_tokens_len, max_len, self.n_head)
        feat = self.trans_base(idxs, clip_feature, m_tokens_len, src_mask)
        logits = self.trans_head(feat, src_mask)
        return logits

    def sample(self, clip_feature, m_tokens_len, max_len, if_categorial=False):
        training = self.training 
        self.eval()

        idx = torch.zeros(m_tokens_len.shape[0], max_len, dtype=torch.int, device=clip_feature.device)
        # [TODO] double check generating all tokens the same as picking 1 by 1
        max_steps = torch.floor(m_tokens_len.max()/2)
        for k in range(max_steps.int().item()):
            idx = get_bidirec_input(idx.squeeze(-1), m_tokens_len)
            logits = self.forward(idx, clip_feature, m_tokens_len, max_len)
            # logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                # if idx == self.num_vq:
                #     break
                # idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                idx = idx.squeeze(-1)
                # if idx[0] == self.num_vq:
                #     break
            # append to the sequence and continue
            # if k == 0:
            #     xs = idx
            # else:
            #     xs = torch.cat((xs, idx), dim=1)
            
            # if k == self.block_size - 1:
            #     return xs[:, :-1]
        if training:
            self.train()
        return idx

class Attention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.n_head = n_head

    def forward(self, x, src_mask):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if src_mask is not None:
            att[~src_mask] = float('-inf')
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln1(x), src_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq + 1, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.cond_emb2 = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, m_tokens_len, src_mask):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        origi_idx = idx
        idx = idx.clone()
        idx[idx==-1] = self.cond_emb.weight.shape[1]
        idx[idx==-2] = self.cond_emb.weight.shape[1]
        token_embeddings = self.tok_emb(idx)
        token_embeddings[origi_idx==-1] = self.cond_emb(clip_feature)
        token_embeddings[origi_idx==-2] = self.cond_emb2(clip_feature)

        x = self.pos_embed(token_embeddings)
        for block in self.blocks:
            x = block(x, src_mask)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def repeat_row_col(src_mask, n_head, B, T):
    mask_row = src_mask.view(B, 1, 1, T).repeat(1, n_head, T, 1)
    mask_column = src_mask.view(B, 1, T, 1).repeat(1, n_head, 1, T)
    return mask_row * mask_column

def get_attn_mask(m_tokens_len, T, n_head):
    B = m_tokens_len.shape[0]
    ones_temp = torch.ones(B, T, T, 
                        device=m_tokens_len.device, 
                        dtype=torch.bool)

    # idx bottom from to
    arange_idx = torch.arange(T).repeat(B, 1).to(m_tokens_len.device)
    bottom_mask_from = (arange_idx > (T-m_tokens_len-1).unsqueeze(-1))
    bottom_mask_from = bottom_mask_from.unsqueeze(-1).repeat(1, 1, T)
    bottom_mask_to = (arange_idx < m_tokens_len.unsqueeze(-1))
    bottom_mask_to = bottom_mask_to.unsqueeze(-1).repeat(1, 1, T)

    # Left    
    top_left = torch.tril(ones_temp, diagonal=0)
    bottom_left = torch.zeros_like(ones_temp) # need to create zeros as a placeholder, if not the prev thril on the bottom will show up
    bottom_left_tril = torch.flip(top_left, dims=(1,))
    bottom_left[bottom_mask_to] = bottom_left_tril[bottom_mask_from]
    left = top_left * bottom_left

    # Right
    bottom_right = ~torch.tril(ones_temp, diagonal=-1)
    top_right = torch.flip(bottom_right, dims=(1,))
    top_right[bottom_mask_to] = top_right[bottom_mask_from]
    right = top_right * bottom_right
    
    # Len
    len_mask = generate_src_mask(T, m_tokens_len)[:, None, :].repeat(1, T, 1)
    all = torch.logical_or(left, right) * len_mask

    # first and last col
    all[:,:, 0] = 1 # Actually no need but just add first row to avoid 0 all columns
    # end_idx = m_tokens_len[:, None, None].repeat(1,T,1)-1
    # all.scatter_(dim=-1, index=end_idx, value=1)
    return all.unsqueeze(1).repeat(1, n_head, 1, 1)

    ## old: only top-left & bottom right
    B = m_tokens_len.shape[0]
    half_m_tokens_len = torch.ceil(m_tokens_len/2)

    ones_temp = torch.ones(B, T, T, 
                        device=m_tokens_len.device, 
                        dtype=torch.bool)
    top_tril = torch.tril(ones_temp, diagonal=0)
    bottom_tril = ~torch.tril(ones_temp, diagonal=-1)
    top_tril = top_tril.view(B, 1, T, T).repeat(1, n_head, 1, 1)
    bottom_tril = bottom_tril.view(B, 1, T, T).repeat(1, n_head, 1, 1)

    top_mask = torch.arange(T).repeat(B, 1).to(half_m_tokens_len.device) < half_m_tokens_len.unsqueeze(-1)
    top_mask = repeat_row_col(top_mask, n_head, B, T)

    bottom_mask = torch.arange(T).repeat(B, 1).to(half_m_tokens_len.device) >= (m_tokens_len - half_m_tokens_len).unsqueeze(-1)
    bottom_mask = repeat_row_col(bottom_mask, n_head, B, T)

    len_mask = generate_src_mask(T, m_tokens_len)
    len_mask = repeat_row_col(len_mask, n_head, B, T)

    all_mask = (top_mask*top_tril + bottom_mask*bottom_tril) * len_mask

    # first and last col
    all_mask[:,:, :, 0] = 1
    end_idx = m_tokens_len[:, None, None, None].repeat(1,1,T, 1)-1
    all_mask.scatter_(dim=-1, index=end_idx, value=1)
    return all_mask
