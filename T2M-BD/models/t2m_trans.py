import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from exit.utils import cosine_schedule, uniform, top_k, gumbel_sample
from tqdm import tqdm
from einops import rearrange, repeat
from exit.utils import get_model, generate_src_mask

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
        
    def get_attn_mask(self, src_mask):
        src_mask = torch.cat([torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device),
                                src_mask],  dim=1)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward_function(self, idxs, clip_feature, src_mask=None):
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask)
        feat = self.trans_base(idxs, clip_feature, src_mask)
        logits = self.trans_head(feat, src_mask)
        return logits

    def sample(self, clip_feature, m_length=None, if_test=False):
        max_steps = 18
        max_length = 50
        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, self.block_size - 1)
        topk_filter_thres = .9
        starting_temperature = 1.0
        scores = torch.zeros(shape, dtype = torch.float32, device = clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length)/4)
        src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)
        src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len)
        ids = torch.full(shape, mask_id, dtype = torch.long, device = clip_feature.device)
        
        # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id

        ### PlayGround ####
        # score high = mask
        # m_tokens_len = torch.ceil((m_length)/4)
        # src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1)

        # # mock
        # timestep = torch.tensor(.5)
        # rand_mask_prob = cosine_schedule(timestep)
        # scores = torch.arange(self.block_size - 1).repeat(batch_size, 1).cuda()
        # scores[1] = torch.flip(torch.arange(self.block_size - 1), dims=(0,))

        # # iteration
        # num_token_masked = (rand_mask_prob * m_tokens_len).int().clip(min=1)
        # scores[~src_token_mask] = -1e5
        # masked_indices = scores.argsort(dim=-1, descending=True) # This is flipped the order. The highest score is the first in order.
        # masked_indices = masked_indices < num_token_masked.unsqueeze(-1) # So it can filter out by "< num_token_masked". We want to filter the high score as a mask
        # ids[masked_indices] = mask_id
        #########################
        temp = []
        for step in range(max_steps):
            sample_max_steps = torch.round(max_steps/max_length*m_tokens_len)
            timestep = torch.clip(step/(sample_max_steps-1), max=1)
            rand_mask_prob = cosine_schedule(timestep)
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = -1e5
            sorted, sorted_score_indices = scores.sort(descending=True)
            ids[~src_token_mask] = pad_id # [INFO] replace with pad id
            ids.scatter_(-1, m_tokens_len[..., None].long(), end_id) # [INFO] replace with end id
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked)
            # [INFO] add one more index for the end_id places
            last_index = sorted_score_indices.gather(-1, num_token_masked.unsqueeze(-1)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(-1, sorted_score_indices, mask_id)
            # if torch.isclose(timestep, torch.tensor(0.7647), atol=.01):
            #     print('masked_indices:', ids[0], src_token_mask[0])

            logits = self.forward(ids, clip_feature, src_token_mask)[:,1:]
            filtered_logits = logits #top_k(logits, topk_filter_thres)
            temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            is_mask = ids == mask_id
            temp.append(is_mask[0].unsqueeze(0))
            
            # mid = is_mask[0][:m_tokens_len[0].int()]
            # mid = mid.nonzero(as_tuple=True)[0]
            # print(is_mask[0].sum(), m_tokens_len[0])

            ids = torch.where(
                        is_mask,
                        pred_ids,
                        ids
                    )
            
            # if timestep == 1.:
            #     print(probs_without_temperature.shape)
            probs_without_temperature = logits.softmax(dim = -1)
            scores = 1 - probs_without_temperature.gather(-1, pred_ids[..., None])
            scores = rearrange(scores, '... 1 -> ...')
            scores = scores.masked_fill(~is_mask, -1e5)
        if if_test:
            return ids, temp
        return ids

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
        self.tok_emb = nn.Embedding(num_vq + 3, embed_dim) # [INFO] 3 = [end_id, blank_id, mask_id]
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
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
    
    def forward(self, idx, clip_feature, src_mask):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
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
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
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

    


        

