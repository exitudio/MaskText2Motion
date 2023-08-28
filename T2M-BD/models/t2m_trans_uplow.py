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
from models.t2m_trans import Block


class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                vqvae,
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
        self.trans_base = CrossCondTransBase(vqvae, num_vq, embed_dim, clip_dim, block_size, 18, n_head, drop_out_rate, fc_rate)
        # self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

        # self.skip_trans = Skip_Connection_Transformer(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)

    def get_block_size(self):
        return self.block_size

    def forward(self, *args, type='forward', **kwargs):
        '''type=[forward, sample]'''
        if type=='forward':
            return self.forward_function(*args, **kwargs)
        elif type=='sample':
            return self.sample(*args, **kwargs)
        elif type=='inpaint':
            return self.inpaint(*args, **kwargs)
        else:
            raise ValueError(f'Unknown "{type}" type')
        
    def get_attn_mask(self, src_mask, att_txt=None):
        if att_txt is None:
            att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        src_mask = torch.cat([att_txt, src_mask],  dim=1)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask

    def forward_function(self, idxs, clip_feature, src_mask=None, att_txt=None, m_tokens_len=None):
        # MLD:
        # if att_txt is None:
        #     att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        # src_mask = torch.cat([att_txt, src_mask],  dim=1)
        # logits = self.skip_trans(idxs, clip_feature, src_mask)

        # T2M-BD
        if src_mask is not None:
            src_mask = self.get_attn_mask(src_mask, att_txt)
        logits = self.trans_base(idxs, clip_feature, src_mask, m_tokens_len)
        # logits = self.trans_head(feat, src_mask)

        return logits

    def sample(self, clip_feature, m_length=None, if_test=False, rand_pos=False, CFG=-1):
        max_steps = 20
        max_length = 49
        batch_size = clip_feature.shape[0]
        mask_id = self.num_vq + 2
        pad_id = self.num_vq + 1
        end_id = self.num_vq
        shape = (batch_size, self.block_size - 1, 2)
        topk_filter_thres = .9
        starting_temperature = 1.0
        scores = torch.ones(shape, dtype = torch.float32, device = clip_feature.device)
        
        m_tokens_len = torch.ceil((m_length)/4)
        src_token_mask = generate_src_mask(self.block_size-1, m_tokens_len+1).unsqueeze(-1).repeat(1,1,2)
        src_token_mask_noend = generate_src_mask(self.block_size-1, m_tokens_len).unsqueeze(-1).repeat(1,1,2)
        ids = torch.full(shape, mask_id, dtype = torch.long, device = clip_feature.device)
        
        # [TODO] confirm that these 2 lines are not neccessary (repeated below and maybe don't need them at all)
        ids[~src_token_mask] = pad_id # [INFO] replace with pad id
        ids.scatter_(1, m_tokens_len[:, None, None].repeat(1,1,2).long(), end_id) # [INFO] replace with end id
        temp = []
        sample_max_steps = torch.round(max_steps/max_length*m_tokens_len) + 1e-8
        for step in range(max_steps):
            timestep = torch.clip(step/(sample_max_steps), max=1)
            rand_mask_prob = cosine_schedule(timestep) # timestep #
            num_token_masked = (rand_mask_prob * m_tokens_len).long().clip(min=1)
            # [INFO] rm no motion frames
            scores[~src_token_mask_noend] = 0
            scores = scores/scores.sum(1)[:, None] # normalize only unmasked token
            
            # if rand_pos:
            #     sorted_score_indices = scores.multinomial(scores.shape[-1], replacement=False) # stocastic
            # else:
            sorted, sorted_score_indices = scores.sort(descending=True, dim=1) # deterministic
            
            ids[~src_token_mask] = pad_id # [INFO] replace with pad id
            ids.scatter_(1, m_tokens_len[:, None, None].repeat(1,1,2).long(), end_id) # [INFO] replace with end id
            ## [INFO] Replace "mask_id" to "ids" that have highest "num_token_masked" "scores" 
            select_masked_indices = generate_src_mask(sorted_score_indices.shape[1], num_token_masked).unsqueeze(-1).repeat(1,1,2)
            # [INFO] repeat last_id to make it scatter_ the existing last ids.
            last_index = sorted_score_indices.gather(1, num_token_masked.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)-1)
            sorted_score_indices = sorted_score_indices * select_masked_indices + (last_index*~select_masked_indices)
            ids.scatter_(1, sorted_score_indices, mask_id)
            # if torch.isclose(timestep, torch.tensor(0.7647), atol=.01):
            #     print('masked_indices:', ids[0], src_token_mask[0])

            logits = self.forward(ids, clip_feature, src_token_mask[..., 0], m_tokens_len=m_tokens_len)[:,1:]
            filtered_logits = logits #top_k(logits, topk_filter_thres)
            if rand_pos:
                temperature = 1 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed
            else:
                temperature = 0 #starting_temperature * (steps_until_x0 / timesteps) # temperature is annealed

            # [INFO] if temperature==0: is equal to argmax (filtered_logits.argmax(dim = -1))
            # pred_ids = filtered_logits.argmax(dim = -1)
            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            is_mask = ids == mask_id
            temp.append(is_mask[:1])
            
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
            scores = scores.masked_fill(~is_mask, 0)
        if if_test:
            return ids, temp
        return ids
    

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                 vqvae,
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.vqvae = vqvae
        embed_dim = int(embed_dim/2)
        
        cb_dim = self.vqvae.quantizer_upper.code_dim
        self.learn_tok_emb_upper = nn.Embedding(3, cb_dim)# [INFO] 3 = [end_id, blank_id, mask_id]
        self.to_emb_upper = nn.Linear(cb_dim, embed_dim)
        self.learn_tok_emb_lower = nn.Embedding(3, cb_dim)# [INFO] 3 = [end_id, blank_id, mask_id]
        self.to_emb_lower = nn.Linear(cb_dim, embed_dim)

        # self.tok_emb_upper = nn.Embedding(num_vq + 3, embed_dim) # [INFO] 3 = [end_id, blank_id, mask_id]
        # self.tok_emb_lower = nn.Embedding(num_vq + 3, embed_dim) # [INFO] 3 = [end_id, blank_id, mask_id]
        # self.cond_emb_upper = nn.Linear(clip_dim, embed_dim)
        # self.cond_emb_lower = nn.Linear(clip_dim, embed_dim)
        cat_block_size = 101
        self.pos_embedding = nn.Embedding(cat_block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, cat_block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(cat_block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.n_head = n_head

        self.ln_upper = nn.LayerNorm(embed_dim)
        self.ln_lower = nn.LayerNorm(embed_dim)
        self.head_upper = nn.Linear(embed_dim, num_vq, bias=False)
        self.head_lower = nn.Linear(embed_dim, num_vq, bias=False)
        
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
    
    def get_token_emb(self, idx, quantizer, learn_token_emb, to_emb):
        not_learn_idx = idx<self.vqvae.num_code
        learn_idx = ~not_learn_idx
        token_embeddings = torch.empty((*idx.shape, self.vqvae.code_dim), device=idx.device)
        token_embeddings[not_learn_idx] = quantizer.dequantize(idx[not_learn_idx]).requires_grad_(False) 
        token_embeddings[learn_idx] = learn_token_emb(idx[learn_idx]-self.vqvae.num_code)
        return to_emb(token_embeddings)

    
    def forward(self, idx, clip_feature, src_mask, m_tokens_len):
        token_embeddings_upper = self.get_token_emb(idx[..., 0], 
                                                    self.vqvae.quantizer_upper,
                                                    self.learn_tok_emb_upper, 
                                                    self.to_emb_upper)
        token_embeddings_lower = self.get_token_emb(idx[..., 1], 
                                                    self.vqvae.quantizer_lower,
                                                    self.learn_tok_emb_lower, 
                                                    self.to_emb_lower)

        # token_embeddings_upper = self.tok_emb_upper(idx[..., 0])
        # token_embeddings_lower = self.tok_emb_lower(idx[..., 1])
        token_embeddings = torch.cat([clip_feature.unsqueeze(1), 
                                       token_embeddings_upper, 
                                       token_embeddings_lower], dim=1)
        x = self.pos_embed(token_embeddings)
        src_mask = generate_src_mask(50, m_tokens_len+1)
        src_mask = torch.cat([src_mask, src_mask], dim=1)
        src_mask = self.get_attn_mask(src_mask)
        for block in self.blocks:
            x = block(x, src_mask)

        token_len = (self.block_size-1)
        x_upper = x[:, 1:token_len+1]
        x_lower = x[:, token_len+1:]
        x_upper = self.ln_upper(x_upper)
        x_lower = self.ln_lower(x_lower)
        logits_upper = self.head_upper(x_upper).unsqueeze(-2)
        logits_lower = self.head_lower(x_lower).unsqueeze(-2)

        logits = torch.cat((logits_upper, logits_lower), dim=-2)
        shape = list(logits.shape)
        shape[1] = 1
        logits = torch.cat((torch.ones(shape, device=logits.device), logits), dim=1)

        # logits: torch.Size([2, 51, 2, 512])
        return logits

    def get_attn_mask(self, src_mask, att_txt=None):
        if att_txt is None:
            att_txt = torch.tensor([[True]]*src_mask.shape[0]).to(src_mask.device)
        src_mask = torch.cat([att_txt, src_mask],  dim=1)
        B, T = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, T).repeat(1, self.n_head, T, 1)
        return src_mask



        

