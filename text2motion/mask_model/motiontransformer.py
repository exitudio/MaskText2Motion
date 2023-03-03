from torch import nn
import torch
import torch.nn.functional as F
from models.transformer import zero_module
import math
from einops import rearrange, repeat


# [INFO] from http://juditacs.github.io/2018/12/27/masked-attention.html
def generate_src_mask(T, length):
    B = len(length)
    mask = torch.arange(T).repeat(B, 1).to(length.device) < length.unsqueeze(-1)
    return mask
    
    src_mask = torch.ones(B, T)
    for i in range(B):
        for j in range(length[i], T):
            src_mask[i, j] = 0
    # print(torch.equal(src_mask, mask))
    return src_mask


class TemporalSelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum(
            'bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        # [INFO] Mask in the wrong dimiension???
        attention = attention + (1 - src_mask.float().unsqueeze(-1)) * -100000
        weight = F.softmax(attention, dim=2)
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y)
        return y


class MotionTransformerOnly(nn.Module):
    def __init__(self,
                 input_feats,
                 output_feats,
                 num_frames=196,
                 latent_dim=512,
                 num_layers=8,
                 num_heads=8,
                 **kargs):
        super().__init__()

        self.sequence_embedding = nn.Parameter(
            torch.randn(num_frames, latent_dim))

        # Text Transformer

        # Input Embedding
        self.joint_embed = nn.Linear(input_feats, latent_dim)
        self.temporal_decoder_blocks = nn.ModuleList([
            # LinearTemporalSelfAttention(
            TemporalSelfAttention(
                latent_dim=latent_dim,
                num_head=num_heads,
            )for i in range(num_layers)])
        # Output Module
        self.ln_out = nn.LayerNorm(latent_dim)
        self.out = zero_module(nn.Linear(latent_dim, output_feats))

    def forward(self, x, src_mask, length=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        for module in self.temporal_decoder_blocks:
            h = module(h, src_mask)

        output = self.out(self.ln_out(h)).view(B, T, -1).contiguous()
        output = output
        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, src_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn[~src_mask] = float('-inf')
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, src_mask):
        if self.training:
            x = x + self.attn(self.norm1(x), src_mask)
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.attn(self.norm1(x), src_mask)
            x = x + self.mlp(self.norm2(x))
        return x


class MotionTransformerOnly2(nn.Module):
    def __init__(self,
                 input_feats,
                 output_feats,
                 num_frames=196,
                 latent_dim=256,
                 num_layers=8,
                 num_heads=8,
                 **kargs):
        super().__init__()

        self.sequence_embedding = nn.Parameter(
            torch.randn(num_frames, latent_dim))
        self.joint_embed = nn.Linear(input_feats, latent_dim)
        self.temporal_blocks = nn.ModuleList([
            Block(
                dim=latent_dim,
                num_heads=num_heads)
            for i in range(num_layers)])
        self.ln_out = nn.LayerNorm(latent_dim)
        self.out = zero_module(nn.Linear(latent_dim, output_feats))

    def forward(self, x, src_mask):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]

        # B, T, latent_dim
        h = self.joint_embed(x)
        h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]

        for block in self.temporal_blocks:
            h = block(h, src_mask)

        output = self.out(self.ln_out(h)).view(B, T, -1).contiguous()
        output = output
        return output