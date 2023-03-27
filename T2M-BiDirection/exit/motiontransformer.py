from torch import nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange, repeat


# [INFO] from http://juditacs.github.io/2018/12/27/masked-attention.html
def generate_src_mask(T, length):
    B = len(length)
    mask = torch.arange(T).repeat(B, 1).to(length.device) < length.unsqueeze(-1)
    return mask

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

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

        if src_mask is not None:
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
        self.out = nn.Linear(latent_dim, output_feats)

    def forward(self, x, src_mask=None):
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
        return output