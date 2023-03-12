# From PoseFormer: https://github.com/zczcwh/PoseFormer/edit/main/common/model_poseformer.py
# PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

from functools import partial
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import numpy as np


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
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, src_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, src_mask=None):
        if self.training:
            x = x + self.drop_path(self.attn(self.norm1(x), src_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_frame=31):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=.01)
        self.ref_pad = torch.nn.ReflectionPad2d((0, 0, kernel_frame-1, 0))
        # self.conv = torch.nn.Conv2d(in_channels=in_channels,
        #                         out_channels=out_channels,
        #                         kernel_size=(31, 1),
        #                         stride=1,
        #                         padding=0)
        self.conv = DepthwiseSeparableConv(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=(kernel_frame, 1),
                                           padding=0)
        self.act = torch.nn.GELU()
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.no_diff_c = False
        if in_channels != out_channels:
            self.linear = torch.nn.Linear(in_channels, out_channels)
            self.linear_act = torch.nn.GELU()
            self.bn_skip = torch.nn.BatchNorm2d(out_channels, eps=0.001)
            self.no_diff_c = True

    def forward(self, x):
        if self.no_diff_c:
            res_x = rearrange(x, 'b e f j -> b f j e')
            res_x = self.linear(res_x)
            res_x = rearrange(res_x, 'b f j e -> b e f j')
            res_x = self.linear_act(res_x)
            res_x = self.bn_skip(res_x)
        else:
            res_x = x

        x = self.dropout(x)
        x = self.ref_pad(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x += res_x
        return x


class SpatialTransformerTemporalConv(nn.Module):
    def __init__(self, num_frames=9, num_joints=17, in_chans=3, spatial_embed_dim=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  norm_layer=None, out_dim=124, kernel_frame=31):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.spatial_embed_dim = spatial_embed_dim
        # spatial_embed_dim #8*spatial_embed_dim # spatial_embed_dim * num_joints
        self.final_embed_dim = 256
        self.out_dim = out_dim

        # spatial patch embedding
        self.spatial_joint_to_embedding = nn.Linear(
            in_chans, spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, num_joints, spatial_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.spatial_norm = norm_layer(spatial_embed_dim)

        self.conv1 = ConvBlock(
            in_channels=32, out_channels=32, kernel_frame=kernel_frame)
        self.conv2 = ConvBlock(
            in_channels=32, out_channels=64, kernel_frame=kernel_frame)
        self.conv3 = ConvBlock(
            in_channels=64, out_channels=128, kernel_frame=kernel_frame)
        self.conv4 = ConvBlock(
            in_channels=128, out_channels=self.final_embed_dim, kernel_frame=kernel_frame)
        self.avg_pool = torch.nn.AvgPool2d(
            kernel_size=(num_frames, num_joints), stride=1)

        self.head = nn.Sequential(
            nn.LayerNorm(self.final_embed_dim),
            nn.Linear(self.final_embed_dim, self.out_dim),
        )

    def spatial_transformer(self, x):
        for blk in self.spatial_blocks:
            x = blk(x)
        return self.spatial_norm(x)

    def spatial_forward(self, x):
        b, f, j, d = x.shape
        x = rearrange(x, 'b f j d -> (b f) j  d')
        x = self.spatial_joint_to_embedding(x)
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) j e -> b e f j', f=f)
        return x

    def temporal_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = torch.squeeze(x, dim=2)
        x = torch.squeeze(x, dim=2)
        return x

    def forward(self, x):
        x = self.spatial_forward(x)
        x = self.temporal_forward(x)
        x = self.head(x)
        x = F.normalize(x, dim=1, p=2)
        return x


class SpatialPatchEmbedding(nn.Module):
    def __init__(self, num_joints, spatial_embed_dim):
        super().__init__()
        self.num_joints = num_joints
        self.spatial_embed_dim = spatial_embed_dim

        # TEST
        # def linear(x):
        #     r = 11 if x.shape[-1]==12 else 12
        #     return x.repeat(1, 1 , 1, r)
        # def delinear(dim):
        #     def temp(x):
        #         return x[..., :dim]
        #     return temp

        self.root_joint_embed = nn.Linear(11, spatial_embed_dim)
        self.other_joints_embed = nn.Linear(12, spatial_embed_dim)

        self.root_joint_projection = nn.Linear(spatial_embed_dim, 11)
        self.other_joints_projection = nn.Linear(spatial_embed_dim, 12)

        self.idx = np.cumsum(
            [4, 3*(num_joints-1), 6*(num_joints-1), 3*num_joints, 4])
        self.num_tokens = num_joints

    def forward(self, x, type='encode'):
        if type == 'encode':
            b, f, d = x.shape

            root_info = x[:, :, :self.idx[0]]  # [b, f, 4]
            # [b, f, j-1, 3]
            j_pos = x[:, :, self.idx[0]:self.idx[1]].reshape(b, f, -1, 3)
            j_rot = x[:, :, self.idx[1]:self.idx[2]].reshape(b, f, -1, 6)
            j_vel = x[:, :, self.idx[2]:self.idx[3]].reshape(b, f, -1, 3)
            foot_contact = x[:, :, self.idx[3]:]  # [b, f, 4]

            # [b, f, 1, 11]
            root_joint = torch.concat(
                [root_info, j_vel[:, :, 0], foot_contact], dim=-1).unsqueeze(dim=-2)
            # [b, f, j-1, 12]
            other_joints = torch.concat(
                [j_pos, j_rot, j_vel[:, :, 1:]], dim=-1)
            root_joint = self.root_joint_embed(root_joint)
            other_joints = self.other_joints_embed(other_joints)

            x = torch.concat([root_joint, other_joints], dim=-2)
        elif type == 'decode':
            b, f, tk, e = x.shape
            ######## Root Joint ########
            # [b, f, 11]
            root_joint = self.root_joint_projection(x[:, :, 0])
            # [b, f, 4]
            root_info = root_joint[:, :, :4]
            j_vel_0 = root_joint[:, :, 4:-4]
            foot_contact = root_joint[:, :, -4:]

            ######## Other joints ########
            # [b, f, j-1, 12]
            other_joints = self.other_joints_projection(x[:, :, 1:])
            # [b, f, j-1, 3]
            j_pos = other_joints[..., :3].reshape(b, f, -1)
            j_rot = other_joints[..., 3:-3].reshape(b, f, -1)
            j_vel = other_joints[..., -3:].reshape(b, f, -1)
            j_vel = torch.concat([j_vel_0, j_vel], dim=-1).reshape(b, f, -1)

            x = torch.concat(
                [root_info, j_pos, j_rot, j_vel, foot_contact], dim=-1)
        return x



class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_frames=9, num_joints=17, spatial_embed_dim=32, depth=4):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.spatial_embed_dim = spatial_embed_dim
        self.temporal_embed_dim = spatial_embed_dim  * num_joints

        # spatial patch embedding
        self.spatial_patch_embedding = SpatialPatchEmbedding(
            num_joints, spatial_embed_dim)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, self.spatial_patch_embedding.num_tokens, spatial_embed_dim))

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, self.temporal_embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, 0., depth)]

        self.spatial_blocks = nn.ModuleList([
            Block(
                dim=spatial_embed_dim, num_heads=8, mlp_ratio=2., qkv_bias=True,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.temporal_blocks = nn.ModuleList([
            Block(
                dim=self.temporal_embed_dim, num_heads=8, mlp_ratio=2., qkv_bias=True,
                drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.spatial_norm = norm_layer(spatial_embed_dim)
        self.temporal_norm = norm_layer(self.temporal_embed_dim)

    def spatial_transformer(self, x):
        for blk in self.spatial_blocks:
            x = blk(x)
        return self.spatial_norm(x)

    def temporal_transformer(self, x, src_mask):
        for blk in self.temporal_blocks:
            x = blk(x, src_mask)
        return self.temporal_norm(x)

    def spatial_forward(self, x):
        b, f, j, d = x.shape
        x = rearrange(x, 'b f j d -> (b f) j  d')
        x += self.spatial_pos_embed
        x = self.pos_drop(x)
        x = self.spatial_transformer(x)
        x = rearrange(x, '(b f) j e -> b f (j e)', f=f)
        return x

    def temporal_forward(self, x, src_mask):
        b, f, e = x.shape
        x += self.temporal_pos_embed
        x = self.pos_drop(x)
        x = self.temporal_transformer(x, src_mask)
        x = rearrange(x, 'b f (j e) -> b f j e',
                      j=self.spatial_patch_embedding.num_tokens)
        return x

    def forward(self, x, timesteps, length=None, text=None):
        src_mask = self.generate_src_mask(x.shape[1], length).to(x.device).unsqueeze(-1)
        x = self.spatial_patch_embedding(x, type='encode')
        x = self.spatial_forward(x)
        x = self.temporal_forward(x, src_mask)
        x = self.spatial_patch_embedding(x, type='decode')
        return x

    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask