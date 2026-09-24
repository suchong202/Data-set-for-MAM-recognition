# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine import ConfigDict
from mmengine.logging import MMLogger
from mmengine.model.weight_init import kaiming_init, trunc_normal_
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict
from torch.nn.modules.utils import _pair

from mmagic.models.editors.basicvsr.basicvsr_net import SPyNet
from mmagic.models.utils import flow_warp

from mmengine.model import BaseModule
from mmaction.registry import MODELS


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_cfg (dict | None): Config dict for convolution layer. Defaults to
            `dict(type='Conv2d')`.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=dict(type='Conv2d')):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0])
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        # Lecun norm from ClassyVision
        kaiming_init(self.projection, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the module.
        """
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x

class ROI_Attention(nn.Module):
    def __init__(
            self,
            embed_dims, 
            window_size=(4, 4),
            dim_head=64,
            heads=8,
            shift=False
    ):
        super().__init__()

        self.embed_dims = embed_dims
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads

        # Static attention map
        q_l = self.window_size[1] * self.window_size[2]
        kv_l = self.window_size[0] * self.window_size[1] * self.window_size[2]
        self.static_a = nn.Parameter(torch.Tensor(1, heads, q_l, kv_l))
        nn.init.trunc_normal_(self.static_a, std=0.02)

        # Norm layers and Conv2D layers
        self.norm_q = nn.LayerNorm(embed_dims)
        self.norm_kv = nn.LayerNorm(embed_dims)
        self.to_q = nn.Conv2d(embed_dims, inner_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.to_kv = nn.Conv2d(embed_dims, inner_dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q_inp, k_inp, flow):
        b, f_q, c, h, w = q_inp.shape
        hb, wb = self.window_size

        flow_f, flow_b = flow

        # Sliding window adjustment if shift is enabled
        if self.shift:
            q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb // 2, -wb // 2), dims=(-2, -1)), (q_inp, k_inp))
            if flow_f is not None:
                flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
            if flow_b is not None:
                flow_b = torch.roll(flow_b, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))

        # Align using flow
        def apply_flow_alignment(flow, keyframe):
            grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing="ij")
            grid = torch.stack((grid_x, grid_y), 2).float().to(keyframe.device)
            vgrid = grid + flow.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            return F.grid_sample(keyframe.float(), vgrid_scaled, mode='nearest')

        if flow_f is not None:
            k_f = apply_flow_alignment(flow_f, k_inp[:, 0])
        else:
            k_f = k_inp[:, 0]

        if flow_b is not None:
            k_b = apply_flow_alignment(flow_b, k_inp[:, 2])
        else:
            k_b = k_inp[:, 2]

        k_r = k_inp[:, 1]
        k_inp = torch.stack([k_f, k_r, k_b], dim=1)

        # Norm and convolution
        q = self.norm_q(q_inp.flatten(0, 1).permute(0, 2, 3, 1))
        kv = self.norm_kv(k_inp.flatten(0, 1).permute(0, 2, 3, 1))
        q = self.to_q(q.permute(0, 3, 1, 2))
        k, v = self.to_kv(kv.permute(0, 3, 1, 2)).chunk(2, dim=1)

        # Rearrange for attention computation
        q, k, v = map(lambda t: rearrange(t, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=hb, p2=wb), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Attention computation
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) + self.static_a
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # Merge heads and output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=h // hb, w=w // wb, p1=hb, p2=wb)
        # Apply output convolution
        out = self.to_out(out)

        # Optional: inverse shift
        if self.shift:
            out = torch.roll(out, shifts=(hb // 2, wb // 2), dims=(-2, -1))

        return out

class ROI_Block(nn.Module):
    def __init__(self,
                 patch_embed,
                 embed_dims,
                 window_size=(4, 4)):
        super().__init__()
        self.patch_embed = patch_embed  # 直接调用 Timesformer 中的 PatchEmbed
        self.attn = ROI_Attention(embed_dims=embed_dims, window_size=window_size)  # ROI_Attention 使用与 PatchEmbed 相同的 embed_dims

    def forward(self, x, flows_forward, flows_backward):
        """
        Args:
            x: 输入视频张量 [batch_size, num_frames, channels, height, width]
            flows_forward: 前向光流
            flows_backward: 后向光流

        Returns:
            Mt: 运动概率分布的嵌入特征，形状为 [batch_size * num_frames, num_patches, embed_dims]
        """
        # 调整为 Time-major 格式 [num_frames, batch_size, channels, height, width]
        x = x.permute(1, 0, 2, 3, 4)
        num_frames, batch_size, _, height, width = x.shape

        # 逐帧调用 ROI_Attention
        outs = []
        for i in range(num_frames):
            flow_f = flows_forward[i - 1] if i > 0 else None
            flow_b = flows_backward[i] if i < num_frames - 1 else None

            # 查询帧和关键帧
            q_inp = x[i:i + 1]
            k_inp = x[max(0, i - 1):i + 2]

            # 调用 ROI_Attention
            mt = self.attn(q_inp=q_inp, k_inp=k_inp, flow=[flow_f, flow_b])  # [batch_size, embed_dims, height, width]
            outs.append(mt)

        # 恢复为原始时序格式 [batch_size, num_frames, embed_dims, height, width]
        outs = torch.stack(outs, dim=1)  # [batch_size, num_frames, embed_dims, height, width]

        # 调用 PatchEmbed
        outs = outs.flatten(0, 1)  # [batch_size * num_frames, embed_dims, height, width]
        patches = self.patch_embed(outs)  # [batch_size * num_frames, num_patches, embed_dims]

        return patches

@MODELS.register_module()
class Mask_Module(BaseModule):
    def __init__(self, in_channels, embed_dims, patch_embed):
        """
        Mask_Module 用于生成运动概率分布 Mt。

        Args:
            in_channels: 输入视频通道数，通常为 3。
            embed_dims: 嵌入维度，与 TimeSFormer 的 embed_dims 一致。
            patch_embed: TimeSFormer 的 PatchEmbed 模块，用于生成 patch 特征。
        """
        super(Mask_Module, self).__init__()
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        # 光流模型
        self.spynet = SPyNet(pretrained=None)

        # ROI_Block 用于提取运动概率分布
        self.roi_block = ROI_Block(
            patch_embed=patch_embed,
            embed_dims=self.embed_dims
        )

    def compute_flow(self, frames):
        """计算输入视频帧的前向和后向光流。"""
        n, t, c, h, w = frames.size()
        flows = {'forward': [], 'backward': []}
        for i in range(t - 1):
            flow_forward = self.spynet(frames[:, i + 1], frames[:, i])
            flow_backward = self.spynet(frames[:, i], frames[:, i + 1])
            flows['forward'].append(flow_forward)
            flows['backward'].append(flow_backward)

        # 合并到统一张量
        flows['forward'] = torch.stack(flows['forward'], dim=1)  # [batch_size, t-1, h, w]
        flows['backward'] = torch.stack(flows['backward'], dim=1)  # [batch_size, t-1, h, w]
        return flows

    def forward(self, frames):
        """
        Args:
            frames: 输入视频帧，形状为 [batch_size, num_frames, channels, height, width]

        Returns:
            Mt: 运动概率分布，形状为 [batch_size * num_frames, num_patches, embed_dims]
        """
        # 计算前向和后向光流
        flows = self.compute_flow(frames)
        flows_forward = flows['forward']
        flows_backward = flows['backward']

        # 调用 ROI_Block 生成运动概率分布
        mt = self.roi_block(frames, flows_forward, flows_backward)  # [batch_size * num_frames, num_patches, embed_dims]

        return mt


@MODELS.register_module()
class TimeSformer(nn.Module):
    """TimeSformer. A PyTorch impl of `Is Space-Time Attention All You Need for
    Video Understanding? <https://arxiv.org/abs/2102.05095>`_

    Args:
        num_frames (int): Number of frames in the video.
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        pretrained (str | None): Name of pretrained model. Default: None.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        num_transformer_layers (int): Number of transformer layers. Defaults to
            12.
        in_channels (int): Channel num of input features. Defaults to 3.
        dropout_ratio (float): Probability of dropout layer. Defaults to 0..
        transformer_layers (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict` | None): Config of transformerlayer in
            TransformerCoder. If it is obj:`mmcv.ConfigDict`, it would be
            repeated `num_transformer_layers` times to a
            list[obj:`mmcv.ConfigDict`]. Defaults to None.
        attention_type (str): Type of attentions in TransformerCoder. Choices
            are 'divided_space_time', 'space_only' and 'joint_space_time'.
            Defaults to 'divided_space_time'.
        norm_cfg (dict): Config for norm layers. Defaults to
            `dict(type='LN', eps=1e-6)`.
    """
    supported_attention_types = [
        'divided_space_time', 'space_only', 'joint_space_time'
    ]

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 pretrained=None,
                 embed_dims=768,
                 num_heads=12,
                 num_transformer_layers=12,
                 in_channels=3,
                 dropout_ratio=0.,
                 transformer_layers=None,
                 attention_type='divided_space_time',
                 norm_cfg=dict(type='LN', eps=1e-6),
                 mask_module=None, 
                 **kwargs):
        super().__init__(**kwargs)
        assert attention_type in self.supported_attention_types, (
            f'Unsupported Attention Type {attention_type}!')
        assert transformer_layers is None or isinstance(
            transformer_layers, (dict, list))

        self.num_frames = num_frames
        self.pretrained = pretrained
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.attention_type = attention_type
        self.mask_module = mask_module 

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=dropout_ratio)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dims))
            self.drop_after_time = nn.Dropout(p=dropout_ratio)

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        if transformer_layers is None:
            # stochastic depth decay rule
            dpr = np.linspace(0, 0.1, num_transformer_layers)

            if self.attention_type == 'divided_space_time':
                _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='DividedTemporalAttentionWithNorm_ada',  # First layer with _ada
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6)),
                            dict(
                                type='DividedSpatialAttentionWithNorm',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6))
                        ],
                        ffn_cfgs=dict(
                            type='FFNWithNorm',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i]),
                            norm_cfg=dict(type='LN', eps=1e-6)),
                        operation_order=('self_attn', 'self_attn', 'ffn'))
                    for i in range(1)  # Only for the first transformer layer
                ]
                # Use DividedTemporalAttentionWithNorm for the remaining layers
                _transformerlayers_cfg += [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='DividedTemporalAttentionWithNorm',  # Use original attention for subsequent layers
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6)),
                            dict(
                                type='DividedSpatialAttentionWithNorm',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                num_frames=num_frames,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]),
                                norm_cfg=dict(type='LN', eps=1e-6))
                        ],
                        ffn_cfgs=dict(
                            type='FFNWithNorm',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i]),
                            norm_cfg=dict(type='LN', eps=1e-6)),
                        operation_order=('self_attn', 'self_attn', 'ffn'))
                    for i in range(1, num_transformer_layers)  # Remaining layers
                ]
            else:
                # Sapce Only & Joint Space Time
                _transformerlayers_cfg = [
                    dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=num_heads,
                                batch_first=True,
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=dpr[i]))
                        ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=embed_dims * 4,
                            num_fcs=2,
                            act_cfg=dict(type='GELU'),
                            dropout_layer=dict(
                                type='DropPath', drop_prob=dpr[i])),
                        operation_order=('norm', 'self_attn', 'norm', 'ffn'),
                        norm_cfg=dict(type='LN', eps=1e-6),
                        batch_first=True)
                    for i in range(num_transformer_layers)
                ]

            transformer_layers = ConfigDict(
                dict(
                    type='TransformerLayerSequence',
                    transformerlayers=_transformerlayers_cfg,
                    num_layers=num_transformer_layers))

        self.transformer_layers = build_transformer_layer_sequence(
            transformer_layers)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            state_dict = _load_checkpoint(self.pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            if self.attention_type == 'divided_space_time':
                # modify the key names of norm layers
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'norms' in old_key:
                        new_key = old_key.replace('norms.0',
                                                  'attentions.0.norm')
                        new_key = new_key.replace('norms.1', 'ffns.0.norm')
                        state_dict[new_key] = state_dict.pop(old_key)

                # copy the parameters of space attention to time attention
                old_state_dict_keys = list(state_dict.keys())
                for old_key in old_state_dict_keys:
                    if 'attentions.0' in old_key:
                        new_key = old_key.replace('attentions.0',
                                                  'attentions.1')
                        state_dict[new_key] = state_dict[old_key].clone()

            load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call."""
        batches = x.shape[0]
        
        # Extract motion mask (Mt) using Mask_Module
        motion_mask = self.mask_module(x)  # motion_mask shape: [batch_size * num_frames, num_patches, embed_dims]        

        # x [batch_size * num_frames, num_patches, embed_dims]
        x = self.patch_embed(x)

        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Add Time Embedding
        if self.attention_type != 'space_only':
            # x [batch_size, num_patches * num_frames + 1, embed_dims]
            cls_tokens = x[:batches, 0, :].unsqueeze(1)
            x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=batches)
            x = x + self.time_embed
            x = self.drop_after_time(x)
            x = rearrange(x, '(b p) t m -> b (p t) m', b=batches)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_layers(x, motion_mask=motion_mask)

        if self.attention_type == 'space_only':
            # x [batch_size, num_patches + 1, embed_dims]
            x = x.view(-1, self.num_frames, *x.size()[-2:])
            x = torch.mean(x, 1)

        x = self.norm(x)

        # Return Class Token
        return x[:, 0]
