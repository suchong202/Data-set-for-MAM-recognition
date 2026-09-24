import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mmedit.models.backbones.sr_backbones.basicvsr_net import SPyNet  # 光流模型
from mmedit.models.common import flow_warp  # 光流对齐 flow_warp


class ROI_Attention(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(5, 4, 4),
            dim_head=64,
            heads=8,
            shift=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.shift = shift
        inner_dim = dim_head * heads

        # Convolution layers for q and kv with Conv3D 
        self.to_q = nn.Conv3d(dim, inner_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.to_kv = nn.Conv3d(dim, inner_dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.to_out = nn.Conv3d(inner_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q_inp, k_inp, flow):
        """
        :param q_inp: [batch_size, 1, channels, height, width]
        :param k_inp: [batch_size, 2r+1, channels, height, width]
        :param flow: list: [[batch_size, 2, height, width], [batch_size, 2, height, width]]
        :return: out: [batch_size, 1, channels, height, width]
        """
        b, f_q, c, h, w = q_inp.shape
        fb, hb, wb = self.window_size

        flow_f, flow_b = flow

        # Sliding window adjustment if shift is enabled
        if self.shift:
            q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb // 2, -wb // 2), dims=(-2, -1)), (q_inp, k_inp))
            flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
            flow_b = torch.roll(flow_b, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))

        # Keyframe retrieval
        k_f, k_r, k_b = k_inp[:, 1:5], k_inp[:, 0], k_inp[:, 5]  # Middle frames + first/last frames as references

        # Align using flow
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).float().to(k_f.device)
        grid.requires_grad = False

        def apply_flow_alignment(flow, keyframe):
            vgrid = grid + flow.permute(0, 2, 3, 1)
            vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
            vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
            vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
            return F.grid_sample(keyframe.float(), vgrid_scaled, mode='nearest')

        k_f = apply_flow_alignment(flow_f, k_f)
        k_b = apply_flow_alignment(flow_b, k_b)

        k_inp = torch.stack([k_f, k_r, k_b], dim=1)

        # Process q, k, v with Conv3D
        q = self.to_q(q_inp)
        k, v = self.to_kv(k_inp).chunk(2, dim=1)

        # Rearrange tensors for attention computation
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> (b h w) c', h=h // hb, w=w // wb), (q, k, v))
        
        # Attention scaling
        q = q * self.scale

        # Attention computation
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # Reshape to original format
        out = rearrange(out, '(b h w) c (p1 p2) -> b (h p1) (w p2) c', p1=hb, p2=wb, h=h // hb, w=w // wb)
        out = self.to_out(out)

        # Inverse shift if applied
        if self.shift:
            out = torch.roll(out, shifts=(hb // 2, wb // 2), dims=(-2, -1))

        return out


class ROI_Block(nn.Module):
    def __init__(self, 
                 q_dim, 
                 emb_dim, 
                 window_size=(3, 4, 4), 
                 dim_head=64, heads=8, 
                 num_resblocks=5, 
                 shift=False):
        
        super(ROI_Block, self).__init__()
        self.window_size = window_size
        self.heads = heads
        self.embed_dim = emb_dim
        self.q_dim = q_dim
        self.shift = shift

        # Initialize ROI_Attention
        self.attn = ROI_Attention(q_dim, window_size, dim_head, heads, shift=shift)

        # 2D convolution to combine embedding and current frame features
        self.conv = nn.Conv2d(q_dim + emb_dim, q_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, flows_forward, flows_backward):
        """
        :param x: [n, t, c, h, w] - 输入视频帧序列
        :param flows_forward: [n, t-1, 2, h, w] - 正向光流
        :param flows_backward: [n, t-1, 2, h, w] - 反向光流
        :return: outs: [n, t, c, h, w] - 输出特征序列
        """
        # x 变为 [t, n, c, h, w]
        x = x.permute(1, 0, 2, 3, 4) 
        t, n, c, h, w = x.shape

        # 输出特征列表初始化
        outs = []
        embedding = flows_forward[0].new_zeros(n, self.embed_dim, h, w)

        for i in range(t):
            flow_f, flow_b = None, None
            if i > 0:
                flow_f = flows_forward[i - 1]  # 正向光流
                embedding = flow_warp(embedding, flow_f.permute(0, 2, 3, 1))  # 前向对齐
                k_f = x[i - 1]
            else:
                k_f = x[i]

            if i < t - 1:
                flow_b = flows_backward[i]  # 反向光流
                k_b = x[i + 1]
            else:
                k_b = x[i]
    
            x_current = x[i]

            # 融合嵌入特征和当前帧特征
            q_inp = self.lrelu(self.conv(torch.cat((embedding, x_current), dim=1))).unsqueeze(1)
            k_inp = torch.stack([k_f, x_current, k_b], dim=1)

            # 使用 ROI_Attention 模块进行注意力处理
            out = self.attn(q_inp=q_inp, k_inp=k_inp, flow=[flow_f, flow_b]) + q_inp
            out = out.squeeze(1)
            
            # 更新 embedding 为当前的输出特征
            embedding = out

            # 保存每个时间步的输出
            outs.append(out)

        # 将输出特征列表拼接并还原时间维度
        outs = torch.stack(outs, dim=1)
        return outs.permute(1, 0, 2, 3, 4)  # 返回 [n, t, c, h, w] 格式

class Mask_Module(nn.Module):
    def __init__(self, config):
        super(Mask_Module, self).__init__()
        self.in_channels = config['in_channels']
        self.window_size = [3, 3, 3]
        
        self.spynet = SPyNet(pretrained=None)
        
        self.roi_block = ROI_Block(config['in_channels'], config['embed_dim'], window_size=(3, 4, 4))

    def spatial_padding(self, frames):
        n, t, c, h, w = frames.shape
        tb, hb, wb = self.window_size
        pad_h = (hb - h % hb) % hb
        pad_w = (wb - w % wb) % wb

        frames = frames.view(-1, c, h, w)
        frames = F.pad(frames, [0, pad_w, 0, pad_h], mode='reflect')
        return frames.view(n, t, c, h + pad_h, w + pad_w)

    def compute_flow(self, frames):
        n, t, c, h, w = frames.size()
        flows = {'forward': [], 'backward': []}
        for i in range(t - 1):
            flow_forward = self.spynet(frames[:, i + 1], frames[:, i])
            flow_backward = self.spynet(frames[:, i], frames[:, i + 1])
            flows['forward'].append(flow_forward)
            flows['backward'].append(flow_backward)
        return flows

    def forward(self, frames):
        n, t, c, h, w = frames.size()
        roi_weights_sequence = []

        # Process each segment and compute ROI weights for middle frames only
        for i in range(0, t, 6):  # t = 24, split into segments of 6 frames
            segment = frames[:, i:i + 6]
            padded_segment = self.spatial_padding(segment)
            flows = self.compute_flow(padded_segment)

            # Only compute ROI weights for middle 4 frames in each segment
            roi_weights = self.roi_block(padded_segment, flows['forward'], flows['backward'])
            roi_weights_sequence.append(roi_weights)

        return torch.cat(roi_weights_sequence, dim=1)  # Concatenate all segments for final output