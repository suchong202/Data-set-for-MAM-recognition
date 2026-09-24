import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from mmedit.models.backbones.sr_backbones.basicvsr_net import SPyNet  # 光流模型
# from mmedit.models.common import flow_warp  # 光流对齐 flow_warp

from mmagic.models.editors.basicvsr.basicvsr_net import SPyNet
from mmagic.models.utils import flow_warp

from mmengine.model import BaseModule
#from mmengine.registry import MODELS
from mmaction.registry import MODELS


class ROI_Attention(nn.Module):
    def __init__(
            self,
            dim=768,
            patch_size=16,
            #window_size=(3, 4, 4),
            dim_head=64,
            heads=8,
            #shift=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.patch_size = patch_size
        #self.shift = shift
        inner_dim = dim_head * heads

        # Static attention map
        #q_l = self.window_size[1] * self.window_size[2]
        #kv_l = self.window_size[0] * self.window_size[1] * self.window_size[2]
        num_patches = patch_size * patch_size
        #self.static_a = nn.Parameter(torch.Tensor(1, heads, q_l, kv_l))
        self.static_a = nn.Parameter(torch.Tensor(1, heads, num_patches, num_patches))
        nn.init.trunc_normal_(self.static_a, std=0.02)

        # Norm layers and Conv2D layers
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q_inp, k_inp, flow):
        b, f_q, c, h, w = q_inp.shape
        #fb, hb, wb = self.window_size

        flow_f, flow_b = flow

        # Sliding window adjustment if shift is enabled
        #if self.shift:
            #q_inp, k_inp = map(lambda x: torch.roll(x, shifts=(-hb // 2, -wb // 2), dims=(-2, -1)), (q_inp, k_inp))
            #if flow_f is not None:
             #   flow_f = torch.roll(flow_f, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))
            #if flow_b is not None:
             #   flow_b = torch.roll(flow_b, shifts=(-hb // 2, -wb // 2), dims=(-2, -1))

        #Align using flow
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
        q, k, v = map(lambda t: rearrange(t, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=self.patch_size, p2=self.patch_size), (q, k, v))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Attention computation
        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) + self.static_a
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # Merge heads and output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b=b, h=h // self.patch_size, w=w // self.patch_size, p1=self.patch_size, p2=self.patch_size)
        return out
        #return self.to_out(out)



class ROI_Block(nn.Module):
    def __init__(self,
                 q_dim, 
                 #emb_dim, 
                 #window_size=(3, 4, 4), 
                 atch_size=16,
                 dim_head=64, 
                 heads=8, 
                 #shift=False):
        super().__init__()
        self.attn = ROI_Attention(q_dim, window_size, dim_head, heads, shift=shift)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, flows_forward, flows_backward):
        x = x.permute(1, 0, 2, 3, 4)  # Time-major
        outs = []
        for i in range(x.size(0)):
            flow_f, flow_b = (flows_forward[i - 1] if i > 0 else None, flows_backward[i] if i < x.size(0) - 1 else None)
            q_inp, k_inp = x[i:i + 1], x[max(0, i - 1):i + 2]
            mt = self.attn(q_inp=q_inp, k_inp=k_inp, flow=[flow_f, flow_b])
            outs.append(mt)
        return torch.stack(outs, dim=1).permute(1, 0, 2, 3, 4)

@MODELS.register_module()
class Mask_Module(BaseModule):
    def __init__(self, config):
        super(Mask_Module, self).__init__()
        self.in_channels = config['in_channels']
        self.window_size = [3, 4, 4]
        
        self.spynet = SPyNet(pretrained=None)
        
        self.roi_block = ROI_Block(config['in_channels'], config['embed_dim'], window_size=self.window_size)

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