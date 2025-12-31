# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import Optional, List

import torch
import torch.nn as nn
from torch import Tensor

import hotwheels.amct_pytorch.nn.modules as amct_nn


class ConvModule(nn.Module):  # offical conv2d_bn
    def __init__(self, k: int, in_chn: int, out_chn: int, stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        self.c = nn.Conv2d(in_chn, out_chn, k, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chn)

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        x = self.c(x)
        x = self.bn(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chn: int, out_chn: int):
        super().__init__()
        self.seq = nn.ModuleList()
        self.seq.append(ConvModule(3, in_chn, out_chn // 2, stride=2))
        self.seq.append(amct_nn.GELU())
        self.seq.append(ConvModule(3, out_chn // 2, out_chn, stride=2))

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        for m in self.seq:
            x = m(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_chn: int, out_chn: int, stride: int):
        super().__init__()
        self.conv1 = ConvModule(1, in_chn, out_chn, padding=0)
        self.conv2 = ConvModule(3, out_chn, out_chn, stride=stride, groups=out_chn)
        self.conv3 = ConvModule(1, out_chn, out_chn, padding=0)
        self.act = amct_nn.GELU()

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_chn, out_chn: int):
        super().__init__()
        self.conv1 = ConvModule(1, in_chn, in_chn * 4, padding=0)
        self.act1 = amct_nn.GELU()
        self.conv2 = ConvModule(3, in_chn * 4, in_chn * 4, groups=in_chn * 4)
        self.act2 = amct_nn.GELU()
        self.conv3 = ConvModule(1, in_chn * 4, out_chn, padding=0)
        self.act3 = amct_nn.GELU()

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = x + shortcut
        x = self.act3(x)
        return x


class ConvStage(nn.Module):
    def __init__(self, in_chn, chn: int, depth: int, is_downsample: bool = False, out_chn: Optional[int] = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(MBConv(in_chn, chn))
        self.is_downsample = is_downsample
        if is_downsample and out_chn is not None:
            self.downsample = PatchMerging(chn, out_chn, stride=2)

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        if self.is_downsample:
            x = self.downsample(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_chn: int, chn: int):
        super().__init__()
        self.norm = amct_nn.LayerNorm(in_chn)
        self.fc1 = nn.Conv2d(in_chn, chn, 1)
        self.fc2 = nn.Conv2d(chn, in_chn, 1)
        self.act = amct_nn.GELU()

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        local_state = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                if val.shape != self.state_dict().get(name).shape:
                    state_dict[key] = val.unsqueeze(-1).unsqueeze(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class Attention(nn.Module):
    def __init__(self, chn: int, size: int, n_heads: int):
        super().__init__()
        self.scale = (chn // n_heads) ** -0.5
        self.n_heads = n_heads
        self.size = size
        self.chn = chn

        points = list(itertools.product(range(size), range(size)))
        points_num = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(n_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(points_num, points_num),
                             persistent=False)

        self.q = nn.Conv2d(chn, chn, 1)
        self.k = nn.Conv2d(chn, chn, 1)
        self.v = nn.Conv2d(chn, chn, 1)
        self.proj = nn.Conv2d(chn, chn, 1)

    def forward(self, x):
        batch, channel, _, n = x.data.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(batch, self.n_heads, -1, n)
        k = k.reshape(batch, self.n_heads, -1, n)
        v = v.reshape(batch, self.n_heads, -1, n)

        attn = k.transpose(-1, -2) @ q
        attn = attn * self.scale

        attention_biases = self.attention_biases[:, self.attention_bias_idxs]
        attn = attn + attention_biases.data
        attn = attn.softmax(-2)

        x = v @ attn
        x = x.reshape(batch, channel, 1, n)
        x = self.proj(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        local_state = ['proj.weight', 'proj.bias']
        for name in local_state:
            key = prefix + name
            if key in state_dict.keys():
                val = state_dict[key]
                if val.shape != self.state_dict().get(name).shape:
                    state_dict[key] = val.unsqueeze(-1).unsqueeze(-1)
        qkv_weight_key = prefix + 'qkv.weight'
        weight_state = ['q.weight', 'k.weight', 'v.weight']
        if qkv_weight_key in state_dict.keys():
            qkv_weight_val = state_dict[qkv_weight_key].unflatten(0, (self.n_heads, 3, -1))
            for idx, weight_key in enumerate(weight_state):
                state_dict[prefix + weight_key] = qkv_weight_val[:, idx].flatten(0, 1).unsqueeze(-1).unsqueeze(-1)
            del state_dict[qkv_weight_key]

        qkv_bias_key = prefix + 'qkv.bias'
        bias_state = ['q.bias', 'k.bias', 'v.bias']
        if qkv_bias_key in state_dict.keys():
            qkv_bias_val = state_dict[qkv_bias_key].unflatten(0, (self.n_heads, 3, -1))
            for idx, bias_key in enumerate(bias_state):
                state_dict[prefix + bias_key] = qkv_bias_val[:, idx].flatten(0, 1)
            del state_dict[qkv_bias_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


@torch.fx.wrap
def win_partition(inputs, size):
    b, c, h, w = inputs.data.shape
    inputs = inputs.reshape(b, c, int(h / size), size, int(w / size), size)
    inputs = inputs.permute(0, 2, 4, 1, 3, 5).reshape(int(b * h * w / size / size), c, 1, int(size * size))
    return inputs


@torch.fx.wrap
def win_partition_reverse(inputs, size, h, w):
    c = inputs.data.shape[1]
    inputs = inputs.reshape(-1, int(h / size), int(w / size), c, size, size)
    inputs = inputs.permute(0, 3, 1, 4, 2, 5).reshape(-1, c, h, w)
    return inputs


class TinyViTBlock(nn.Module):
    def __init__(self, chn: int, n_heads: int, size: int = 7, mlp_ratio: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.size = size
        self.chn = chn

        self.norm = amct_nn.LayerNorm(chn)
        self.attn = Attention(chn, size, n_heads)
        self.local_conv = ConvModule(3, chn, chn, groups=chn)
        self.mlp = MLP(chn, int(chn * mlp_ratio))

    def forward(self, x):
        hight, width = x.data.shape[-2:]
        shortcut = x

        x = self.norm(x)
        # compute shift
        x_windows = win_partition(x, self.size)

        attn_windows = self.attn(x_windows)

        # reverse shift
        shifted_x = win_partition_reverse(attn_windows, self.size, hight, width)

        x = shifted_x
        x = shortcut + x

        x = self.local_conv(x)
        otherway = self.mlp(x)
        x = x + otherway
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        norm_state = ['norm.weight', 'norm.bias']
        for name in norm_state:
            key = prefix + 'attn.' + name
            if key in state_dict:
                state_dict[prefix + name] = state_dict.get(key)
                del state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class BasicLayer(nn.Module):
    def __init__(self, chn: int, depth: int, n_heads: int, window_size: int, stride: int, \
                 is_downsample: bool = False, out_chn: Optional[int] = None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(TinyViTBlock(chn, n_heads, window_size))
        self.is_downsample = is_downsample
        if is_downsample and out_chn is not None:
            self.downsample = PatchMerging(chn, out_chn, stride)

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        if self.is_downsample:
            x = self.downsample(x)
        return x


class TinyVit(nn.Module):
    def __init__(self, in_chn, embed_dims: List[int], depths: List[int], heads: List[int], window_sizes: List[int]):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chn, embed_dims[0])
        self.layers = nn.ModuleList()
        self.layers.append(ConvStage(embed_dims[0], embed_dims[0], depths[0], True, embed_dims[1]))
        self.layers.append(BasicLayer(embed_dims[1], depths[1], heads[1], window_sizes[1], 2, True, embed_dims[2]))
        self.layers.append(BasicLayer(embed_dims[2], depths[2], heads[2], window_sizes[2], 1, True, embed_dims[3]))
        self.layers.append(BasicLayer(embed_dims[3], depths[3], heads[3], window_sizes[3], 1))
        self.neck = nn.ModuleList()
        self.neck.append(nn.Conv2d(embed_dims[3], 256, 1, bias=False))
        self.neck.append(amct_nn.LayerNorm(256))
        self.neck.append(nn.Conv2d(256, 256, 3, bias=False, padding=1))
        self.neck.append(amct_nn.LayerNorm(256))

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        for n in self.neck:
            x = n(x)
        return x
