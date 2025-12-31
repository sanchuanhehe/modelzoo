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

import math
from typing import Tuple

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import hotwheels.amct_pytorch.nn.modules as amct_nn
from TinySAM.tinysam.modeling.common import MLPBlock


def select_masks(masks: torch.Tensor, iou_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    best_idx = torch.argmax(iou_preds, dim=1, keepdim=True).flatten()
    masks = masks[:, best_idx, :, :]
    iou_preds = iou_preds[:, best_idx]
    return masks, iou_preds


@torch.fx.wrap
def post_process(masks, iou_pred, image_size):
    mask_slice = slice(1, None)
    # (1, 4, 112, 112) -> (1, 3, 112, 112)
    masks = masks[:, mask_slice, :, :]
    # (1, 4, 1, 1) -> (1, 3, 1, 1)
    iou_pred = iou_pred[:, mask_slice]
    masks, iou_pred = select_masks(masks, iou_pred)
    upscaled_masks = F.interpolate(
        masks,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return iou_pred, upscaled_masks


class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads, downsample_rate=1):
        super().__init__()
        self.q_proj = nn.Conv2d(embed_dim, embed_dim // downsample_rate, 1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim // downsample_rate, 1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim // downsample_rate, 1)
        self.out_proj = nn.Conv2d(embed_dim // downsample_rate, embed_dim, 1)
        self.scale = (embed_dim // downsample_rate // n_heads) ** -0.5
        self.n_heads = n_heads

    def forward(self, q, k, v):
        """
        :param q       (B, C, 1, N)     e.g. (1, 512, 1, 65)
        :param k       (B, C, 1, N)     e.g. (1, 512, 1, 65)
        :param v       (B, C, 1, N)     e.g. (1, 512, 1, 65)

        return out     (B, C, 1, N)     e.g. (1, 512, 1, 65)
        """
        # (B, C, 1, N) -> (B, C, 1, N)
        # e.g. (1, 256, 1, 7) -> (1, 256, 1, 7)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # (B, C, 1, N) -> (B, H, D, N)
        # e.g. (1, 256, 1, 7) -> (1, 8, 32, 7)
        q = q.reshape(q.data.shape[0], self.n_heads, -1, q.data.shape[-1])
        k = k.reshape(k.data.shape[0], self.n_heads, -1, k.data.shape[-1])
        v = v.reshape(v.data.shape[0], self.n_heads, -1, v.data.shape[-1])

        # 1. q transposed from (B, H, D, N) to (B, H, N, D).  e.g. (1, 8, 32, 7) -> (1, 8, 7, 32)
        # 2. q.T matmul k. (B, H, N_k, D_k) * (B, H, D_q, N_q) -> (B, H, N_k, N_q)
        #             e.g. (1, 8, 7, 32) * (1, 8, 32, 7) -> (1, 8, 7, 7)
        qk = q.transpose(-1, -2) @ k
        qk = qk * self.scale
        attn = qk.softmax(-1)

        # e.g. (1, 8, 7, 7) * (1, 8,  7,  32) -> (1, 8, 7, 32)
        out = attn @ v.transpose(-1, -2)
        # (1, 8, 7, 32) -> (1, 8, 32, 7)
        out = out.transpose(-1, -2)

        # (B, H, D, N) -> (B, C, 1, N)
        # e.g. (1, 8, 32, 7) -> (1, 256, 1, 7)
        out = out.reshape(out.data.shape[0], -1, 1, out.data.shape[-1])

        out = self.out_proj(out)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_state = ['q_proj.weight', 'k_proj.weight', 'v_proj.weight', 'out_proj.weight']
        for name in weight_state:
            key = prefix + name
            if key in state_dict.keys():
                val = state_dict[key]
                if val.shape != self.state_dict().get(name).shape:
                    state_dict[key] = val.unsqueeze(-1).unsqueeze(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class MLP(MLPBlock):
    def __init__(self, in_features, hidden_features=None, out_features=None, act=torch.nn.ReLU):
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        super().__init__(in_features, hidden_features, act)
        self.lin1 = nn.Conv2d(in_features, hidden_features, 1)
        self.lin2 = nn.Conv2d(hidden_features, out_features, 1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_state = ['lin1.weight', 'lin2.weight']
        for name in weight_state:
            key = prefix + name
            if key in state_dict.keys():
                val = state_dict[key]
                if val.shape != self.state_dict().get(name).shape:
                    state_dict[key] = val.unsqueeze(-1).unsqueeze(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads, downsample_rate, skip_first_pe=False):
        super().__init__()
        self.self_attn = Attention(dim, n_heads)
        self.norm1 = amct_nn.LayerNorm(dim)
        self.cross_attn_token_to_image = Attention(dim, n_heads, downsample_rate)
        self.norm2 = amct_nn.LayerNorm(dim)

        self.mlp = MLP(dim, dim * 8)
        self.norm3 = amct_nn.LayerNorm(dim)

        self.norm4 = amct_nn.LayerNorm(dim)
        self.cross_attn_image_to_token = Attention(dim, n_heads, downsample_rate)

        self.skip_first_pe = skip_first_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_pe:
            # (1, 256, 1, 7) -> (1, 256, 1, 7)
            queries = self.self_attn(queries, queries, queries)
        else:
            # (1, 256, 1, 7) -> (1, 256, 1, 7)
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        # (1, 256, 1, 7) -> (1, 256, 1, 7)
        attn_out = self.cross_attn_token_to_image(q, k, keys)

        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        # (1, 256, 1, 7) -> (1, 256, 1, 784)
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, n_heads: int, attn_downsample_rate: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(dim, n_heads, attn_downsample_rate, i == 0))
        self.final_attn_token_to_image = Attention(dim, n_heads, attn_downsample_rate)
        self.norm_final_attn = amct_nn.LayerNorm(dim)

    def forward(self, src: Tensor, pos: Tensor, tokens: Tensor):
        # (1, 7, 256) -> (1, 256, 7) -> (1, 256, 1, 7)
        tokens = tokens.permute(0, 2, 1).unsqueeze(-2)

        queries = tokens
        keys = src

        for layer in self.layers:
            # queries: (1, 256, 1, 7), keys: (1, 256, 28, 28)
            queries, keys = layer(queries=queries, keys=keys, query_pe=tokens, key_pe=pos)

        q = queries + tokens
        k = keys + pos
        # (1, 256, 1, 7) -> (1, 256, 1, 7)
        attn_out = self.final_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        # (1, 256, 1, 7) -> (1, 256, 1, 7)
        queries = self.norm_final_attn(queries)
        return queries, keys


class HyperMLP(nn.Module):
    def __init__(self, dim: int, out_dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers-1):
            self.layers.append(nn.Conv2d(dim, dim, 1))
        self.layers.append(nn.Conv2d(dim, out_dim, 1))

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = torch.relu(x)
        x = self.layers[-1](x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_state = ['layers.0.weight', 'layers.1.weight', 'layers.2.weight']
        for name in weight_state:
            key = prefix + name
            if key in state_dict.keys():
                val = state_dict[key]
                if val.shape != self.state_dict().get(name).shape:
                    state_dict[key] = val.unsqueeze(-1).unsqueeze(-1)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class MaskDecoder(nn.Module):
    def __init__(self, dim: int, image_size):
        super().__init__()
        self.image_size = image_size
        self.iou_token = nn.Embedding(1, dim)
        self.mask_tokens = nn.Embedding(4, dim)
        self.transformer = TwoWayTransformer(dim, 2, 8)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 4, 2, stride=2),
            amct_nn.LayerNorm(dim // 4),
            amct_nn.GELU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, 2, stride=2),
            amct_nn.GELU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList()
        for _ in range(4):
            self.output_hypernetworks_mlps.append(HyperMLP(dim, dim // 8, 3))

        self.iou_prediction_head = HyperMLP(dim, 4, 3)
        self.image_pe = None
        self.dense_embedding = None

    def set_image_pe(self, image_pe):
        self.image_pe = image_pe.reshape(image_pe.data.shape[0], image_pe.data.shape[1], 1, -1)

    def set_dense_embedding(self, dense_embedding):
        self.dense_embedding = dense_embedding.detach()

    def forward(self, image_embeddings: Tensor, sparse_embedding: Tensor):
        # concat([(1, 256), (4, 256)]) -> (5, 256)
        output_tokens = torch.cat([self.iou_token.weight.data, self.mask_tokens.weight.data], dim=0)
        # (5, 256) -> (1, 5, 256)
        output_tokens = output_tokens.unsqueeze(0)
        # concat([(1, 5, 256), (1, 2, 256)]) -> (1, 7, 256)
        tokens = torch.cat([output_tokens, sparse_embedding], dim=1)

        # (1, 256, 28, 28) -> (1, 256, 1, 784)
        src = image_embeddings + self.dense_embedding
        src_n, src_c, src_h, src_w = src.data.shape
        src = src.reshape(src_n, src_c, 1, -1)

        # (1, 256, 1, 784) -> (1, 256, 1, 7), (1, 256, 1, 784)
        hs, src = self.transformer(src, self.image_pe, tokens)
        # (1, 256, 1, 784) -> (1, 256, 28, 28) -> (1, 32, 112, 112)
        src = src.reshape(src_n, src_c, src_h, src_w)
        upscaled = self.output_upscaling(src)

        # (1, 256, 1, 7) -> (1, 256, 1, 1)
        iou_token_out = hs[..., 0:1]
        # (1, 256, 1, 7) -> (1, 256, 1, 4)
        mask_tokens_out = hs[..., 1: 5]

        hyper_in_list = []
        for i in range(4):
            # (1, 256, 1, 1) -> (1, 32, 1, 1)
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[..., i:i + 1]))
        # (1, 32, 1, 1) -> (1, 32, 1, 4)
        hyper_in = torch.cat(hyper_in_list, dim=-1)

        hight, width = upscaled.data.shape[-2], upscaled.data.shape[-1]
        # (1, 32, 1, 4) -> (1, 32, 4) -> (1, 4, 32)
        hyper_in = hyper_in.flatten(-2).transpose(-1, -2)
        # (1, 32, 112, 112) -> (1, 32, 12544)
        upscaled = upscaled.flatten(-2)
        # (1, 4, 32) * (1, 32, 12544) -> (1, 4, 12544)
        masks = hyper_in @ upscaled
        # (1, 4, 12544) -> (1, 4, 112, 112)
        masks = masks.reshape(1, -1, hight, width)

        # (1, 256, 1, 1) -> (1, 4, 1, 1)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return post_process(masks, iou_pred, self.image_size)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_state = ['layers.0.weight', 'layers.1.weight', 'layers.2.weight']
        local_state_dict = {}
        for key in self.state_dict().keys():
            name = prefix + key
            if name in state_dict.keys():
                local_state_dict[key] = state_dict.get(name)

        for name in weight_state:
            key = prefix + name
            if key in state_dict.keys():
                val = state_dict[key]
                if val.shape != self.state_dict().get(name).shape:
                    local_state_dict[key] = val.unsqueeze(-1).unsqueeze(-1)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
