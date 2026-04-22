# Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import MaxSigmoidAttnBlock, BNContrastiveHead, C2f, C2fAttn, ImagePoolingAttn
from ultralytics.nn.modules.head import Detect, WorldDetect
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils.tal import make_anchors


TEXT_OFFLINE = False
CUSTOM_RPN_TOPK = 300
CUSTOM_RPN = False


def max_sigmoid_attn_block_forward(self, x, guide):
    """
    Custom forward function for MaxSigmoidAttnBlock.
    """
    bs, _, h, w = x.data.shape
    _, num_query, channel = guide.data.shape
    guide = guide.reshape(-1, channel)
    guide = self.gl(guide).reshape(bs, num_query, -1)
    guide = guide.view(bs, num_query, self.nh, self.hc).transpose(1, 2)

    embed = self.ec(x) if self.ec is not None else x
    embed = embed.view(bs, self.nh * self.hc, 1, h * w).view(bs, self.nh, self.hc, h * w)

    if TEXT_OFFLINE:
        guide = guide.data + embed.max() * 0

    aw = guide @ embed
    aw = aw * (self.hc ** (-0.5))
    aw = aw.max(dim=-2, keepdims=True)[0]
    aw = aw + self.bias[None, :, None, None]
    aw = aw.sigmoid() if self.scale == 1.0 else aw.sigmoid() * self.scale

    x = self.proj_conv(x)
    c_out = x.data.shape[1]
    aw = aw.repeat(1, 1, c_out // self.nh, 1).reshape(bs, c_out, 1, -1).reshape(bs, c_out, h, w)
    return x * aw


def bn_contrastive_head_forward(self, x, w):
    """
    Custom forward function for BNContrastiveHead.
    """
    x = self.norm(x)
    w = F.normalize(w, dim=-1, p=2)
    if TEXT_OFFLINE:
        w = w.data
    batch, channel, height, width = x.data.shape
    x = x.reshape(batch, channel, -1)
    y = (w @ x).reshape(batch, w.data.shape[1], height, width)
    return y * self.logit_scale.exp().detach().data + self.bias

def c2f_forward_split(self, x):
    """
    Custom forward function for C2f.
    """
    y0, y1 = self.cv1(x).split((self.c, self.c), 1)
    y = [y0, y1]
    for m in self.m:
        y.append(m(y[-1]))
    return self.cv2(torch.cat(y, 1))


def c2f_attn_forward_split(self, x, guide):
    """
    Custom forward function for C2fAttn.
    """
    y0, y1 = self.cv1(x).split((self.c, self.c), 1)
    y = [y0, y1]
    for m in self.m:
        y.append(m(y[-1]))
    y.append(self.attn(y[-1], guide))
    return self.cv2(torch.cat(y, 1))


@torch.fx.wrap
def txt_feats_repeat(x, txt_feats):
    if len(txt_feats) != len(x):
        txt_feats = txt_feats.repeat(len(x), 1, 1)
    return txt_feats


def world_model_forward(self, x, txt_feats):
    return self.predict(x, txt_feats)


def world_model_predict(self, x, txt_feats, embed=None):
    """
    Custom forward function for WorldModel.
    """
    txt_feats = txt_feats_repeat(x, txt_feats)
    ori_txt_feats = txt_feats.clone()

    y, _, embeddings = [], [], []
    for m in self.model:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        if isinstance(m, C2fAttn):
            x = m(x, txt_feats)
        elif isinstance(m, WorldDetect):
            x = m(x, ori_txt_feats)
        elif isinstance(m, ImagePoolingAttn):
            txt_feats = m(x, txt_feats)
        else:
            x = m(x)
        
        y.append(x if m.i in self.save else None)
        if embed and m.i in embed:
            embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
            if m.i == max(embed):
                return torch.unbind(torch.cat(embeddings, 1), dims=0)
    return x


@torch.fx.wrap
def dist2bbox_stride(distance, anchor_points, strides, cls, nc, dim=-1):
    anchor_points = anchor_points.unsqueeze(0)
    lt, rb = torch.split(distance, 2, 1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    c_xy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    dbox = torch.cat((c_xy, wh), dim)
    y = torch.cat((dbox * strides, cls.sigmoid()), 1)
    return y


@torch.fx.wrap
def post_process(x, no, nc, stride, reg_max):
    shape = x[0].data.shape
    x_cat = torch.cat([xi.view(shape[0], no, -1) for xi in x], 2)

    anchors, strides = (x.transpose(0, 1) for x in make_anchors(x, stride, 0.5))
    anchors, strides = anchors.data, strides.data

    box, cls = x_cat.split((reg_max * 4, nc), 1)
    return anchors, strides, box, cls


def detect_forward(self, x):
    """
    Custom forward function for Detect.
    """
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    no = self.nc + self.reg_max * 4
    anchors, strides, box, cls = post_process(x, no, self.nc, self.stride, self.reg_max)
    y = dist2bbox_stride(self.dfl(box), anchors, strides, cls, self.nc, dim=1)
    return y


def world_detect_forward(self, x, text):
    """
    Custom forward function for WorldDetect.
    """
    for i in range(self.nl):
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
    no = self.nc + self.reg_max * 4
    anchors, strides, box, cls = post_process(x, no, self.nc, self.stride, self.reg_max)
    y = dist2bbox_stride(self.dfl(box), anchors, strides, cls, self.nc, dim=1)
    return y


DEPLOY_MODULE_MAPPINGS = {
    MaxSigmoidAttnBlock: {"forward": max_sigmoid_attn_block_forward},
    BNContrastiveHead: {"forward": bn_contrastive_head_forward},
    C2f: {"forward": c2f_forward_split, "forward_split": c2f_forward_split},
    C2fAttn: {"forward": c2f_attn_forward_split, "forward_split": c2f_attn_forward_split},
    Detect: {"forward": detect_forward},
    WorldDetect: {"forward": world_detect_forward},
    WorldModel: {"predict": world_model_predict, "forward": world_model_forward},
}

    
