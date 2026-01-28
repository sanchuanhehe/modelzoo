# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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
import torch_npu
from torch.autograd import Function
import modelzoo_ops._ext


class GroupPoints(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, index: torch.Tensor):
        output = modelzoo_ops._ext.npu_group_points(features, index)
        return output

    @staticmethod
    def symbolic(g, features: torch.Tensor, index: torch.Tensor):
        return g.op(
            "npu::GroupPoints",
            features,
            index,
            outputs = 1)

def group_points(points: torch.Tensor, index: torch.Tensor):
    """
    根据索引对特征进行分组操作

    points (torch.Tensor): 输入特征，shape为(batch, channel, totol_points)
    index (torch.Tensor): 索引张量，shape为(batch, num_points, num_sample)

    output (torch.Tensor): 分组后的特征，shape为(batch, channel, num_points, num_sample)
    """
    trans_features = points.contiguous()
    index = index.contiguous()

    batch, channel, _ = points.shape
    _, num_points, num_sample = index.shape

    trans_features = trans_features.transpose(1, 2)
    features = trans_features.contiguous()

    out = GroupPoints.apply(features, index)
    return out.view(batch, num_points, num_sample, channel).permute(0, 3, 1, 2)
