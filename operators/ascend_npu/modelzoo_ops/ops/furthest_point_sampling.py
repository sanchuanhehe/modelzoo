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

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz_trans: torch.Tensor, dist: torch.Tensor, nsample: int):
        if (torch.numel(xyz_trans) == 0 | torch.numel(dist) == 0):
            raise Exception("Empty input tensor\n")
        
        if (nsample == 0):
            raise Exception("nsample is zero\n")

        return modelzoo_ops._ext.npu_furthest_point_sampling(xyz_trans, dist, nsample)

    @staticmethod
    def symbolic(g, xyz_trans: torch.Tensor, dist: torch.Tensor, nsample: int):
        return g.op(
            "npu::FurthestPointSampling",
            xyz_trans,
            dist,
            nsample_i = nsample,
            outputs = 1)

def furthest_point_sampling(xyz: torch.Tensor, nsample: int):
    """
    最远点采样

    xyz (torch.Tensor): 输入点云，shape为 (batch, total_points, 3)
    nsample (int): 采样点的数量

    out (torch.Tensor): 采样点的索引，shape为 (batch, nsample)
    """
    batch, total_points, _ = xyz.size()
    xyz_trans = xyz.permute(0, 2, 1).contiguous()

    dist = torch.full((batch, total_points), 1e10, dtype=xyz_trans.dtype, device="npu").contiguous()
    return FurthestPointSampling.apply(xyz_trans, dist, nsample)
