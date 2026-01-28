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
from torch.autograd import Function
import torch_npu
import modelzoo_ops._ext

class CylinderQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, h_min: float, h_max: float, nsample: int,
                seed_xyz: torch.Tensor, point_xyz: torch.Tensor, rot_mat: torch.Tensor, default_idx: torch.Tensor):
        out = modelzoo_ops._ext.npu_cylinder_query(seed_xyz, point_xyz, rot_mat, default_idx, radius, h_min, h_max, nsample)
        return out

    @staticmethod
    def symbolic(g, radius: float, h_min: float, h_max: float, nsample: int,
                seed_xyz: torch.Tensor, point_xyz: torch.Tensor, rot_mat: torch.Tensor, default_idx: torch.Tensor):
        return g.op(
            "npu::CylinderQuery",
            seed_xyz,
            point_xyz,
            rot_mat,
            default_idx,
            radius_f = radius,
            h_min_f = h_min,
            h_max_f = h_max,
            nsample_i = nsample,
            outputs = 1)

def sort_idx(out_idx: torch.Tensor, nsample: int) -> torch.Tensor:
    _, _, num_points = out_idx.shape
    # 从K个候选邻域中选择最近的nsample个点
    out_idx = out_idx.sort(dim=-1)[0][:, :, :nsample].long()
    mask = out_idx >= num_points
    # 将无效索引（mask=True）替换为0, 有效索引保持不变
    valid_idx= torch.where(mask, 0, out_idx)
    # 替换溢出索引
    out_idx = torch.where(mask, valid_idx[..., 0:1], out_idx).to(dtype=torch.int32)
    return out_idx

def cylinder_query(seed_xyz: torch.Tensor, point_xyz: torch.Tensor, rot_mat: torch.Tensor, radius: float,
                   h_min: float, h_max: float, nsample: int) -> torch.Tensor:
    """
    在给定点云中查询每个点在指定旋转圆柱领域内的点

    seed_xyz (torch.Tensor): 查询点坐标，shape为(batch, num_queries, 3)
    point_xyz (torch.Tensor): 输入点云坐标，shape为(batch, num_points, 3)
    rot_mat (torch.Tensor): 旋转矩阵张量，shape为(batch, num_queries, 9)
    radius (float): 圆柱底面半径
    h_min (float): 圆柱的最小高度
    h_max (float): 圆柱的最大高度
    nsample (int): 每个查询点最多采样点数

    out (torch.Tensor): 每个查询点邻域内点的索引，shape为(batch, num_queries, nsample)
    """
    _, num_points, _ = point_xyz.size()
    default_idx = torch.arange(0, num_points, dtype = torch.float, device = seed_xyz.device)
    out_idx = CylinderQuery.apply(radius, h_min, h_max, nsample, seed_xyz, point_xyz, rot_mat, default_idx)
    return sort_idx(out_idx, nsample)