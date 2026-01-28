/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common.h"
#include "functions.h"
#include "pytorch_npu_helper.hpp"

static constexpr uint32_t B_IDX = 0;
static constexpr uint32_t N_IDX = 1;
static constexpr uint32_t M_IDX = 1;

at::Tensor npu_cylinder_query(const at::Tensor& seed_xyz, const at::Tensor& point_xyz, const at::Tensor& rot_mat,
    const at::Tensor& default_idx, double radius, double h_min, double h_max, int nsample)
{
    TORCH_CHECK(is_npu(seed_xyz), "seed_xyz must be NPU tensor");
    TORCH_CHECK(is_npu(point_xyz), "point_xyz must be NPU tensor");
    TORCH_CHECK(is_npu(rot_mat), "rot_mat must be NPU tensor");
    TORCH_CHECK(seed_xyz.dim() == 3, "invalid dimension of seed_xyz, dim: ", seed_xyz.dim());
    TORCH_CHECK(point_xyz.dim() == 3, "invalid dimension of point_xyz, dim: ", point_xyz.dim());

    uint32_t b = static_cast<uint32_t>(seed_xyz.size(B_IDX));
    uint32_t n = static_cast<uint32_t>(point_xyz.size(N_IDX));
    uint32_t m = static_cast<uint32_t>(seed_xyz.size(M_IDX));

    at::Tensor out = at::empty({b, m, n}, seed_xyz.options());
    EXEC_NPU_CMD(aclnnCylinderQuery, seed_xyz, point_xyz, rot_mat, default_idx, radius, h_min, h_max, nsample, out);
    return out;
}