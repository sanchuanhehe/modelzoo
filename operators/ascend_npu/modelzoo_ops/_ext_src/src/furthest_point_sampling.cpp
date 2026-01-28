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

#include "functions.h"
#include "common.h"
#include "pytorch_npu_helper.hpp"

at::Tensor npu_furthest_point_sampling(const at::Tensor& xyz_trans, const at::Tensor& dist, int32_t nsample)
{
    TORCH_CHECK(is_npu(xyz_trans), "points must be NPU tensor");
    TORCH_CHECK(is_npu(dist), "idx must be NPU tensor");
    at::Tensor out_idx = at::empty({static_cast<int64_t>(xyz_trans.sizes()[0]), static_cast<int64_t>(nsample)},
        dist.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnFurthestPointSampling, xyz_trans, dist, nsample, out_idx);
    return out_idx;
}
