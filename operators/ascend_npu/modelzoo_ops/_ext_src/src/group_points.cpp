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
static constexpr uint32_t C_IDX = 2;
static constexpr uint32_t N_POINT_IDX = 1;
static constexpr uint32_t N_SAMPLE_IDX = 2;

at::Tensor npu_group_points(const at::Tensor& features, const at::Tensor& index)
{
    TORCH_CHECK(is_npu(features), "features must be NPU tensor");
    TORCH_CHECK(is_npu(index), "index must be NPU tensor");
    TORCH_CHECK(features.scalar_type() == at::kFloat,
        "features must be a float32 tensor in op group_points.")
    TORCH_CHECK(features.dim() == 3, "invalid dimension of features, dim: ", features.dim());
    TORCH_CHECK(index.dim() == 3, "invalid dimension of index, dim: ", index.dim());
    TORCH_CHECK(features.size(0) == index.size(0), "first dimension of features and index is not same.")

    uint32_t b = static_cast<uint32_t>(features.size(B_IDX));
    uint32_t c = static_cast<uint32_t>(features.size(C_IDX));
    uint32_t nPoint = static_cast<uint32_t>(index.size(N_POINT_IDX));
    uint32_t nSample = static_cast<uint32_t>(index.size(N_SAMPLE_IDX));

    at::Tensor out_points = at::empty({b, c, nPoint, nSample}, features.options());

    EXEC_NPU_CMD(aclnnGroupPoints, features, index, out_points);
    return out_points;
}