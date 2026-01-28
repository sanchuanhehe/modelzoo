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
#ifndef EXT_FUNCTIONS_H_
#define EXT_FUNCTIONS_H_

#include <ATen/ATen.h>

at::Tensor npu_group_points(const at::Tensor& features, const at::Tensor& index);

at::Tensor npu_furthest_point_sampling(const at::Tensor& xyz_trans, const at::Tensor& dist, int32_t nsample);

at::Tensor npu_cylinder_query(const at::Tensor& seed_xyz, const at::Tensor& point_xyz, const at::Tensor& rot_mat,
    const at::Tensor& default_idx, double radius, double h_min, double h_max, int nsample);

#endif // EXT_FUNCTIONS_H_
