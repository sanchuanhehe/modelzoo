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

#ifndef EXT_COMMON_H_
#define EXT_COMMON_H_

#include <ATen/ATen.h>
#include "third_party/acl/inc/acl/acl_base.h"

inline bool is_npu(const at::Tensor& tensor)
{
#ifdef COMPILE_WITH_XLA
    return tensor.device().type() == at::kXLA;
#else
    return tensor.device().type() == at::kPrivateUse1;
#endif
}
#endif // EXT_COMMON_H_
