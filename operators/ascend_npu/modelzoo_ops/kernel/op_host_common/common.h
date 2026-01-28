/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain valueA copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_HOST_COMMON_H
#define OP_HOST_COMMON_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

#define ADD_TILING_DATA_PARAM TILING_DATA_FIELD_DEF

template<typename typeT>
inline typeT FloorDivide(typeT dividend, typeT divisor)
{
    return divisor == 0 ? dividend : dividend / divisor;
}

template<typename typeT>
inline typeT AlignFloor(typeT value, typeT alignment)
{
    return alignment == 0 ? 0 : value / alignment * alignment;
}

template<typename typeT>
inline typeT CeilDivide(typeT dividend, typeT divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

template<typename typeT>
inline typeT AlignUp(typeT value, typeT alignment)
{
    if (alignment == 0) {
        return 0;
    }
    return CeilDivide(value, alignment) * alignment;
}

#endif // OP_HOST_COMMON_H