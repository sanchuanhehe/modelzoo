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

#ifndef GROUP_POINTS_TILING_H
#define GROUP_POINTS_TILING_H
#include "common.h"

using namespace ge;
namespace optiling {
BEGIN_TILING_DATA_DEF(GroupPointsTilingData)

    ADD_TILING_DATA_PARAM(uint32_t, batch);
    ADD_TILING_DATA_PARAM(uint32_t, channel);
    ADD_TILING_DATA_PARAM(uint32_t, totalPoints);
    ADD_TILING_DATA_PARAM(uint32_t, numPoints);
    ADD_TILING_DATA_PARAM(uint32_t, numSample);
    ADD_TILING_DATA_PARAM(uint32_t, chnSizeAligned);
    ADD_TILING_DATA_PARAM(uint32_t, ubMaxTasks);
    ADD_TILING_DATA_PARAM(uint32_t, numCoreWork);
    ADD_TILING_DATA_PARAM(uint32_t, tasksPerCore);
    ADD_TILING_DATA_PARAM(uint32_t, tasksOnLastCore);
    ADD_TILING_DATA_PARAM(uint32_t, itersPerMainCore);
    ADD_TILING_DATA_PARAM(uint32_t, tailTaskNumOnMainCore);
    ADD_TILING_DATA_PARAM(uint32_t, itersOnLastCore);
    ADD_TILING_DATA_PARAM(uint32_t, tailTaskNumOnLastCore);
    ADD_TILING_DATA_PARAM(uint32_t, tailTaskNumAligned);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupPoints, GroupPointsTilingData)

graphStatus TilingFunctionForGroupPoints(gert::TilingContext* tilingContext);
} // namespace optiling
#endif // GROUP_POINTS_GRAD_TILING_H