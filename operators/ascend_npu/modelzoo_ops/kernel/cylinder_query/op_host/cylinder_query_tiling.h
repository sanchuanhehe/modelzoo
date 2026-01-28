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
#ifndef CYLINDER_QUERY_TILING_H
#define CYLINDER_QUERY_TILING_H
#include "common.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CylinderQueryTilingData)
    ADD_TILING_DATA_PARAM(uint32_t, batch);
    ADD_TILING_DATA_PARAM(uint32_t, numPoints); // 点云数量 n
    ADD_TILING_DATA_PARAM(uint32_t, numQueries); // 查询点数量
    ADD_TILING_DATA_PARAM(float, radius);
    ADD_TILING_DATA_PARAM(float, hMin);
    ADD_TILING_DATA_PARAM(float, hMax);
    ADD_TILING_DATA_PARAM(uint32_t, nSample);
    ADD_TILING_DATA_PARAM(uint32_t, pointsPerTask);
    ADD_TILING_DATA_PARAM(uint32_t, blocksPerTask);
    ADD_TILING_DATA_PARAM(uint32_t, totalIterations);
    ADD_TILING_DATA_PARAM(uint32_t, lastIterationPoints);
    ADD_TILING_DATA_PARAM(uint32_t, lastIterationBlocks);
    ADD_TILING_DATA_PARAM(uint32_t, tasksPerCore);
    ADD_TILING_DATA_PARAM(uint32_t, tasksOnLastCore);
    ADD_TILING_DATA_PARAM(uint32_t, pointsPerCore);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CylinderQuery, CylinderQueryTilingData)

ge::graphStatus TilingFunctionForCylinderQuery(gert::TilingContext* context);
} // namespace optiling
#endif // CYLINDER_QUERY_TILING_H