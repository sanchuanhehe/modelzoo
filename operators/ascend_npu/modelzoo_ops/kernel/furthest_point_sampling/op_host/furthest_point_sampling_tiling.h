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

#ifndef FURTHEST_POINT_SAMPLING_TILING_H
#define FURTHEST_POINT_SAMPLING_TILING_H
#include "common.h"

using namespace ge;
namespace optiling {
BEGIN_TILING_DATA_DEF(FurthestPointSamplingTilingData)
    ADD_TILING_DATA_PARAM(uint32_t, batchSize);
    ADD_TILING_DATA_PARAM(uint32_t, totalPoints);
    ADD_TILING_DATA_PARAM(uint32_t, nSample);
    ADD_TILING_DATA_PARAM(uint32_t, numTile);
    ADD_TILING_DATA_PARAM(uint32_t, tileSize);
    ADD_TILING_DATA_PARAM(uint32_t, lastTileSize);
    ADD_TILING_DATA_PARAM(uint32_t, ubWorkSpaceSize);
    ADD_TILING_DATA_PARAM(uint32_t, pairBufSize);
    ADD_TILING_DATA_PARAM(uint32_t, batchsPerLargeCore);
    ADD_TILING_DATA_PARAM(uint32_t, batchsPerLittleCore);
    ADD_TILING_DATA_PARAM(uint32_t, numLargeCore);
    ADD_TILING_DATA_PARAM(uint32_t, repeatsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FurthestPointSampling, FurthestPointSamplingTilingData)

graphStatus TilingFunctionForFurthestPointSampling(gert::TilingContext* tilingContext);
}

#endif // FURTHEST_POINT_SAMPLING_TILING_H