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

#include "furthest_point_sampling.h"

using namespace AscendC;

// Entrance of kernel
extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR xyzTrans, GM_ADDR dist, GM_ADDR outIdx, GM_ADDR workspace, GM_ADDR tilingData)
{
    GET_TILING_DATA(furthestPointSamplingTilingData, tilingData);

    TilingPara tilingPara;

    tilingPara.totalPoints = furthestPointSamplingTilingData.totalPoints;
    tilingPara.batchSize = furthestPointSamplingTilingData.batchSize;
    tilingPara.nSample = furthestPointSamplingTilingData.nSample;
    tilingPara.numTile = furthestPointSamplingTilingData.numTile;
    tilingPara.tileSize = furthestPointSamplingTilingData.tileSize;
    tilingPara.lastTileSize = furthestPointSamplingTilingData.lastTileSize;
    tilingPara.ubWorkSpaceSize = furthestPointSamplingTilingData.ubWorkSpaceSize;
    tilingPara.pairBufSize = furthestPointSamplingTilingData.pairBufSize;
    tilingPara.batchsPerLargeCore = furthestPointSamplingTilingData.batchsPerLargeCore;
    tilingPara.batchsPerLittleCore = furthestPointSamplingTilingData.batchsPerLittleCore;
    tilingPara.numLargeCore = furthestPointSamplingTilingData.numLargeCore;
    tilingPara.repeatsPerBlock = furthestPointSamplingTilingData.repeatsPerBlock;

    KernelFurthestPointSampling op(xyzTrans, dist, outIdx, workspace, &tilingPara);
    op.Process();
}