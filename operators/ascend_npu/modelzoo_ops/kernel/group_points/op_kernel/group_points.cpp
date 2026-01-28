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

#include "group_points.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void group_points(
    GM_ADDR features, GM_ADDR index, GM_ADDR outPoints, GM_ADDR workspace, GM_ADDR tilingData)
{
    GET_TILING_DATA(groupPointsTilingData, tilingData);
    TilingPara tilingPara;

    tilingPara.batch = groupPointsTilingData.batch;
    tilingPara.channel = groupPointsTilingData.channel;
    tilingPara.totalPoints = groupPointsTilingData.totalPoints;
    tilingPara.numPoints = groupPointsTilingData.numPoints;
    tilingPara.numSample = groupPointsTilingData.numSample;
    tilingPara.chnSizeAligned = groupPointsTilingData.chnSizeAligned;
    tilingPara.ubMaxTasks = groupPointsTilingData.ubMaxTasks;
    tilingPara.numCoreWork = groupPointsTilingData.numCoreWork;
    tilingPara.tasksPerCore = groupPointsTilingData.tasksPerCore;
    tilingPara.tasksOnLastCore = groupPointsTilingData.tasksOnLastCore;
    tilingPara.itersPerMainCore = groupPointsTilingData.itersPerMainCore;
    tilingPara.tailTaskNumOnMainCore = groupPointsTilingData.tailTaskNumOnMainCore;
    tilingPara.itersOnLastCore = groupPointsTilingData.itersOnLastCore;
    tilingPara.tailTaskNumOnLastCore = groupPointsTilingData.tailTaskNumOnLastCore;
    tilingPara.tailTaskNumAligned = groupPointsTilingData.tailTaskNumAligned;

    KernelGroupPoints op(features, index, outPoints, &tilingPara);
    op.Process();
}