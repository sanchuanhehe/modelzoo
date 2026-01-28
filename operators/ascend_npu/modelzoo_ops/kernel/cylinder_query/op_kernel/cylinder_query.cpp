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
#include "cylinder_query.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void cylinder_query(GM_ADDR seedXyz, GM_ADDR pointXyz, GM_ADDR rotMat,
    GM_ADDR defaultIdx, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingData)
{
    GET_TILING_DATA(cylinderQueryTilingData, tilingData);

    TilingPara tilingPara;
    tilingPara.batch = cylinderQueryTilingData.batch;
    tilingPara.numPoints = cylinderQueryTilingData.numPoints;
    tilingPara.numQueries = cylinderQueryTilingData.numQueries;
    tilingPara.radius = cylinderQueryTilingData.radius * cylinderQueryTilingData.radius;
    tilingPara.hMin = cylinderQueryTilingData.hMin;
    tilingPara.hMax = cylinderQueryTilingData.hMax;
    tilingPara.nSample = cylinderQueryTilingData.nSample;
    tilingPara.pointsPerTask = cylinderQueryTilingData.pointsPerTask;
    tilingPara.blocksPerTask = cylinderQueryTilingData.blocksPerTask;
    tilingPara.totalIterations = cylinderQueryTilingData.totalIterations;
    tilingPara.lastIterationPoints = cylinderQueryTilingData.lastIterationPoints;
    tilingPara.lastIterationBlocks = cylinderQueryTilingData.lastIterationBlocks;
    tilingPara.tasksPerCore = cylinderQueryTilingData.tasksPerCore;
    tilingPara.tasksOnLastCore = cylinderQueryTilingData.tasksOnLastCore;
    tilingPara.pointsPerCore = cylinderQueryTilingData.pointsPerCore;

    CylinderQueryKernel op(seedXyz, pointXyz, rotMat, defaultIdx, output, &tilingPara);
    op.Process();
}