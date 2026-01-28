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
#include "cylinder_query_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace gert;
using namespace platform_ascendc;
namespace optiling {
// 属性下标
static constexpr uint32_t RADIUS_IDX = 0;
static constexpr uint32_t H_MIN_IDX = 1;
static constexpr uint32_t H_MAX_IDX = 2;
static constexpr uint32_t NSAMPLE_IDX = 3;

static constexpr uint32_t BATCH_IDX = 0;
static constexpr uint32_t POINTS_IDX = 1;
static constexpr uint32_t QUERY_IDX = 1;

static constexpr uint32_t UB_OVERHEAD = 1000; // UB预留空间
static constexpr uint32_t NSAMPLE_OVERHEAD = 8; // nsample相关开销

static constexpr uint32_t FLOAT_SIZE = 4;
static constexpr uint32_t POINTS_PER_BLOCK = 8;
static constexpr uint32_t BYTES_PER_BLOCK = 96; // POINTS_PER_BLOCK * 3 * FLOAT_SIZE

struct CylinderQueryTilingPara {
    uint32_t pointsPerTask;         // 单次点数
    uint32_t blocksPerTask;         // 单次最大块数
    uint32_t totalIterations;       // 总迭代次数
    uint32_t lastIterationPoints;   // 单次最大块数
    uint32_t lastIterationBlocks;   // 单次点数
    uint32_t availableCores;        // 可用核数
    uint32_t tasksPerCore;          // 每核心任务数
    uint32_t tasksOnLastCore;       // 尾核任务数
    uint32_t pointsPerCore;         // 尾核点数
};

static bool ValidateShapes(TilingContext* tilingContext, const StorageShape*& seedShape, 
    const StorageShape*& pointsShape)
{
    seedShape = tilingContext->GetInputShape(0);
    pointsShape = tilingContext->GetInputShape(1);
    return seedShape != nullptr && pointsShape != nullptr;
}

static bool ParseAttributes(const RuntimeAttrs* attr, float& radius, float& hMin,
    float& hMax, uint32_t& nSample)
{
    if (attr == nullptr) return false;

    radius = *(attr->GetAttrPointer<float>(RADIUS_IDX));
    hMin = *(attr->GetAttrPointer<float>(H_MIN_IDX));
    hMax = *(attr->GetAttrPointer<float>(H_MAX_IDX));
    nSample = *(attr->GetAttrPointer<uint32_t>(NSAMPLE_IDX));
    return true;
}

static void CalculateTaskConfig(CylinderQueryTilingPara& tilingPara, uint32_t totalPoints,
    uint64_t ubSize, uint32_t nSample)
{
    // 计算块数
    uint32_t blockCount = CeilDivide(totalPoints, POINTS_PER_BLOCK);

    // 计算单次最大处理块数
    uint32_t availableUB = ubSize / 2;
    uint32_t overhead = UB_OVERHEAD + nSample * NSAMPLE_OVERHEAD;
    uint32_t blockMemoryCost = 340; // 每块内存消耗

    tilingPara.blocksPerTask = (availableUB - overhead) / blockMemoryCost;
    tilingPara.blocksPerTask = std::max(tilingPara.blocksPerTask, 1u);

    // 计算单次处理点数
    tilingPara.pointsPerTask = (tilingPara.blocksPerTask * BYTES_PER_BLOCK) / (FLOAT_SIZE * 3);
    tilingPara.pointsPerTask = std::min(tilingPara.pointsPerTask, totalPoints);

    // 计算迭代次数
    uint32_t totalIterations = blockCount / tilingPara.blocksPerTask;
    tilingPara.totalIterations = (blockCount % tilingPara.blocksPerTask == 0 ? totalIterations : totalIterations + 1);

    // 计算最后一次迭代参数
    uint32_t lastIterationPoints = totalPoints - (totalIterations * tilingPara.pointsPerTask);
    tilingPara.lastIterationPoints = (lastIterationPoints == 0) ? tilingPara.pointsPerTask : lastIterationPoints;
    // 最后一次循环中参与计算的元素块数
    uint32_t lastIterationBlocks = CeilDivide(tilingPara.lastIterationPoints, POINTS_PER_BLOCK);
    tilingPara.lastIterationBlocks = (lastIterationBlocks == 0)? tilingPara.blocksPerTask: lastIterationBlocks; 
}

static void CalculateCoreDistribution(CylinderQueryTilingPara& tilingPara, 
    uint32_t totalQueries, uint32_t availableCores)
{
    tilingPara.availableCores = std::min(availableCores, totalQueries);
    tilingPara.availableCores = std::max(tilingPara.availableCores, 1u);

    tilingPara.tasksPerCore = CeilDivide(totalQueries, tilingPara.availableCores);
    tilingPara.pointsPerCore = (totalQueries % tilingPara.availableCores == 0)?
        tilingPara.availableCores : (totalQueries % tilingPara.availableCores);
    tilingPara.tasksOnLastCore = CeilDivide(totalQueries, tilingPara.tasksPerCore);
}

static void CalcWorkspaceForCylinderQuery(TilingContext* tilingContext,
    uint32_t batch, uint32_t numQueries, uint32_t numPoints)
{
    auto platInfo = tilingContext->GetPlatformInfo();
    if (platInfo == nullptr) {
        return;
    }
    auto platform = PlatformAscendC(platInfo);

    size_t* workspace = tilingContext->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return;
    }

    size_t systemSize = platform.GetLibApiWorkSpaceSize();
    size_t userSize = batch * numQueries * numPoints * FLOAT_SIZE;
    workspace[0] = userSize + systemSize;
}

graphStatus TilingFunctionForCylinderQuery(TilingContext* tilingContext) {
    if (tilingContext == nullptr) {
        return GRAPH_FAILED;
    }

    // 1. 验证平台信息
    auto platInfo = tilingContext->GetPlatformInfo();
    if (platInfo == nullptr) {
        return GRAPH_FAILED;
    }
    auto platform = PlatformAscendC(platInfo);

    // 2. 验证输入形状
    const StorageShape *seedShape = nullptr;
    const StorageShape *pointsShape = nullptr;
    if (!ValidateShapes(tilingContext, seedShape, pointsShape)) {
        return GRAPH_FAILED;
    }

    // 3. 解析属性
    float radius, hMin, hMax;
    uint32_t nSample;
    if (!ParseAttributes(tilingContext->GetAttrs(), radius, hMin, hMax, nSample)) {
        return GRAPH_FAILED;
    }

    // 4. 获取维度信息
    uint32_t batch = seedShape->GetStorageShape().GetDim(BATCH_IDX);
    uint32_t numPoints = pointsShape->GetStorageShape().GetDim(POINTS_IDX);
    uint32_t numQueries = seedShape->GetStorageShape().GetDim(QUERY_IDX);
    uint32_t totalQueries = batch * numQueries;

    // 5. 计算任务配置
    uint64_t ubSize;
    platform.GetCoreMemSize(CoreMemType::UB, ubSize);

    CylinderQueryTilingPara tilingPara;
    CalculateTaskConfig(tilingPara, numPoints, ubSize, nSample);

    // 6. 计算核心分配
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        return GRAPH_FAILED;
    }

    CalculateCoreDistribution(tilingPara, totalQueries, availableCores);

    // 7. 设置tiling数据
    tilingContext->SetBlockDim(tilingPara.availableCores);

    CylinderQueryTilingData tilingData;
    tilingData.set_batch(batch);
    tilingData.set_numPoints(numPoints);
    tilingData.set_numQueries(numQueries);
    tilingData.set_radius(radius);
    tilingData.set_hMin(hMin);
    tilingData.set_hMax(hMax);
    tilingData.set_nSample(nSample);
    tilingData.set_pointsPerTask(tilingPara.pointsPerTask);
    tilingData.set_blocksPerTask(tilingPara.blocksPerTask);
    tilingData.set_totalIterations(tilingPara.totalIterations);
    tilingData.set_lastIterationPoints(tilingPara.lastIterationPoints);
    tilingData.set_lastIterationBlocks(tilingPara.lastIterationBlocks);
    tilingData.set_tasksPerCore(tilingPara.tasksPerCore);
    tilingData.set_tasksOnLastCore(tilingPara.tasksOnLastCore);
    tilingData.set_pointsPerCore(tilingPara.pointsPerCore);

    // 8. 计算workspace
    CalcWorkspaceForCylinderQuery(tilingContext, batch, numQueries, numPoints);

    // 9. 保存tiling数据
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr) {
        return GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return GRAPH_SUCCESS;
}
}