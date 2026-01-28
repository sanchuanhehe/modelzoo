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
#include "group_points_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace gert;
using namespace platform_ascendc;
namespace optiling {
static constexpr uint32_t MIN_BLOCK_SIZE = 32;
static constexpr uint32_t FLOAT_SIZE = 4;
static constexpr uint32_t INT32_SIZE = 4;
static constexpr uint32_t MIN_TASK_NUM_PER_CORE = 64;
static constexpr uint32_t INT32_ELEMENTS_NUM_PER_BLOCK = MIN_BLOCK_SIZE / INT32_SIZE;
static constexpr uint32_t FLOAT_ELEMENTS_NUM_PER_BLOCK = MIN_BLOCK_SIZE / FLOAT_SIZE;
static constexpr uint64_t RPC_RESERVED_SIZE = 20 * 1024;
static constexpr uint64_t DMA_MAX_BLOCKS = 4095;

static constexpr uint32_t BATCH_IDX = 0;
static constexpr uint32_t TOTAL_POINTS_IDX = 1;
static constexpr uint32_t CHANNEL_IDX = 2;
static constexpr uint32_t NUM_POINT_IDX = 1;
static constexpr uint32_t NUM_SAMPLE_IDX = 2;

static bool ValidateShapes(TilingContext* tilingContext, const StorageShape*& featuresShape, 
    const StorageShape*& indexShape)
{
    featuresShape = tilingContext->GetInputShape(0);
    indexShape = tilingContext->GetInputShape(1);
    return featuresShape != nullptr && indexShape != nullptr;
}

static void CalcWorkspaceForGroupPoints(TilingContext* tilingContext,
    uint32_t batch, uint32_t numPoints, uint32_t numSample, uint32_t channel)
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
    size_t userSize = batch * numPoints * numSample * channel * FLOAT_SIZE;
    workspace[0] = userSize + systemSize;
}

graphStatus TilingFunctionForGroupPoints(gert::TilingContext* tilingContext)
{
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
    const StorageShape *featuresShape = nullptr;
    const StorageShape *indexShape = nullptr;
    if (!ValidateShapes(tilingContext, featuresShape, indexShape)) {
        return GRAPH_FAILED;
    }

    // 3. 获取维度信息
    auto batch = featuresShape->GetStorageShape().GetDim(BATCH_IDX);
    auto totalPoints = featuresShape->GetStorageShape().GetDim(TOTAL_POINTS_IDX);
    auto channel = featuresShape->GetStorageShape().GetDim(CHANNEL_IDX);
    auto numPoints = indexShape->GetStorageShape().GetDim(NUM_POINT_IDX);
    auto numSample = indexShape->GetStorageShape().GetDim(NUM_SAMPLE_IDX);

    uint32_t chnSizeAligned = AlignUp(static_cast<uint32_t>(channel), FLOAT_ELEMENTS_NUM_PER_BLOCK);

    // 4. 计算任务配置和core分配
    uint64_t ubSize;
    platform.GetCoreMemSize(CoreMemType::UB, ubSize);
    uint64_t availableUbSize = ubSize - RPC_RESERVED_SIZE;
    static uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        return GRAPH_FAILED;
    }
    uint32_t totalTaskNum = batch * numPoints * numSample;
    uint32_t tasksPerCore = CeilDivide(totalTaskNum, availableCores);
    tasksPerCore = AlignUp(tasksPerCore, MIN_TASK_NUM_PER_CORE);
    if (tasksPerCore == 0) {
        return GRAPH_FAILED;
    }
    uint32_t numCoreWork = CeilDivide(totalTaskNum, tasksPerCore);
    uint32_t tasksOnLastCore = (totalTaskNum % tasksPerCore == 0) ? tasksPerCore : (totalTaskNum % tasksPerCore);

    uint64_t singleTaskSize = chnSizeAligned * FLOAT_SIZE + INT32_SIZE;
    uint32_t ubMaxTasks = AlignFloor(std::min(DMA_MAX_BLOCKS, FloorDivide(availableUbSize, singleTaskSize)),
        static_cast<uint64_t>(INT32_ELEMENTS_NUM_PER_BLOCK));
    if (ubMaxTasks == 0) {
        return GRAPH_FAILED;
    }
    uint32_t tailTaskNumAligned = AlignUp(tasksOnLastCore % ubMaxTasks, INT32_ELEMENTS_NUM_PER_BLOCK);

    // 7. 设置tiling数据
    GroupPointsTilingData tilingData;
    tilingContext->SetBlockDim(numCoreWork);
    tilingData.set_batch(batch);
    tilingData.set_channel(channel);
    tilingData.set_totalPoints(totalPoints);
    tilingData.set_numPoints(numPoints);
    tilingData.set_numSample(numSample);
    tilingData.set_chnSizeAligned(chnSizeAligned);
    tilingData.set_ubMaxTasks(ubMaxTasks);
    tilingData.set_numCoreWork(numCoreWork);
    tilingData.set_tasksPerCore(tasksPerCore);
    tilingData.set_tasksOnLastCore(tasksOnLastCore);
    tilingData.set_itersPerMainCore(tasksPerCore / ubMaxTasks);
    tilingData.set_tailTaskNumOnMainCore(tasksPerCore % ubMaxTasks);
    tilingData.set_itersOnLastCore(tasksOnLastCore / ubMaxTasks);
    tilingData.set_tailTaskNumOnLastCore(tasksOnLastCore % ubMaxTasks);
    tilingData.set_tailTaskNumAligned(tailTaskNumAligned);

    // 8. 计算workspace
    CalcWorkspaceForGroupPoints(tilingContext, batch, numPoints, numSample, channel);

    // 9. 保存tiling数据
    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr) {
        return GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    return GRAPH_SUCCESS;
}
} // namespace optiling