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
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "furthest_point_sampling_tiling.h"

using namespace ge;
using namespace std;
using namespace gert;
using namespace platform_ascendc;
using namespace AscendC;

namespace optiling {
static constexpr uint32_t FLOAT_SIZE = 4;
static constexpr uint32_t UB_RESERVE = 3072;
static constexpr uint32_t UB_DATA_BLOCKS = 8;
static constexpr uint32_t DATA_SIZE_PER_REPEAT = 256;
static constexpr uint32_t DATA_SIZE_PER_BLOCK = 32;

struct FurthestPointSamplingTilingPara {
    uint32_t coreNum;
    uint32_t batchSize;
    uint32_t totalPoints;
    uint32_t nSample;
    uint32_t numTile;
    uint32_t tileSize;
    uint32_t lastTileSize;
    uint32_t ubWorkSpaceSize;
    uint32_t pairBufSize;
    uint32_t batchsPerLargeCore;
    uint32_t batchsPerLittleCore;
    uint32_t numLargeCore;
    uint32_t repeatsPerBlock;
};

static bool ValidateShapes(TilingContext* tilingContext, const StorageShape*& xyzTransShape)
{
    xyzTransShape = tilingContext->GetInputShape(0);
    return xyzTransShape != nullptr;
}

static bool ParseAttributes(const RuntimeAttrs* attr, FurthestPointSamplingTilingPara& tilingPara)
{
    if (attr == nullptr) return false;

    tilingPara.nSample = *(attr->GetAttrPointer<uint32_t>(0));
    return true;
}

static uint64_t GetWorkSpaceSize(uint32_t repeatsPerBlock)
{
    uint32_t numDataPerBlock = DATA_SIZE_PER_BLOCK / FLOAT_SIZE;
    uint32_t numDataPerRepeat = DATA_SIZE_PER_REPEAT / FLOAT_SIZE;
    
    uint32_t StepOneOutputDataNum = repeatsPerBlock * 2;
    uint32_t StepTwoInputAlign = AlignUp<uint32_t>(StepOneOutputDataNum, numDataPerBlock);
    uint32_t StepTwoOutputDataNum = CeilDivide<uint32_t>(StepOneOutputDataNum, numDataPerRepeat) * 2;
    uint32_t StepThreeInputAlign = AlignUp<uint32_t>(StepTwoOutputDataNum, numDataPerBlock);
    uint32_t StepThreeOutputDataNum = CeilDivide<uint32_t>(StepTwoOutputDataNum, numDataPerRepeat) * 2;
    uint32_t StepThreeAlign = AlignUp<uint32_t>(StepThreeOutputDataNum, numDataPerBlock);

    uint32_t totalWorkSpaceDataNum = StepTwoInputAlign + StepThreeInputAlign + StepThreeAlign;
    uint64_t totalWorkSpaceSize = (uint64_t)totalWorkSpaceDataNum * FLOAT_SIZE;

    if (totalWorkSpaceSize % DATA_SIZE_PER_BLOCK != 0) {
        return GRAPH_FAILED;
    }
    return totalWorkSpaceSize;
}

static uint64_t GetUbWorkSpace(FurthestPointSamplingTilingPara& tilingPara, uint64_t numData)
{
    uint64_t dataSpace = AlignUp<uint64_t>(numData * FLOAT_SIZE, DATA_SIZE_PER_REPEAT);
    uint64_t totalRepeats = CeilDivide<uint64_t>(tilingPara.totalPoints * FLOAT_SIZE, DATA_SIZE_PER_REPEAT);
    uint64_t repeatsPerBlock = CeilDivide<uint64_t>(numData * FLOAT_SIZE, DATA_SIZE_PER_REPEAT);

    tilingPara.numTile = (uint32_t)CeilDivide<uint64_t>(totalRepeats, repeatsPerBlock);
    tilingPara.repeatsPerBlock = (repeatsPerBlock < totalRepeats) ? (uint32_t)repeatsPerBlock : (uint32_t)totalRepeats;
    tilingPara.ubWorkSpaceSize = GetWorkSpaceSize(tilingPara.repeatsPerBlock);
    tilingPara.pairBufSize = AlignUp<uint32_t>(tilingPara.numTile * 2 * FLOAT_SIZE, DATA_SIZE_PER_BLOCK);

    return (dataSpace * UB_DATA_BLOCKS + tilingPara.ubWorkSpaceSize + tilingPara.pairBufSize);
}

static uint64_t GetMaxDataBlockOnUb(FurthestPointSamplingTilingPara& tilingPara, uint64_t& ubSize)
{
    uint64_t result = 0;
    uint64_t maxDataSize = ubSize - UB_RESERVE;
    uint64_t l = 1;
    uint64_t r = maxDataSize / FLOAT_SIZE;

    while (l <= r) {
        uint64_t mid = l + (r - l) / 2;
        if (maxDataSize >= GetUbWorkSpace(tilingPara, mid)) {
            result = mid;
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
    return result;
}

static void CalculateCoreDistribution(FurthestPointSamplingTilingPara& tilingPara, uint32_t availableCores)
{
    tilingPara.coreNum = availableCores;
    tilingPara.batchsPerLittleCore = tilingPara.batchSize / tilingPara.coreNum;
    tilingPara.batchsPerLargeCore = tilingPara.batchsPerLittleCore + 1;  // 大核多处理1个

    // 大核数量 = (总任务数 - 小核任务数*总核心数) / 1
    tilingPara.numLargeCore = tilingPara.batchSize - (tilingPara.batchsPerLittleCore * tilingPara.coreNum);

    // 当能整除时
    if (tilingPara.numLargeCore == 0) {
        tilingPara.batchsPerLargeCore = tilingPara.batchsPerLittleCore;  // 所有核心任务数相同
        tilingPara.numLargeCore = tilingPara.coreNum;           // 所有核心都算大核
    }
}

static void CalcWorkspaceForFurthestPointSampling(TilingContext* tilingContext,
    FurthestPointSamplingTilingPara& tilingPara)
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
    size_t userSize = tilingPara.batchSize * tilingPara.totalPoints * FLOAT_SIZE;
    workspace[0] = userSize + systemSize;
}

graphStatus TilingFunctionForFurthestPointSampling(gert::TilingContext* tilingContext)
{
    if (tilingContext == nullptr) {
        return GRAPH_FAILED;
    }
    FurthestPointSamplingTilingPara tilingPara;

    // 1. 验证平台信息
    auto platInfo = tilingContext->GetPlatformInfo();
    if (platInfo == nullptr) {
        return GRAPH_FAILED;
    }
    auto platform = PlatformAscendC(platInfo);

    // 2. 验证输入形状
    const gert::StorageShape *xyzTransShape = nullptr;
    if (!ValidateShapes(tilingContext, xyzTransShape)) {
        return GRAPH_FAILED;
    }

    // 3. 解析属性
    if (!ParseAttributes(tilingContext->GetAttrs(), tilingPara)) {
        return GRAPH_FAILED;
    }

    // 4. 获取维度信息
    tilingPara.batchSize = xyzTransShape->GetStorageShape().GetDim(0);
    tilingPara.totalPoints = xyzTransShape->GetStorageShape().GetDim(2);

    // 5. 计算空间和任务配置
    uint64_t ubSize;
    platform.GetCoreMemSize(CoreMemType::UB, ubSize);

    uint64_t maxDataSize = GetMaxDataBlockOnUb(tilingPara, ubSize);

    if (tilingPara.repeatsPerBlock == 0) {
        return GRAPH_FAILED;
    }
    GetUbWorkSpace(tilingPara, maxDataSize);
    tilingPara.tileSize = ((tilingPara.repeatsPerBlock * DATA_SIZE_PER_REPEAT) / FLOAT_SIZE);
    tilingPara.lastTileSize = tilingPara.totalPoints - tilingPara.tileSize * (tilingPara.numTile - 1);

    // 6. 计算核心分配
    uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        return GRAPH_FAILED;
    }
    CalculateCoreDistribution(tilingPara, availableCores);

    // 7. 设置tiling数据
    if (tilingPara.batchSize <= tilingPara.coreNum) {
        tilingContext->SetBlockDim(tilingPara.batchSize);
    } else {
        tilingContext->SetBlockDim(tilingPara.coreNum);
    }

    FurthestPointSamplingTilingData tilingData;
    tilingData.set_batchSize(tilingPara.batchSize);
    tilingData.set_totalPoints(tilingPara.totalPoints);
    tilingData.set_nSample(tilingPara.nSample);
    tilingData.set_numTile(tilingPara.numTile);
    tilingData.set_tileSize(tilingPara.tileSize);
    tilingData.set_lastTileSize(tilingPara.lastTileSize);
    tilingData.set_ubWorkSpaceSize(tilingPara.ubWorkSpaceSize);
    tilingData.set_pairBufSize(tilingPara.pairBufSize);
    tilingData.set_batchsPerLargeCore(tilingPara.batchsPerLargeCore);
    tilingData.set_batchsPerLittleCore(tilingPara.batchsPerLittleCore);
    tilingData.set_numLargeCore(tilingPara.numLargeCore);
    tilingData.set_repeatsPerBlock(tilingPara.repeatsPerBlock);

    // 8. 计算workspace
    CalcWorkspaceForFurthestPointSampling(tilingContext, tilingPara);

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