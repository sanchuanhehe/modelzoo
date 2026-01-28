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

#ifndef OP_KERNEL_FURTHEST_POINT_SAMPLING_BASE_H
#define OP_KERNEL_FURTHEST_POINT_SAMPLING_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t FLOAT_SIZE = 4;
constexpr uint32_t INT32_SIZE = 4;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t Y_INDEX = 1;
constexpr uint32_t Z_INDEX = 2;
constexpr uint32_t MIN_BLOCK_SIZE = 32;
constexpr uint32_t POINT_DIMENSION = 3;
constexpr uint32_t SINGLE_BUFFER = 1;

constexpr uint32_t DATA_NUM_ALIGN_WITH_32 = 8; // 32 / FLOAT_SIZE
constexpr uint32_t DATA_NUM_ALIGN_WITH_64 = 16; // 64 / FLOAT_SIZE
constexpr uint32_t DATA_NUM_ALIGN_WITH_256 = 64; // 256 / FLOAT_SIZE
constexpr uint32_t DATA_NUM_ALIGN_WITH_1024 = 256; // 1024 / FLOAT_SIZE

constexpr uint32_t MAX_TIMES = 255;
constexpr uint32_t ALIGN_DATA_SIZE = 256u;

class TilingPara {
public:
    __aicore__ inline TilingPara() = default;
public:
    uint32_t totalPoints;
    uint32_t batchSize;
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

template <typename typeT>
__aicore__ inline void DataCopyPadCustom_GM2UB(
    const LocalTensor<typeT>& dstLocal, const GlobalTensor<typeT>& srcGlobal, const uint32_t calCount)
{
    constexpr uint32_t ELEMENTS_PER_BLOCK = MIN_BLOCK_SIZE / sizeof(typeT);
    uint32_t alignedCount = (calCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;

    // 批量拷贝对齐部分
    if (alignedCount > 0) {
        DataCopy(dstLocal, srcGlobal, alignedCount);
    }

    // 逐元素处理剩余部分
    for (int32_t i = alignedCount; i < calCount; i++) {
        dstLocal.SetValue(i, srcGlobal.GetValue(i));
    }
}

template <typename typeT>
__aicore__ inline void DataCopyPadCustom_UB2GM(
    const GlobalTensor<typeT>& dstGlobal, const LocalTensor<typeT>& srcLocal, const uint32_t calCount)
{
    constexpr uint32_t ELEMENTS_PER_BLOCK = MIN_BLOCK_SIZE / sizeof(typeT);

    uint32_t alignedCount = (calCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;

    // 批量拷贝对齐部分
    if (alignedCount > 0) {
        DataCopy(dstGlobal, srcLocal, alignedCount);
    }

    // 对齐部分处理
    if (calCount - alignedCount > 0) {
        if (calCount > ELEMENTS_PER_BLOCK) {
            // 拷贝最后ELEMENTS_PER_BLOCK个数据
            for (int32_t i = 0; i < ELEMENTS_PER_BLOCK; i++) {
                auto tensorValue = srcLocal.GetValue(calCount - ELEMENTS_PER_BLOCK + i);
                srcLocal.SetValue(i, tensorValue);
            }
            DataCopy(dstGlobal[calCount - ELEMENTS_PER_BLOCK], srcLocal, ELEMENTS_PER_BLOCK);
        } else {
            AscendC::printf("calCount:%u alignedCount:%u ELEMENTS_PER_BLOCK:%u\n", calCount, alignedCount, ELEMENTS_PER_BLOCK);
            DataCopy(dstGlobal[alignedCount], srcLocal[alignedCount], ELEMENTS_PER_BLOCK);
        }
    }
}

#endif // OP_KERNEL_FURTHEST_POINT_SAMPLING_BASE_H