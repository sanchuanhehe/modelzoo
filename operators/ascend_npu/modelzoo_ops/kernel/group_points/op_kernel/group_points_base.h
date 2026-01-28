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
#ifndef OP_KERNEL_GROUP_POINTS_BASE_H
#define OP_KERNEL_GROUP_POINTS_BASE_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t MIN_BLOCK_SIZE = 32;
constexpr uint32_t FlOAT_SIZE = 4;
constexpr uint32_t INT32_SIZE = 4;

class TilingPara {
public:
    __aicore__ inline TilingPara() = default;
public:
    uint32_t batch;
    uint32_t channel;
    uint32_t totalPoints;
    uint32_t numPoints;
    uint32_t numSample;
    uint32_t chnSizeAligned;
    uint32_t ubMaxTasks;
    uint32_t numCoreWork;
    uint32_t tasksPerCore;
    uint32_t tasksOnLastCore;
    uint32_t itersPerMainCore;
    uint32_t tailTaskNumOnMainCore;
    uint32_t itersOnLastCore;
    uint32_t tailTaskNumOnLastCore;
    uint32_t tailTaskNumAligned;
};

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

template <typename typeT>
__aicore__ inline void DataCopyPadCustom_UB2UB(
    const LocalTensor<typeT>& dstLocal, const LocalTensor<typeT>& srcLocal, const uint32_t calCount)
{
    constexpr uint32_t ELEMENTS_PER_BLOCK = MIN_BLOCK_SIZE / sizeof(typeT);
    uint32_t alignedCount = (calCount / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;

    // 批量拷贝对齐部分
    if (alignedCount > 0) {
        DataCopy(dstLocal, srcLocal, alignedCount);
    }

    // 逐元素处理剩余部分
    for (int32_t i = alignedCount; i < calCount; i++) {
        dstLocal.SetValue(i, srcLocal.GetValue(i));
    }
}

#endif // OP_KERNEL_GROUP_POINTS_BASE_H