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
#ifndef OP_KERNEL_CYLINDER_QUERY_BASE_H
#define OP_KERNEL_CYLINDER_QUERY_BASE_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t INT32_SIZE = 4;
constexpr uint32_t FLOAT_SIZE = 4;

// 计算参数
constexpr uint32_t POINTS_PER_BLOCK = 8;
constexpr uint32_t BYTES_PER_BLOCK = 96;
constexpr uint32_t POINT_DIMENSION = 3;
constexpr uint32_t ROTATION_MATRIX_SIZE = 9;
constexpr uint32_t DOUBLE_BUFFER_COUNT = 2;
constexpr uint32_t MIN_BLOCK_SIZE = 32;

// 步长参数
constexpr int32_t GATHER_SRC0_STRIDE = 3;
constexpr int32_t GATHER_SRC1_STRIDE = 0;

// 掩码模式常量
constexpr uint32_t MASK_PATTERN_0 = 0x49249249;
constexpr uint32_t MASK_PATTERN_1 = 0x92492492;
constexpr uint32_t MASK_PATTERN_2 = 0x24924924;

// 数据偏移常量和索引
constexpr uint32_t OFFSET_MASK_0 = 0;
constexpr uint32_t OFFSET_MASK_1 = 1;
constexpr uint32_t OFFSET_MASK_2 = 2;

constexpr uint32_t X_IDX = 0;
constexpr uint32_t Y_IDX = 1;
constexpr uint32_t Z_IDX = 2;

constexpr uint32_t ROT_MAT_X_VALUE0_IDX = 0;
constexpr uint32_t ROT_MAT_Y_VALUE0_IDX = 1;
constexpr uint32_t ROT_MAT_Z_VALUE0_IDX = 2;
constexpr uint32_t ROT_MAT_X_VALUE1_IDX = 3;
constexpr uint32_t ROT_MAT_Y_VALUE1_IDX = 4;
constexpr uint32_t ROT_MAT_Z_VALUE1_IDX = 5;
constexpr uint32_t ROT_MAT_X_VALUE2_IDX = 6;
constexpr uint32_t ROT_MAT_Y_VALUE2_IDX = 7;
constexpr uint32_t ROT_MAT_Z_VALUE2_IDX = 8;

#define CEIL_ALIGN(x, align) (((x) + (align) - 1) / (align) * (align))
#define CEIL_ALIGN32(x) CEIL_ALIGN(x, 32)
#define CEIL_ALIGN64(x) CEIL_ALIGN(x, 64)
#define CEIL_DIV8(x) (((x) + 7) / 8)

template<typename typeT>
inline typeT CeilDivide(typeT dividend, typeT divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (dividend + divisor - 1) / divisor;
}

template<typename typeT>
inline typeT AlignUp(typeT value, typeT alignment)
{
    if (alignment == 0) {
        return 0;
    }
    return CeilDivide(value, alignment) * alignment;
}

class TilingPara {
public:
    __aicore__ inline TilingPara() = default;
public:
    uint32_t batch;
    uint32_t numPoints;
    uint32_t numQueries;
    float radius;
    float hMin;
    float hMax;
    uint32_t nSample;
    uint32_t pointsPerTask;
    uint32_t blocksPerTask;
    uint32_t totalIterations;
    uint32_t lastIterationPoints;
    uint32_t lastIterationBlocks;
    uint32_t tasksPerCore;
    uint32_t tasksOnLastCore;
    uint32_t pointsPerCore;
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
#endif // OP_KERNEL_CYLINDER_QUERY_BASE_H