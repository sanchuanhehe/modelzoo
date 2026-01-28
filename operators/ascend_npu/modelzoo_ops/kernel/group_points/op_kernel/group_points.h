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
#ifndef OP_KERNEL_GROUP_POINTS_H
#define OP_KERNEL_GROUP_POINTS_H

#include "kernel_operator.h"
#include "group_points_base.h"

using namespace AscendC;

class KernelGroupPoints {
public:
    // 初始化核函数
    __aicore__ inline KernelGroupPoints(GM_ADDR features, GM_ADDR index, GM_ADDR outPoints, TilingPara *tilingPara)
    {
        this->tilingPara = tilingPara;
        // 初始化UB缓冲区
        this->pipeline.InitBuffer(this->featuresBuf, this->tilingPara->ubMaxTasks *
            this->tilingPara->chnSizeAligned * FlOAT_SIZE);
        this->pipeline.InitBuffer(this->idxBuf, this->tilingPara->ubMaxTasks * INT32_SIZE);
        // 设置全局内存张量：将CPU侧数据搬运到核函数侧
        InitGlobalMemory(features, index, outPoints);
        // 获取硬件事件ID：用于流水线同步
        mte2Mte3EventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        mte3Mte2EventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    // 主流程
    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx > this->tilingPara->numCoreWork) {
            return;
        }
        // 计算当前core处理的任务起始偏移量
        uint64_t offset = blockIdx * this->tilingPara->tasksPerCore;
        // 默认使用main core的迭代次数和尾部任务数
        uint32_t iters = this->tilingPara->itersPerMainCore;
        uint32_t numTasks = this->tilingPara->tailTaskNumOnMainCore;
        // 对齐后的尾部任务数（32字节对齐）
        uint32_t tailTaskNumAligned = this->tilingPara->tailTaskNumOnMainCore;
        // 尾核特殊处理
        if (blockIdx == this->tilingPara->numCoreWork - 1) {
            iters = this->tilingPara->itersOnLastCore;
            numTasks = this->tilingPara->tailTaskNumOnLastCore;
            tailTaskNumAligned = this->tilingPara->tailTaskNumAligned;
        }

        for (int32_t i = 0; i < iters; i++) {
            CopyInAndCopyOut(offset, this->tilingPara->ubMaxTasks, this->tilingPara->ubMaxTasks);
            offset += this->tilingPara->ubMaxTasks;
        }
        if (numTasks != 0) {
            CopyInAndCopyOut(offset, numTasks, tailTaskNumAligned);
        }
    }

private:
    // 初始化全局内存
    __aicore__ inline void InitGlobalMemory(GM_ADDR features, GM_ADDR index, GM_ADDR outPoints)
    {
        // 计算features总长度
        uint64_t featuresTotalLen = static_cast<uint64_t>(this->tilingPara->batch) *
            this->tilingPara->totalPoints * this->tilingPara->channel;
        // 计算index总长度
        uint64_t idxTotalLen = static_cast<uint64_t>(this->tilingPara->batch) *
            this->tilingPara->numPoints * this->tilingPara->numSample;
        // 计算outPoints总长度
        uint64_t outPointsTotalLen = idxTotalLen * this->tilingPara->channel;

        this->featuresGlobal.SetGlobalBuffer((__gm__ float*)features, featuresTotalLen);
        this->idxGlobal.SetGlobalBuffer((__gm__ int32_t*)index, idxTotalLen);
        this->outPointsGlobal.SetGlobalBuffer((__gm__ float*)outPoints, outPointsTotalLen);
    }

    // 数据搬运和处理
    __aicore__ inline void CopyInAndCopyOut(uint64_t offset, uint32_t numTasks, uint32_t numTasksAligned)
    {
        // 初始化Local Buffer
        LocalTensor<float> featuresLocal = featuresBuf.Get<float>();
        LocalTensor<int32_t> idxLocal = idxBuf.Get<int32_t>();
        // 从GM复制idx数据到UB
        DataCopy(idxLocal, idxGlobal[offset], numTasksAligned);
        // 等待索引数据复制完成
        SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2EventId);
        WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2EventId);
        // 从GM复制特征数据到UB
        for (int32_t i = 0; i < numTasks; i++) {
            uint32_t idx = idxLocal.GetValue(i);
            uint32_t batchIdx = (offset + i) / this->tilingPara->numPoints / this->tilingPara->numSample;
            uint64_t featureIdx = batchIdx * this->tilingPara->totalPoints * this->tilingPara->channel +
                idx * this->tilingPara->channel;
            DataCopy(featuresLocal[i * this->tilingPara->chnSizeAligned], featuresGlobal[featureIdx],
                this->tilingPara->chnSizeAligned);
        }
        // 等待特征数据复制完成
        SetFlag<HardEvent::MTE2_MTE3>(mte2Mte3EventId);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2Mte3EventId);

        // 数据处理和拷出
        for (int32_t i = 1; i < numTasks; i++) {
            DataCopyPadCustom_UB2UB(featuresLocal[i * this->tilingPara->channel],
                featuresLocal[i * this->tilingPara->chnSizeAligned], this->tilingPara->channel);
        }
        DataCopyPadCustom_UB2GM(outPointsGlobal[offset * this->tilingPara->channel],
            featuresLocal, this->tilingPara->channel * numTasks);
    }

private:
    TPipe pipeline;
    TilingPara *tilingPara;
    TBuf<TPosition::VECCALC> idxBuf, featuresBuf;

    GlobalTensor<int32_t> idxGlobal;
    GlobalTensor<float> featuresGlobal, outPointsGlobal;

    uint8_t mte2Mte3EventId, mte3Mte2EventId;
};

#endif // OP_KERNEL_GROUP_POINTS_H