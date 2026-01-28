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
#ifndef OP_KERNEL_CYLINDER_QUERY_H
#define OP_KERNEL_CYLINDER_QUERY_H

#include "kernel_operator.h"
#include "cylinder_query_base.h"

using namespace AscendC;

/**
 * 该算子用于在点云数据中查询满足圆柱体约束条件的点：
 * 1. 计算点相对于查询点的相对位置
 * 2. 应用旋转矩阵进行坐标变换
 * 3. 检查变换后的点是否在圆柱体内（半径和高度的约束）
 */
class CylinderQueryKernel {
public:
    __aicore__ inline CylinderQueryKernel(GM_ADDR seedXyz, GM_ADDR pointXyz, GM_ADDR rotMat, GM_ADDR defaultIdx,
        GM_ADDR output, TilingPara *tilingPara)
    {
        this->tilingPara = tilingPara;
        // 获取当前核函数块索引，用于任务分配
        uint32_t blockIndex = GetBlockIdx();
        // 计算任务起始偏移：根据核函数块索引分配任务
        if (blockIndex < this->tilingPara->pointsPerCore) {
            // 前pointsPerCore个核函数块每个处理tasksPerCore个任务
            this->taskStartOffset = blockIndex * this->tilingPara->tasksPerCore;
        } else {
            // 剩余的核函数块每个处理tasksPerCore-1个任务
            this->taskStartOffset = this->tilingPara->pointsPerCore * this->tilingPara->tasksPerCore +
                (blockIndex - this->tilingPara->pointsPerCore) * (this->tilingPara->tasksPerCore - 1);
            this->tilingPara->tasksPerCore = this->tilingPara->tasksPerCore - 1;
        }
        // 初始化Local Buffer
        InitLocalBuffer();

        // 设置全局内存张量：将CPU侧数据搬运到核函数侧
        this->seedXyzGm.SetGlobalBuffer((__gm__ float*) seedXyz);
        this->pointXyzGm.SetGlobalBuffer((__gm__ float*) pointXyz);
        this->rotMatGm.SetGlobalBuffer((__gm__ float*) rotMat);
        this->outPointsGlobal.SetGlobalBuffer((__gm__ float*) output);
        this->defaultIdxGm.SetGlobalBuffer((__gm__ float*) defaultIdx);
    }

    __aicore__ inline ~CylinderQueryKernel() = default;

    __aicore__ inline void Process()
    {
        // 获取硬件事件ID：用于流水线同步
        int32_t svEventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        int32_t mte3Mte2EventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        int32_t mte2EventId =  static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        int32_t mte3EventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        // 遍历当前核函数块的所有任务
        for (int task_idx = 0; task_idx < this->tilingPara->tasksPerCore; task_idx++) {
            // 计算全局任务偏移
            uint32_t taskOffset = this->taskStartOffset + task_idx;
            // 计算当前批次索引：用于确定点云数据的批次偏移
            this->curBatchIdx = taskOffset / this->tilingPara->numQueries;
            // 加载SeedXyz坐标和旋转矩阵
            LoadSeedPointAndRotMat(taskOffset);
            // 同步事件：等待数据加载完成
            SetFlag<HardEvent::S_V>(svEventId);
            WaitFlag<HardEvent::S_V>(svEventId);
            // 设置本轮计算处理的点个数
            this->curPointsCount = this->tilingPara->pointsPerTask;

            for (int iter = 0; iter < this->tilingPara->totalIterations; iter++) {
                // 同步事件：等待内存传输就绪
                SetFlag<HardEvent::MTE3_MTE2>(mte3Mte2EventId);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3Mte2EventId);
                // 计算结果偏移
                uint32_t dataOffset = this->tilingPara->pointsPerTask * iter;
                int outOffset = iter * this->curPointsCount;
                this->curBlockCount = this->tilingPara->blocksPerTask;
                if (iter == this->tilingPara->totalIterations - 1) {
                    this->curBlockCount = this->tilingPara->lastIterationBlocks;
                    this->curPointsCount = this->tilingPara->lastIterationPoints;
                }
                // 初始化输出
                InitOut(outOffset);
                // 加载点云数据
                LoadPointCloud(dataOffset);
                // 同步事件：等待点云数据加载完成
                SetFlag<HardEvent::MTE2_V>(mte2EventId);
                WaitFlag<HardEvent::MTE2_V>(mte2EventId);
                // 执行圆柱查询计算
                CylinderQueryCompute();
                // 等待计算完成
                SetFlag<HardEvent::V_MTE3>(mte3EventId);
                WaitFlag<HardEvent::V_MTE3>(mte3EventId);
                // 将结果写回全局内存
                CopyOut(taskOffset, dataOffset);
            }
        }
    }

private:
    __aicore__ inline void InitLocalBuffer()
    {
        // 初始化点云数据（双Buf）
        this->pipeline.InitBuffer(this->pointXyzBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK);
        // 初始化XYZ坐标分量分离
        this->pipeline.InitBuffer(this->pointXBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK / POINT_DIMENSION);
        this->pipeline.InitBuffer(this->pointYBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK / POINT_DIMENSION);
        this->pipeline.InitBuffer(this->pointZBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK / POINT_DIMENSION);
        // 初始化旋转矩阵
        this->pipeline.InitBuffer(this->rotMatBuf, DOUBLE_BUFFER_COUNT * ROTATION_MATRIX_SIZE * FLOAT_SIZE);
        // 初始化旋转后坐标
        this->pipeline.InitBuffer(this->rotMatXBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK / POINT_DIMENSION);
        this->pipeline.InitBuffer(this->rotMatYBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK / POINT_DIMENSION);
        this->pipeline.InitBuffer(this->rotMatZBuf, DOUBLE_BUFFER_COUNT *
            this->tilingPara->blocksPerTask * BYTES_PER_BLOCK / POINT_DIMENSION);
        // 初始化掩码：用于存储距离和高度条件的判断结果
        this->pipeline.InitBuffer(this->rotYMaskBuf, DOUBLE_BUFFER_COUNT * 
            CEIL_DIV8(CEIL_ALIGN64(this->tilingPara->pointsPerTask)));
        this->pipeline.InitBuffer(this->rotXMaskBuf, DOUBLE_BUFFER_COUNT * 
            CEIL_DIV8(CEIL_ALIGN64(this->tilingPara->pointsPerTask)));
        // 初始化掩码模式：用于从交错存储的点云数据中提取XYZ分量
        this->pipeline.InitBuffer(this->maskPatternXBuf, DOUBLE_BUFFER_COUNT * BYTES_PER_BLOCK);
        this->pipeline.InitBuffer(this->maskPatternYBuf, DOUBLE_BUFFER_COUNT * BYTES_PER_BLOCK);
        this->pipeline.InitBuffer(this->maskPatternZBuf, DOUBLE_BUFFER_COUNT * BYTES_PER_BLOCK);
        // 初始化结果：32字节对齐
        this->pipeline.InitBuffer(this->outBuf, DOUBLE_BUFFER_COUNT * CEIL_ALIGN32(this->tilingPara->pointsPerTask) * FLOAT_SIZE);
        // 点云原始数据
        this->pointXyzLocal = this->pointXyzBuf.Get<float>();
        // XYZ分量
        this->pointXLocal = this->pointXBuf.Get<float>();
        this->pointYLocal = this->pointYBuf.Get<float>();
        this->pointZLocal = this->pointZBuf.Get<float>();
        // 旋转矩阵
        this->rotMatLocal = this->rotMatBuf.Get<float>();
        // 旋转后XYZ分量
        this->rotMatXLocal = this->rotMatXBuf.Get<float>();
        this->rotMatYLocal = this->rotMatYBuf.Get<float>();
        this->rotMatZLocal = this->rotMatZBuf.Get<float>();
        // XY方向距离掩码
        this->rotYMaskLocal = this->rotYMaskBuf.Get<uint8_t>();
        this->rotXMaskLocal = this->rotXMaskBuf.Get<uint8_t>();
        // XYZ分量提取模式
        this->maskPatternXLocal = this->maskPatternXBuf.Get<uint32_t>();
        this->maskPatternYLocal = this->maskPatternYBuf.Get<uint32_t>();
        this->maskPatternZLocal = this->maskPatternZBuf.Get<uint32_t>();
        // 结果
        this->outLocal = this->outBuf.Get<float>();
        // 设置掩码模式：用于从交错存储的点云数据中提取XYZ分量
        // 点云数据存储格式：[x0,y0,z0, x1,y1,z1, x2,y2,z2]
        // 需要提取为：X分量：[x0,x1,x2], Y分量：[y0,y1,y2], Z分量：[z0,z1,z2]

        // X分量掩码模式：选择每三个元素中的第一个（x0, x1, x2）
        this->maskPatternXLocal.SetValue(OFFSET_MASK_0, MASK_PATTERN_0);
        this->maskPatternXLocal.SetValue(OFFSET_MASK_1, MASK_PATTERN_1);
        this->maskPatternXLocal.SetValue(OFFSET_MASK_2, MASK_PATTERN_2);
        // Y分量掩码模式：选择每三个元素中的第二个（y0, y1, y2）
        this->maskPatternYLocal.SetValue(OFFSET_MASK_0, MASK_PATTERN_1);
        this->maskPatternYLocal.SetValue(OFFSET_MASK_1, MASK_PATTERN_2);
        this->maskPatternYLocal.SetValue(OFFSET_MASK_2, MASK_PATTERN_0);
        // Z分量掩码模式：选择每三个元素中的第三个（z0, z1, z2）
        this->maskPatternZLocal.SetValue(OFFSET_MASK_0, MASK_PATTERN_2);
        this->maskPatternZLocal.SetValue(OFFSET_MASK_1, MASK_PATTERN_0);
        this->maskPatternZLocal.SetValue(OFFSET_MASK_2, MASK_PATTERN_1);
    }

    __aicore__ inline void InitOut(int outOffset)
    {
        DataCopyPadCustom_GM2UB(this->outLocal, this->defaultIdxGm[outOffset], CEIL_ALIGN32(this->curPointsCount));
    }

    __aicore__ inline void LoadSeedPointAndRotMat(uint32_t taskOffset)
    {
        int32_t mte2EventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        // 从全局内存读取种子点坐标
        this->seedX = this->seedXyzGm.GetValue(taskOffset * POINT_DIMENSION + X_IDX);
        this->seedY = this->seedXyzGm.GetValue(taskOffset * POINT_DIMENSION + Y_IDX);
        this->seedZ = this->seedXyzGm.GetValue(taskOffset * POINT_DIMENSION + Z_IDX);
        // 从全局内存加载旋转矩阵
        DataCopyPadCustom_GM2UB(rotMatLocal, this->rotMatGm[static_cast<uint64_t>(taskOffset * ROTATION_MATRIX_SIZE)],
            ROTATION_MATRIX_SIZE);
        // 同步事件：等待旋转矩阵加载完成
        SetFlag<HardEvent::MTE2_S>(mte2EventId);
        WaitFlag<HardEvent::MTE2_S>(mte2EventId);
        // 将旋转矩阵元素加载到寄存器中，提高访问速度
        rotMatX0 = rotMatLocal.GetValue(ROT_MAT_X_VALUE0_IDX);
        rotMatX1 = rotMatLocal.GetValue(ROT_MAT_X_VALUE1_IDX);
        rotMatX2 = rotMatLocal.GetValue(ROT_MAT_X_VALUE2_IDX);
        rotMatY0 = rotMatLocal.GetValue(ROT_MAT_Y_VALUE0_IDX);
        rotMatY1 = rotMatLocal.GetValue(ROT_MAT_Y_VALUE1_IDX);
        rotMatY2 = rotMatLocal.GetValue(ROT_MAT_Y_VALUE2_IDX);
        rotMatZ0 = rotMatLocal.GetValue(ROT_MAT_Z_VALUE0_IDX);
        rotMatZ1 = rotMatLocal.GetValue(ROT_MAT_Z_VALUE1_IDX);
        rotMatZ2 = rotMatLocal.GetValue(ROT_MAT_Z_VALUE2_IDX);
    }

    // offset：查询点偏移量
    __aicore__ inline void LoadPointCloud(uint32_t dataOffset)
    {
        DataCopyPadCustom_GM2UB(pointXyzLocal, this->pointXyzGm[POINT_DIMENSION * (static_cast<uint64_t>(dataOffset) +
            this->curBatchIdx * this->tilingPara->numPoints)], curPointsCount * POINT_DIMENSION);
    }

    __aicore__ inline void CopyOut(uint32_t taskOffset, uint32_t dataOffset)
    {
        DataCopyPadCustom_UB2GM(this->outPointsGlobal[static_cast<uint64_t>(dataOffset) + taskOffset * this->tilingPara->numPoints],
            this->outLocal, this->curPointsCount);
    }

    __aicore__ inline void SeparateXYZComponents()
    {
        bool mode = true;
        uint16_t repeatTimes = this->curBlockCount;
        uint8_t gatherStrideSrc0 = GATHER_SRC0_STRIDE;
        uint8_t gatherStrideSrc1 = GATHER_SRC1_STRIDE;
        uint64_t reserveCnt = 0;

        // 分离XYZ坐标分量
        for (uint32_t i = 0; i < repeatTimes; i++) {
            GatherMask(this->pointXLocal[i * POINTS_PER_BLOCK], this->pointXyzLocal[i * POINTS_PER_BLOCK * 3],
                this->maskPatternXLocal, mode, POINTS_PER_BLOCK * 3, {1, 1, gatherStrideSrc0, gatherStrideSrc1}, reserveCnt);
            GatherMask(this->pointYLocal[i * POINTS_PER_BLOCK], this->pointXyzLocal[i * POINTS_PER_BLOCK * 3],
                this->maskPatternYLocal, mode, POINTS_PER_BLOCK * 3, {1, 1, gatherStrideSrc0, gatherStrideSrc1}, reserveCnt);
            GatherMask(this->pointZLocal[i * POINTS_PER_BLOCK], this->pointXyzLocal[i * POINTS_PER_BLOCK * 3],
                this->maskPatternZLocal, mode, POINTS_PER_BLOCK * 3, {1, 1, gatherStrideSrc0, gatherStrideSrc1}, reserveCnt);
        }
    }

    __aicore__ inline void ComputeRelativePosition(uint32_t pointsCountAlign8)
    {
        Adds(this->pointXLocal, this->pointXLocal, -seedX, pointsCountAlign8);
        Adds(this->pointYLocal, this->pointYLocal, -seedY, pointsCountAlign8);
        Adds(this->pointZLocal, this->pointZLocal, -seedZ, pointsCountAlign8);
    }

    __aicore__ inline void RotationTransform(uint32_t pointsCountAlign8)
    {
        // 旋转后X坐标：rotX = R00*dX + R01*dY + R02*dZ
        Muls(this->rotMatXLocal, this->pointXLocal, this->rotMatX0, pointsCountAlign8);
        Axpy(this->rotMatXLocal, this->pointYLocal, this->rotMatX1, pointsCountAlign8);
        Axpy(this->rotMatXLocal, this->pointZLocal, this->rotMatX2, pointsCountAlign8);
        // 旋转后Y坐标：rotY = R10*dX + R11*dY + R12*dZ
        Muls(this->rotMatYLocal, this->pointXLocal, this->rotMatY0, pointsCountAlign8);
        Axpy(this->rotMatYLocal, this->pointYLocal, this->rotMatY1, pointsCountAlign8);
        Axpy(this->rotMatYLocal, this->pointZLocal, this->rotMatY2, pointsCountAlign8);
        // 旋转后Z坐标：rotZ = R20*dX + R21*dY + R22*dZ
        Muls(this->rotMatZLocal, this->pointXLocal, this->rotMatZ0, pointsCountAlign8);
        Axpy(this->rotMatZLocal, this->pointYLocal, this->rotMatZ1, pointsCountAlign8);
        Axpy(this->rotMatZLocal, this->pointZLocal, this->rotMatZ2, pointsCountAlign8);
    }

    __aicore__ inline void CheckCylinderConstraints(uint32_t pointsCountAlign8, uint32_t pointsCountAlign64)
    {
        // 计算距离平方：d² = rotY² + rotZ²（节省内存，复用rotMatYLocal）
        Mul(this->rotMatYLocal, this->rotMatYLocal, this->rotMatYLocal, pointsCountAlign8);
        Mul(this->rotMatZLocal, this->rotMatZLocal, this->rotMatZLocal, pointsCountAlign8);
        Add(this->rotMatYLocal, this->rotMatYLocal, this->rotMatZLocal, pointsCountAlign8);
        // 距离平方 < 半径平方（在圆柱半径内）
        CompareScalar(this->rotYMaskLocal, this->rotMatYLocal, this->tilingPara->radius, AscendC::CMPMODE::LT, pointsCountAlign64);
        // 旋转后X坐标 > 最小高度
        CompareScalar(this->rotXMaskLocal, this->rotMatXLocal, this->tilingPara->hMin, AscendC::CMPMODE::GT, pointsCountAlign64);
        And<uint16_t>(
            this->rotYMaskLocal.ReinterpretCast<uint16_t>(),
            this->rotYMaskLocal.ReinterpretCast<uint16_t>(),
            this->rotXMaskLocal.ReinterpretCast<uint16_t>(),
            (CEIL_DIV8(pointsCountAlign64) + 1) / 2);
        // 旋转后X坐标 < 最大高度
        CompareScalar(this->rotXMaskLocal, this->rotMatXLocal, this->tilingPara->hMax, AscendC::CMPMODE::LT, pointsCountAlign64);
        And<uint16_t>(
            this->rotYMaskLocal.ReinterpretCast<uint16_t>(),
            this->rotYMaskLocal.ReinterpretCast<uint16_t>(),
            this->rotXMaskLocal.ReinterpretCast<uint16_t>(),
            (CEIL_DIV8(pointsCountAlign64) + 1) / 2);
    }

    __aicore__ inline void CylinderQueryCompute()
    {
        // 计算对齐后的点数（8字节对齐和64字节对齐）
        uint32_t pointsCountAlign8 = this->curBlockCount * POINTS_PER_BLOCK;
        uint32_t pointsCountAlign64 = CEIL_ALIGN64(pointsCountAlign8);
        // 分离XYZ坐标分量
        SeparateXYZComponents();
        // 计算相对位置
        ComputeRelativePosition(pointsCountAlign8);
        // 旋转矩阵变换
        RotationTransform(pointsCountAlign8);
        // 检查圆柱体约束条件
        CheckCylinderConstraints(pointsCountAlign8, pointsCountAlign64);
        // 基于条件掩码选择结果
        Select(this->outLocal, this->rotYMaskLocal, this->outLocal, float(int32_t(this->tilingPara->numPoints)),
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->curPointsCount);
    }

private:
    TPipe pipeline;
    TilingPara *tilingPara;

    uint32_t curBlockCount;    // 当前迭代处理的块数
    uint32_t curPointsCount;    // 当前迭代处理的点数
    uint32_t taskStartOffset;    // 任务起始偏移
    uint32_t curBatchIdx;    // 当前批次索引

    float seedX, seedY, seedZ;

    float rotMatX0, rotMatX1, rotMatX2;    // 旋转矩阵元素
    float rotMatY0, rotMatY1, rotMatY2;
    float rotMatZ0, rotMatZ1, rotMatZ2;

    GlobalTensor<float> seedXyzGm, pointXyzGm, rotMatGm, outPointsGlobal, defaultIdxGm;

    TBuf<TPosition::VECCALC> pointXyzBuf;    // 点云原始数据
    TBuf<TPosition::VECCALC> pointXBuf, pointYBuf, pointZBuf; // XYZ分量
    TBuf<TPosition::VECCALC> maskPatternXBuf, maskPatternYBuf, maskPatternZBuf;    // XYZ分量提取模式
    TBuf<TPosition::VECCALC> outBuf;     // 输出结果
    TBuf<TPosition::VECCALC> rotMatBuf;    // 旋转矩阵
    TBuf<TPosition::VECCALC> rotMatXBuf, rotMatYBuf, rotMatZBuf;     // 旋转后XYZ分量
    TBuf<TPosition::VECCALC> rotYMaskBuf, rotXMaskBuf;    // XY方向距离掩码

    LocalTensor<float> pointXyzLocal;    // 点云原始数据
    LocalTensor<float> pointXLocal, pointYLocal, pointZLocal; // XYZ分量
    LocalTensor<uint32_t> maskPatternXLocal, maskPatternYLocal, maskPatternZLocal;    // XYZ分量提取模式
    LocalTensor<float> outLocal;     // 输出结果
    LocalTensor<float> rotMatLocal;    // 旋转矩阵
    LocalTensor<float> rotMatXLocal, rotMatYLocal, rotMatZLocal;     // 旋转后XYZ分量
    LocalTensor<uint8_t> rotYMaskLocal, rotXMaskLocal;    // XY方向距离掩码
};

#endif // OP_KERNEL_CYLINDER_QUERY_H