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

#ifndef OP_KERNEL_FURTHEST_POINT_SAMPLING_H
#define OP_KERNEL_FURTHEST_POINT_SAMPLING_H

#include "kernel_operator.h"
#include "furthest_point_sampling_base.h"

using namespace AscendC;

class KernelFurthestPointSampling {
public:
    // 构造函数：初始化核函数
    __aicore__ inline KernelFurthestPointSampling(GM_ADDR pointXyz, GM_ADDR dist,
        GM_ADDR outIdx, GM_ADDR workspace, TilingPara *tilingPara)
    {
        // 初始化tiling参数
        this->tilingPara = tilingPara;
        this->tileByteSize = this->tilingPara->tileSize * FLOAT_SIZE;
        this->lastTileByteSize = this->tilingPara->lastTileSize * FLOAT_SIZE;
        this->tileByteSize = this->tilingPara->tileSize * FLOAT_SIZE;
        this->lastTileByteSize = this->tilingPara->lastTileSize * FLOAT_SIZE;
        // 初始化Global Memory
        InitGlobalMemory(pointXyz, dist, outIdx, workspace);

        // 必须32字节对齐
        for (int i = 0; i < POINT_DIMENSION; i++) {
            // 初始化点云XYZ缓冲区
            this->pipeline.InitBuffer(this->pointXyzBuf[i], SINGLE_BUFFER, this->tileByteSize);
            this->pipeline.InitBuffer(this->pointCompBuf[i], SINGLE_BUFFER, this->tileByteSize);
        }
        // 初始化距离缓冲区
        this->pipeline.InitBuffer(this->distBuf, SINGLE_BUFFER, this->tileByteSize);
        this->pipeline.InitBuffer(this->distCompBuf, SINGLE_BUFFER, this->tileByteSize);
        this->pipeline.InitBuffer(this->tempBuf, SINGLE_BUFFER, this->tilingPara->ubWorkSpaceSize);
        this->pipeline.InitBuffer(this->outIdxBuf, SINGLE_BUFFER, DATA_NUM_ALIGN_WITH_1024 * INT32_SIZE);
        this->pipeline.InitBuffer(this->idxCompBuf, SINGLE_BUFFER, this->tilingPara->pairBufSize);
        // 初始化采样点缓冲区
        this->pipeline.InitBuffer(this->sampledPointBuf, SINGLE_BUFFER, MIN_BLOCK_SIZE * POINT_DIMENSION * 2);
        // 分配本地张量
        for (int i = 0; i < POINT_DIMENSION; i++) {
            this->pointXyzLocal[i] = this->pointXyzBuf[i].AllocTensor<float>();
            this->pointCompLocal[i] = this->pointCompBuf[i].AllocTensor<float>();
        }

        this->distLocal = this->distBuf.AllocTensor<float>();
        this->distCompLocal = this->distCompBuf.AllocTensor<float>();
        this->tempLocal = this->tempBuf.AllocTensor<float>();
        this->outIdxLocal = this->outIdxBuf.AllocTensor<int32_t>();
        this->idxCompLocal = this->idxCompBuf.AllocTensor<float>();
        this->sampledPointLocal = this->sampledPointBuf.AllocTensor<float>();
    }
    // 析构函数
    __aicore__ inline ~KernelFurthestPointSampling()
    {
        for (int i = 0; i < POINT_DIMENSION; i++) {
            this->pointXyzBuf[i].FreeTensor(this->pointXyzLocal[i]);
            this->pointCompBuf[i].FreeTensor(this->pointCompLocal[i]);
        }
        this->distBuf.FreeTensor(this->distLocal);
        this->distCompBuf.FreeTensor(this->distCompLocal);
        this->tempBuf.FreeTensor(this->tempLocal);
        this->outIdxBuf.FreeTensor(this->outIdxLocal);
        this->idxCompBuf.FreeTensor(this->idxCompLocal);
        this->sampledPointBuf.FreeTensor(this->sampledPointLocal);
    }
    // 主流程函数
    __aicore__ inline void Process()
    {
        uint32_t numBatch = (GetBlockIdx() < this->tilingPara->numLargeCore) ?
            (this->tilingPara->batchsPerLargeCore) : (this->tilingPara->batchsPerLittleCore);
        // 遍历每个batch
        for (this->batchIdx = 0; this->batchIdx < numBatch; this->batchIdx++) {
            // 计算当前batch的偏移量
            this->pointOffset = this->batchIdx * this->tilingPara->totalPoints * POINT_DIMENSION;
            this->distOffset = this->batchIdx * this->tilingPara->totalPoints;

            DataCopyIn(0);
            // 特殊情况：只采样一个点
            if (this->tilingPara->nSample == 1) {
                DataCopyOut(0);
            }
            if (this->tilingPara->numTile == 1) {
                // 单块数据处理
                ProcessSingleTileData();
            } else {
                // 多块数据处理
                ProcessTiledData();
            }
        }
    }

private:
    // 初始化全局内存
    __aicore__ inline void InitGlobalMemory(GM_ADDR pointXyz, GM_ADDR dist, GM_ADDR outIdx, GM_ADDR workspace)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint64_t numPointsOnLargeCore = this->tilingPara->batchsPerLargeCore * this->tilingPara->totalPoints;
        uint64_t numOutIdxOnLargeCore = this->tilingPara->batchsPerLargeCore * this->tilingPara->nSample;
        uint64_t numPoints, numOutIdx, pointsOffset, idxOffset;
        // 分配内存偏移
        if (blockIdx < this->tilingPara->numLargeCore) {
            numPoints = numPointsOnLargeCore;
            numOutIdx = numOutIdxOnLargeCore;
            pointsOffset = numPoints * blockIdx;
            idxOffset = numOutIdx * blockIdx;
        } else {
            numPoints = this->tilingPara->batchsPerLittleCore * this->tilingPara->totalPoints;
            numOutIdx = this->tilingPara->batchsPerLittleCore * this->tilingPara->nSample;
            pointsOffset = this->tilingPara->numLargeCore * numPointsOnLargeCore +
                (blockIdx - this->tilingPara->numLargeCore) * numPoints;
            idxOffset = this->tilingPara->numLargeCore * numOutIdxOnLargeCore +
                (blockIdx - this->tilingPara->numLargeCore) * numOutIdx;
        }
        GM_ADDR userSpace = GetUserWorkspace(workspace);
        // 设置全局张量
        this->pointXyzGlobal.SetGlobalBuffer((__gm__ float*)pointXyz +
            pointsOffset * POINT_DIMENSION, numPoints * POINT_DIMENSION);
        this->distGlobal.SetGlobalBuffer((__gm__ float*)dist + pointsOffset, numPoints);
        this->outIdxGlobal.SetGlobalBuffer((__gm__ int32_t*)outIdx + idxOffset, numOutIdx);
        this->distTmpGlobal.SetGlobalBuffer((__gm__ float*)userSpace + pointsOffset, numPoints);
    }

    // 从全局内存拷贝数据到本地缓冲区
    __aicore__ inline void DataCopyIn(uint32_t iter)
    {
        // 偏移计算
        uint32_t pointXOffsetLocal = 0;
        uint32_t pointYOffsetLocal = DATA_NUM_ALIGN_WITH_32;
        uint32_t pointZOffsetLocal = DATA_NUM_ALIGN_WITH_64;
        uint32_t pointXOffsetGm = this->pointOffset + this->furthestDistIdx;
        uint32_t pointYOffsetGm = pointXOffsetGm + this->tilingPara->totalPoints;
        uint32_t pointZOffsetGm = pointYOffsetGm + this->tilingPara->totalPoints;
        uint32_t outIdxOffset = iter & (DATA_NUM_ALIGN_WITH_1024 - 1);
        uint32_t sampleMask = MIN_BLOCK_SIZE * POINT_DIMENSION / FLOAT_SIZE;

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        // 拷贝采样点坐标
        this->sampledPointLocal.SetValue(pointXOffsetLocal, pointXyzGlobal.GetValue(pointXOffsetGm));
        this->sampledPointLocal.SetValue(pointYOffsetLocal, pointXyzGlobal.GetValue(pointYOffsetGm));
        this->sampledPointLocal.SetValue(pointZOffsetLocal, pointXyzGlobal.GetValue(pointZOffsetGm));

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        // 取负值，用于距离计算
        Muls<float>(this->sampledPointLocal, this->sampledPointLocal, float(-1.0), sampleMask);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        // 获取采样点坐标值
        this->sampledPointXyz[X_INDEX] = this->sampledPointLocal.GetValue(pointXOffsetLocal);
        this->sampledPointXyz[Y_INDEX] = this->sampledPointLocal.GetValue(pointYOffsetLocal);
        this->sampledPointXyz[Z_INDEX] = this->sampledPointLocal.GetValue(pointZOffsetLocal);
        // 存储输出索引
        this->outIdxLocal.SetValue(outIdxOffset, this->furthestDistIdx);
        this->furthestDistVal = 0;
        this->furthestDistIdx = 0;
    }

    // 将结果从本地缓冲区拷贝到GM
    __aicore__ inline void DataCopyOut(uint32_t iter)
    {
        uint32_t dataNum = DATA_NUM_ALIGN_WITH_1024;
        uint64_t batchOffset = this->batchIdx * this->tilingPara->nSample;
        DataCopyExtParams copyExtParams = {1, FLOAT_SIZE, 0, 0, 0};
        // 当采样的点数小于256且不是最后一次循环时返回
        if ((iter != 0) && (((iter + 1) & (dataNum - 1)) != 0) && ((iter + 1) != this->tilingPara->nSample)) {
            return ;
        }
        // 计算拷贝偏移和长度
        if (((iter + 1) & (dataNum - 1)) == 0) {
            batchOffset = batchOffset + iter / dataNum * dataNum;
            copyExtParams.blockLen = DATA_NUM_ALIGN_WITH_1024 * INT32_SIZE;
        } else if (iter == this->tilingPara->nSample - 1) {
            // 最后一次迭代处理
            batchOffset = batchOffset + (this->tilingPara->nSample / dataNum * dataNum);
            copyExtParams.blockLen = INT32_SIZE *
                (this->tilingPara->nSample - (this->tilingPara->nSample / dataNum * dataNum));
        }

        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

        DataCopyPadCustom_UB2GM(outIdxGlobal[batchOffset], this->outIdxLocal, copyExtParams.blockLen / INT32_SIZE);
    }

    // 单块数据处理
    __aicore__ inline void ProcessSingleTileData()
    {
        for (uint32_t iter = 1; iter < this->tilingPara->nSample; iter++) {
            if (iter == 1) {
                // 第一次采样特殊处理
                HandleFirstSampling(0);
            } else {
                // 计算三维坐标差的平方
                CalcThreeDimDeltaSquared();

                pipe_barrier(PIPE_V);
                // 计算欧式距离
                CalcEuclideanDistance();

                pipe_barrier(PIPE_V);
                // 距离更新
                ProcessDistanceUpdate(0, 0);
            }
            pipe_barrier(PIPE_V);
            // 更新全局最远点
            UpdateGlobalFurthestPoint();

            DataCopyIn(iter);
            DataCopyOut(iter);
        }
    }

    __aicore__ inline void CalcDeltaSquaredAndCopyIn(uint32_t tileIdx)
    {
        // 计算X维度差平方，同时拷贝X数据
        CalcSingleDimDeltaSquared(this->pointXyzLocal[X_INDEX], this->pointCompLocal[X_INDEX],
            this->sampledPointXyz[X_INDEX]);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        PointDataCopyIn(X_INDEX, tileIdx);
        // 计算Y维度差平方，同时拷贝Y数据
        CalcSingleDimDeltaSquared(this->pointXyzLocal[Y_INDEX], this->pointCompLocal[Y_INDEX],
            this->sampledPointXyz[Y_INDEX]);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        PointDataCopyIn(Y_INDEX, tileIdx);
        // 计算Z维度差平方，同时拷贝Y数据
        CalcSingleDimDeltaSquared(this->pointXyzLocal[Z_INDEX], this->pointCompLocal[Z_INDEX],
            this->sampledPointXyz[Z_INDEX]);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        PointDataCopyIn(Z_INDEX, tileIdx);
    }

    // 分块数据处理
    __aicore__ inline void ProcessTiledData()
    {
        for (uint32_t iter = 1; iter < this->tilingPara->nSample; iter++) {
            // 遍历所有数据块
            for (uint32_t tileIdx = 0; tileIdx < this->tilingPara->numTile; tileIdx++) {
                if (iter == 1) {
                    // 第一次采样特殊处理
                    HandleFirstSampling(tileIdx);
                } else {
                    uint32_t preTileIdx = (tileIdx + this->tilingPara->numTile - 1) % this->tilingPara->numTile;

                    CalcDeltaSquaredAndCopyIn(tileIdx);

                    pipe_barrier(PIPE_ALL);
                    // 计算欧式距离
                    CalcEuclideanDistance();

                    pipe_barrier(PIPE_ALL);
                    // 距离更新
                    ProcessDistanceUpdate(tileIdx, preTileIdx);

                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);

                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

                    DistTmpDataCopyIn(tileIdx);
                }
            }
            pipe_barrier(PIPE_V);
            // 更新全局最远点
            UpdateGlobalFurthestPoint();

            DataCopyIn(iter);
            DataCopyOut(iter);
        }
    }

    // 处理第一次采样：初始化距离计算
    __aicore__ inline void HandleFirstSampling(uint32_t tileIdx)
    {
        // 拷贝X数据，然后计算X维度差平方
        PointDataCopyIn(X_INDEX, tileIdx);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        CalcSingleDimDeltaSquared(this->pointXyzLocal[X_INDEX], this->pointCompLocal[X_INDEX],
            this->sampledPointXyz[X_INDEX]);
        // 拷贝Y数据，然后计算Y维度差平方
        PointDataCopyIn(Y_INDEX, tileIdx);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        CalcSingleDimDeltaSquared(this->pointXyzLocal[Y_INDEX], this->pointCompLocal[Y_INDEX],
            this->sampledPointXyz[Y_INDEX]);
        // 拷贝Z数据，然后计算Y维度差平方
        PointDataCopyIn(Z_INDEX, tileIdx);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

        CalcSingleDimDeltaSquared(this->pointXyzLocal[Z_INDEX], this->pointCompLocal[Z_INDEX],
            this->sampledPointXyz[Z_INDEX]);

        pipe_barrier(PIPE_V);
        // 计算欧式距离
        CalcEuclideanDistance();

        DistDataCopyIn(tileIdx);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
        // 距离更新
        ProcessDistanceUpdate(tileIdx, tileIdx);
    }

    // 点数据拷贝
    __aicore__ inline void PointDataCopyIn(uint32_t dimIdx, uint32_t tileIdx)
    {
        DataCopyParams copyParam = {1, 0, 0, 0};
        uint64_t pointGmOffset = this->pointOffset + this->tilingPara->tileSize * tileIdx +
            this->tilingPara->totalPoints * dimIdx;
        // 设置拷贝块长度（最后一块特殊处理）
        if (tileIdx == (this->tilingPara->numTile - 1)) {
            copyParam.blockLen = this->lastTileByteSize;
        } else {
            copyParam.blockLen = this->tileByteSize;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

        DataCopyPadCustom_GM2UB(this->pointXyzLocal[dimIdx], pointXyzGlobal[pointGmOffset],
            copyParam.blockLen / FLOAT_SIZE);
    }

    // 距离数据拷贝
    __aicore__ inline void DistDataCopyIn(uint32_t tileIdx)
    {
        DataCopyParams copyParam = {1, 0, 0, 0};
        uint64_t distGmOffset = this->distOffset + this->tilingPara->tileSize * tileIdx;

        if (tileIdx == (this->tilingPara->numTile - 1)) {
            copyParam.blockLen = this->lastTileByteSize;
        } else {
            copyParam.blockLen = this->tileByteSize;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

        DataCopyPadCustom_GM2UB(this->distLocal, distGlobal[distGmOffset], copyParam.blockLen / FLOAT_SIZE);
    }

    __aicore__ inline void DistTmpDataCopyIn(uint32_t tileIdx)
    {
        DataCopyParams copyParam = {1, 0, 0, 0};
        uint64_t distTmpOffset = this->distOffset + this->tilingPara->tileSize * tileIdx;

        if (tileIdx == (this->tilingPara->numTile - 1)) {
            copyParam.blockLen = this->lastTileByteSize;
        } else {
            copyParam.blockLen = this->tileByteSize;
        }

        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

        DataCopyPadCustom_GM2UB(this->distLocal, distTmpGlobal[distTmpOffset], copyParam.blockLen / FLOAT_SIZE);
    }

    __aicore__ inline void DistDataCopyOut(uint32_t tileIdx)
    {
        DataCopyExtParams copyParam = {1, 0, 0, 0, 0};
        uint64_t offset = this->distOffset + this->tilingPara->tileSize * tileIdx;

        if (tileIdx == (this->tilingPara->numTile - 1)) {
            copyParam.blockLen = this->lastTileByteSize;
        } else {
            copyParam.blockLen = this->tileByteSize;
        }

        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

        DataCopyPadCustom_UB2GM(distTmpGlobal[offset], this->distLocal, copyParam.blockLen / FLOAT_SIZE);
    }

    // 计算三维坐标差的平方
    __aicore__ inline void CalcThreeDimDeltaSquared()
    {
        uint32_t pointCompOffset = 0;
        uint32_t dataNum = this->tilingPara->tileSize;
        uint32_t repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
        repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;
        uint32_t computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;

        // 循环处理每个256字节对齐的数据块
        for (; dataNum > 0; dataNum = dataNum - computeDataNum) {
            repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
            repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;

            set_flag(PIPE_S, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID3);
            // 计算每个维度的坐标差：pointComp = pointXyz - sampledPointXyz
            for (int i = 0; i < POINT_DIMENSION; i++) {
                Adds<float>(this->pointCompLocal[i][pointCompOffset], this->pointXyzLocal[i][pointCompOffset],
                    this->sampledPointXyz[i], DATA_NUM_ALIGN_WITH_256, repeatTime, {1, 1, 8, 8});
            }

            pipe_barrier(PIPE_V);
            // 计算每个维度的平方：pointComp = pointComp * pointComp
            for (int i = 0; i < POINT_DIMENSION; i++) {
                Mul<float>(this->pointCompLocal[i][pointCompOffset], this->pointCompLocal[i][pointCompOffset],
                    this->pointCompLocal[i][pointCompOffset], DATA_NUM_ALIGN_WITH_256,
                    repeatTime, {1, 1, 1, 8, 8, 8});
            }
            computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;
            pointCompOffset = pointCompOffset + computeDataNum;
        }
    }

    // 计算单维度坐标差的平方
    __aicore__ inline void CalcSingleDimDeltaSquared(LocalTensor<float> &pointXyzLocal,
        LocalTensor<float> &pointCompLocal, float sampledPointXyz)
    {
        uint32_t pointOffset = 0;
        uint32_t dataNum = this->tilingPara->tileSize;
        uint32_t repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
        repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;
        uint32_t computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;

        // 循环处理每个256字节对齐的数据块
        for (; dataNum > 0; dataNum = dataNum - computeDataNum) {
            repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
            repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;

            set_flag(PIPE_S, PIPE_V, EVENT_ID3);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID3);
            // 计算坐标差：pointComp = pointXyz - sampledPointXyz
            Adds<float>(pointCompLocal[pointOffset], pointXyzLocal[pointOffset], sampledPointXyz,
                DATA_NUM_ALIGN_WITH_256, repeatTime, {1, 1, 8, 8});

            pipe_barrier(PIPE_V);
            // 计算平方：pointComp = pointComp * pointComp
            Mul<float>(pointCompLocal[pointOffset], pointCompLocal[pointOffset], pointCompLocal[pointOffset],
                DATA_NUM_ALIGN_WITH_256, repeatTime, {1, 1, 1, 8, 8, 8});

            computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;
            pointOffset = pointOffset + computeDataNum;
        }
    }

    // 计算欧式距离：distComp = (dx² + dy²) + dz²
    __aicore__ inline void CalcEuclideanDistance()
    {
        uint32_t distOffset = 0;
        uint32_t dataNum = this->tilingPara->tileSize;
        uint32_t repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
            repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;
        uint32_t computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;

        // 循环处理每个256字节对齐的数据块
        for (; dataNum > 0; dataNum = dataNum - computeDataNum) {
            repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
            repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;

            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            // 计算前两个维度的平方和：distComp = dx² + dy²
            Add<float>(this->distCompLocal[distOffset], this->pointCompLocal[X_INDEX][distOffset],
                this->pointCompLocal[Y_INDEX][distOffset], DATA_NUM_ALIGN_WITH_256, repeatTime, {1, 1, 1, 8, 8, 8});

            pipe_barrier(PIPE_V);
            // 加上第三个维度的平方：distComp = distComp + dz²
            Add<float>(this->distCompLocal[distOffset], this->distCompLocal[distOffset],
                this->pointCompLocal[Z_INDEX][distOffset], DATA_NUM_ALIGN_WITH_256, repeatTime, {1, 1, 1, 8, 8, 8});
            
            computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;
            distOffset = distOffset + computeDataNum;
        }
    }

    // 处理距离更新
    __aicore__ inline void ProcessDistanceUpdate(uint32_t tileIdx, uint32_t preTileIdx)
    {
        uint32_t distOffset = 0;
        uint32_t dataNum = this->tilingPara->tileSize;
        uint32_t repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
            repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;
        uint32_t computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;
        // 循环处理每个256字节对齐的数据块
        for (; dataNum > 0; dataNum = dataNum - computeDataNum) {
            repeatTime = (dataNum * FLOAT_SIZE) / ALIGN_DATA_SIZE;
            repeatTime = (repeatTime <= MAX_TIMES) ? repeatTime : MAX_TIMES;

            set_flag(PIPE_S, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID1);
            // 更新最小距离：dist = min(dist, distComp)
            Min<float>(this->distLocal[distOffset], this->distLocal[distOffset],
                this->distCompLocal[distOffset], DATA_NUM_ALIGN_WITH_256, repeatTime, {1, 1, 1, 8, 8, 8});

            computeDataNum = repeatTime * DATA_NUM_ALIGN_WITH_256;
            distOffset = distOffset + computeDataNum;
        }
        // 多块数据处理
        if (this->tilingPara->numTile > 1) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            DistDataCopyOut(preTileIdx);
        }

        pipe_barrier(PIPE_ALL);
        // 计算当前块的最大距离和对应索引
        uint32_t count = ((this->tilingPara->tileSize != this->tilingPara->lastTileSize) &&
            (preTileIdx == (this->tilingPara->numTile - 1))) ?
            this->tilingPara->lastTileSize : this->tilingPara->tileSize;
        uint32_t offset = preTileIdx * 2;
        // 规约操作，找到最大距离和对应索引
        ReduceMax<float>(this->idxCompLocal[offset], this->distLocal,
            this->tempLocal, count, true);
    }
    // 更新全局最远点
    __aicore__ inline void UpdateGlobalFurthestPoint()
    {
        float idxTmp;
        // 遍历所有块的结果，找到全局最远点
        for (uint32_t i = 1; i < (2 * this->tilingPara->numTile); i = (i + 2)) {
            idxTmp = this->idxCompLocal.GetValue(i);
            if (float(this->furthestDistVal) < float(this->idxCompLocal.GetValue(i-1))) {
                this->furthestDistVal = this->idxCompLocal.GetValue(i-1);
                this->furthestDistIdx = (this->tilingPara->tileSize * (i / 2)) + (*reinterpret_cast<int32_t*>(&idxTmp));
            }
        }
    }

private:
    TPipe pipeline;
    TilingPara *tilingPara;
    // 输入、工作、输出缓冲区
    TQue<QuePosition::VECIN, SINGLE_BUFFER> pointXyzBuf[POINT_DIMENSION]; // XYZ输入
    TQue<QuePosition::VECIN, SINGLE_BUFFER> pointCompBuf[POINT_DIMENSION]; /// XYZ工作缓冲区
    TQue<QuePosition::VECIN, SINGLE_BUFFER> distBuf, distCompBuf;
    TQue<QuePosition::VECOUT, SINGLE_BUFFER> tempBuf, outIdxBuf, idxCompBuf, sampledPointBuf;
    // 全局张量
    GlobalTensor<float> pointXyzGlobal, distGlobal, distTmpGlobal;
    GlobalTensor<int32_t> outIdxGlobal;
    // 本地张量
    LocalTensor<float> pointXyzLocal[POINT_DIMENSION];
    LocalTensor<float> pointCompLocal[POINT_DIMENSION];
    LocalTensor<float> distLocal, distCompLocal, idxCompLocal, sampledPointLocal, tempLocal;
    LocalTensor<int32_t> outIdxLocal;

    float sampledPointXyz[POINT_DIMENSION] {0.0f}; // 当前采样点坐标
    float furthestDistVal {0.0f}; // 当前最远距离值
    uint32_t furthestDistIdx {0}; // 当前最远点索引
    uint32_t batchIdx, tileByteSize, lastTileByteSize, pointOffset, distOffset; // 运行时变量
};

#endif // OP_KERNEL_FURTHEST_POINT_SAMPLING_H