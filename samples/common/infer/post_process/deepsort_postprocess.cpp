/*
 * Copyright (c) ModelZoo. 2026-2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "deepsort_postprocess.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <cmath>

namespace Infer {
namespace DeepSort {

const int YOLO_INPUT_SIZE = 640;
const int COCO_NUM_CLASSES = 80;        // COCO 数据集类别总数
const int YOLO_PRED_CHANNELS = 85;      // 每个 Anchor 输出维度 = 4(坐标) + 1(obj) + 80(类别)
const int YOLO_NUM_ANCHORS = 3;         // 每个网格点的 Anchor 数量
const int PERSON_CLASS_ID = 0;          // COCO 中 "Person" 的类别编号
const float FAST_PRUNE_THRESHOLD = 0.25f;  // 快速剪枝阈值，低于此值跳过解码
const int MAX_DETECTION_BOXES = 100;    // NMS 后最大保留框数，防止 ReID 假死
const int FP16_TYPE_SIZE = 16;          // FP16 半精度类型位宽
const int FP32_TYPE_SIZE = 32;          // FP32 单精度类型位宽
const int BITS_PER_BYTE = 8;            // 每字节位数
const int YOLO_BBOX_FIELDS = 5;         // 坐标 4 维 + objectness 1 维
const float IOU_EPSILON = 1e-5f;        // IoU 计算防除零小量

const std::vector<uint32_t> EXPANDED_STRIDES = {8, 16, 32};
const std::vector<std::vector<uint32_t>> ANCHOR_GRIDS = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};

inline static float Sigmod(float a) { return 1.0f / (1.0f + std::exp(-a)); }

struct YoloLayerCtx {
    const float *data;
    uint32_t outH;
    uint32_t outW;
    uint32_t anchors;
    size_t stride;
    int strideIdx;
};

struct YoloProjCtx {
    float scale;
    int padX, padY, origW, origH;
    float confThres;
    std::vector<DetectBox> *boxes;
};

// 反解单个预测框 (Anchor) 的坐标与类别信息
static void DecodeAnchor(const YoloLayerCtx &lCtx, const YoloProjCtx &pCtx,
                         uint32_t h, uint32_t w, uint32_t k)
{
    // 基于 NPU 底层输出张量的物理内存步长进行三维平铺寻址计算
    // 公式: k*(H*W*stride) + h*(W*stride) + w*stride
    uint32_t baseOffset = k * (lCtx.outH * lCtx.outW * lCtx.stride) +
                          h * (lCtx.outW * lCtx.stride) + w * lCtx.stride;

    // 第 4 位为目标包围盒的置信分数 (Objectness Score)
    float objScoreVal = Sigmod(lCtx.data[baseOffset + 4]);
    if (objScoreVal <= FAST_PRUNE_THRESHOLD) {
        return;
    }

    float maxClassScore = -1000.0f;  // 初始化为极小值确保任何类别得分都能胜出
    uint32_t maxClassIdx = 0;

    // 从第 5 位开始遍历 80 个 COCO 类别取出得分最高者
    for (uint32_t c = 0; c < COCO_NUM_CLASSES; c++) {
        float classScoreVal = lCtx.data[baseOffset + YOLO_BBOX_FIELDS + c];
        if (classScoreVal > maxClassScore) {
            maxClassScore = classScoreVal;
            maxClassIdx = c;
        }
    }

    // NMS 置信度 = 含有目标的概率 × 属于该类别的概率
    float confScore = Sigmod(maxClassScore) * objScoreVal;

    if (confScore > pCtx.confThres && maxClassIdx == PERSON_CLASS_ID) {
        // 解码特征图尺度下的真实坐标：
        // bx, by 经过 Sigmoid 放缩并通过加上当前网格的 x,y 偏移确定中心点
        // YOLOv5 解码公式: (sigmoid(tx) * 2 + grid - 0.5) * stride
        float bx = (Sigmod(lCtx.data[baseOffset + 0]) * 2 + w - 0.5f) *
                   EXPANDED_STRIDES[lCtx.strideIdx];
        float by = (Sigmod(lCtx.data[baseOffset + 1]) * 2 + h - 0.5f) *
                   EXPANDED_STRIDES[lCtx.strideIdx];

        // bw, bh 计算框的长宽: (sigmoid(tw) * 2)^2 * anchor_size
        float bwTmp = Sigmod(lCtx.data[baseOffset + 2]) * 2;
        float bw = bwTmp * bwTmp * ANCHOR_GRIDS[lCtx.strideIdx][k * 2];
        float bhTmp = Sigmod(lCtx.data[baseOffset + 3]) * 2;
        float bh = bhTmp * bhTmp * ANCHOR_GRIDS[lCtx.strideIdx][k * 2 + 1];

        // 逆向反投影：由于预处理时对不同尺寸画幅做了 letterbox 边距填充(Padding)
        // 需除以 scale 并减去原图的偏差回归坐标极点到原始尺寸
        bx = (bx - pCtx.padX) / pCtx.scale;
        by = (by - pCtx.padY) / pCtx.scale;
        bw /= pCtx.scale;
        bh /= pCtx.scale;

        // 防止解码出的预测框飞出屏幕外的硬性阈值夹逼保证
        bx = std::max(0.0f, std::min(bx, (float)pCtx.origW));
        by = std::max(0.0f, std::min(by, (float)pCtx.origH));

        DetectBox box;
        box.classID = maxClassIdx;
        box.confidence = confScore;
        box.x1 = bx - bw / 2.0f;
        box.y1 = by - bh / 2.0f;
        box.x2 = bx + bw / 2.0f;
        box.y2 = by + bh / 2.0f;
        pCtx.boxes->push_back(box);
    }
}

static void ParseYOLOLayer(const YoloLayerCtx &lCtx, const YoloProjCtx &pCtx)
{
    for (uint32_t h = 0; h < lCtx.outH; ++h) {
        for (uint32_t w = 0; w < lCtx.outW; ++w) {
            for (uint32_t k = 0; k < lCtx.anchors; ++k) {
                DecodeAnchor(lCtx, pCtx, h, w, k);
            }
        }
    }
}

void NMS(std::vector<DetectBox> &inputBoxes, float nmsThresh)
{
    // 防御性剔除可能存在因为 NPU 偶发量化噪声抛出的 NaN
    inputBoxes.erase(std::remove_if(inputBoxes.begin(), inputBoxes.end(),
                                    [](const DetectBox &box) {
                                        return std::isnan(box.confidence) ||
                                               std::isinf(box.confidence) ||
                                               std::isnan(box.x1) ||
                                               std::isnan(box.y1) ||
                                               std::isnan(box.x2) ||
                                               std::isnan(box.y2);
                                    }),
                     inputBoxes.end());

    std::sort(inputBoxes.begin(), inputBoxes.end(),
              [](const DetectBox &a, const DetectBox &b) {
                  return a.confidence > b.confidence;
              });

    std::vector<DetectBox> result;
    std::vector<bool> isSuppressed(inputBoxes.size(), false);

    for (size_t i = 0; i < inputBoxes.size(); ++i) {
        if (isSuppressed[i]) {
            continue;
        }
        result.push_back(inputBoxes[i]);

        for (size_t j = i + 1; j < inputBoxes.size(); ++j) {
            if (isSuppressed[j]) {
                continue;
            }

            float xx1 = std::max(inputBoxes[i].x1, inputBoxes[j].x1);
            float yy1 = std::max(inputBoxes[i].y1, inputBoxes[j].y1);
            float xx2 = std::min(inputBoxes[i].x2, inputBoxes[j].x2);
            float yy2 = std::min(inputBoxes[i].y2, inputBoxes[j].y2);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);
            float inter = w * h;
            float area1 = inputBoxes[i].width() * inputBoxes[i].height();
            float area2 = inputBoxes[j].width() * inputBoxes[j].height();

            float iou = inter / (area1 + area2 - inter + IOU_EPSILON);
            if (iou >= nmsThresh) {
                isSuppressed[j] = true;
            }
        }
    }
    inputBoxes = result;

    // 安全防暴毙机制：如果 NPU 因意外杂音爆出大量假阳性框，
    // 限制最大只处理前 MAX_DETECTION_BOXES 个置信度最高的框(已经过 NMS 降序)，
    // 防止随后的 ReID 特征提取被执行数千次导致长达几十分钟的彻底假死卡机。
    if (inputBoxes.size() > MAX_DETECTION_BOXES) {
        inputBoxes.resize(MAX_DETECTION_BOXES);
    }
}

int PostProcessYOLO(const std::vector<TensorBuf> &outBufs,
                    const std::vector<TensorDesc> &outDescs,
                    std::vector<DetectBox> &detectedBoxes,
                    const YoloImageInfo &imgInfo)
{
    float scale = std::min((float)YOLO_INPUT_SIZE / imgInfo.origWidth,
                           (float)YOLO_INPUT_SIZE / imgInfo.origHeight);
    int padX = (YOLO_INPUT_SIZE - std::round(imgInfo.origWidth * scale)) / 2;
    int padY = (YOLO_INPUT_SIZE - std::round(imgInfo.origHeight * scale)) / 2;

    YoloProjCtx pCtx = {scale,
                        padX,
                        padY,
                        imgInfo.origWidth,
                        imgInfo.origHeight,
                        imgInfo.confThres,
                        &detectedBoxes};

    for (size_t detectIdx = 0; detectIdx < outBufs.size(); ++detectIdx) {
        const TensorDesc &desc = outDescs[detectIdx];
        if (desc.dimCount != 5) {  // 5维张量: [batch, anchors, H, W, channels]
            continue;
        }

        uint32_t numAnchors = desc.dims[1];
        uint32_t outHeight = desc.dims[2];
        uint32_t outWidth = desc.dims[3];
        uint32_t numClassesPlus5 = desc.dims[4];

        if (numAnchors != YOLO_NUM_ANCHORS || numClassesPlus5 != YOLO_PRED_CHANNELS) {
            continue;
        }

        // 模型输出通常包括 80*80, 40*40, 20*20 的特征图，根据边长映射对应尺度的
        // Stride 指引
        int strideIdx = -1;
        if (outWidth == 80) {       // stride=8 小目标检测层
            strideIdx = 0;
        } else if (outWidth == 40) {  // stride=16 中目标检测层
            strideIdx = 1;
        } else if (outWidth == 20) {  // stride=32 大目标检测层
            strideIdx = 2;
        }
        if (strideIdx == -1) {
            continue;
        }

        // 这里是修复 NNN 和 SVP_NNN 对齐的核心关键：如果 NPU 未实施致密层宽位Padding
        size_t elementBytes = desc.typeSize / BITS_PER_BYTE;
        size_t realChannelStride = numClassesPlus5;
        if (outBufs[detectIdx].stride != 0) {
            realChannelStride = outBufs[detectIdx].stride / elementBytes;
        }

        std::vector<float> fp16ToFp32Cache;
        const float *outData = nullptr;

        // 动态根据数据解析精度：NNN 模型为原生 FP16(半精度)，SVP_NNN量化后抛出 FP32
        if (desc.typeSize == FP16_TYPE_SIZE) {
            size_t elements = outBufs[detectIdx].size / elementBytes;
            fp16ToFp32Cache.resize(elements);
            const uint16_t *fp16Ptr =
                static_cast<const uint16_t *>(outBufs[detectIdx].GetRawPtr());
            for (size_t i = 0; i < elements; ++i) {
                fp16ToFp32Cache[i] = Infer::HalfToFloat(fp16Ptr[i]);
            }
            outData = fp16ToFp32Cache.data();
        } else if (desc.typeSize == FP32_TYPE_SIZE) {
            outData = static_cast<const float *>(outBufs[detectIdx].GetRawPtr());
        } else {
            continue;
        }

        YoloLayerCtx lCtx = {outData,    outHeight,         outWidth,
                             numAnchors, realChannelStride, strideIdx};
        ParseYOLOLayer(lCtx, pCtx);
    }

    NMS(detectedBoxes, imgInfo.nmsThres);
    return Infer::SUCCESS;
}

} // namespace DeepSort
} // namespace Infer
