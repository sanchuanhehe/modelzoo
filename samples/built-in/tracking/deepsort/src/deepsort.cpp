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

#include "deepsort.h"
#include "log.h"
#include "utils.h"
#include <chrono>
#include <iostream>

using namespace Infer;

const int YOLO_INPUT_SIZE = 640;
constexpr float CONF_THRES = 0.25f;
const int REID_FEATURE_DIM = 512;       // ReID 特征向量维度
const int REID_FPS_TEST_ITERS = 1000;   // ReID 帧率测试循环次数
const float NS_TO_US_FACTOR = 1000.0f;  // 纳秒转微秒的除数(与 ITERS 相乘后共同实现 ns→ms 换算)

const std::vector<uint32_t> EXPANDED_STRIDES = {8, 16, 32};
const std::vector<uint32_t> H_SIZES = {80, 40, 20};
const std::vector<uint32_t> W_SIZES{80, 40, 20};
const std::vector<std::vector<uint32_t>> ANCHOR_GRIDS = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};

DeepSortController::DeepSortController(const DeepSORTConfig &config)
    : config_(config) {}

DeepSortController::~DeepSortController()
{
    if (yoloModel_) {
        yoloModel_->UnLoadModel();
    }
    if (resnetModel_) {
        resnetModel_->UnLoadModel();
    }
}

int DeepSortController::Init()
{
    yoloModel_ = MdlCreate();
    if (yoloModel_->LoadModel(config_.yoloModelPath) != SUCCESS) {
        LOG(ERROR) << "Failed to load YOLO model: " << config_.yoloModelPath;
        return FAILED;
    }

    resnetModel_ = MdlCreate();
    if (resnetModel_->LoadModel(config_.resnetModelPath) != SUCCESS) {
        LOG(ERROR) << "Failed to load ResNet model: " << config_.resnetModelPath;
        return FAILED;
    }

    // 1. 初始化预备分配 YOLO IO Malloc 缓存池
    TensorDesc desc;
    yoloModel_->GetInTensorDescByIdx(0, desc);
    yoloInBufs_.emplace_back(desc.defaultSize, desc.defaultStride);
    for (size_t i = 0; i < yoloModel_->GetOutTensorNum(); i++) {
        yoloModel_->GetOutTensorDescByIdx(i, desc);
        yoloOutBufs_.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    // 2. 初始化预备分配 ResNet IO Malloc 缓存池
    resnetModel_->GetInTensorDescByIdx(0, desc);
    resnetInBufs_.emplace_back(desc.defaultSize, desc.defaultStride);
    for (size_t i = 0; i < resnetModel_->GetOutTensorNum(); i++) {
        resnetModel_->GetOutTensorDescByIdx(i, desc);
        resnetOutBufs_.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    LOG(INFO)
        << "DeepSortController models and buffers initialized successfully.";
    return SUCCESS;
}

std::vector<DetectBox> DeepSortController::RunYOLO(const cv::Mat &frame)
{
    std::vector<DetectBox> detectedBoxes;

    TensorDesc desc;
    yoloModel_->GetInTensorDescByIdx(0, desc);
    Infer::DeepSort::PreProcessYOLO(frame, yoloInBufs_[0], desc);

    std::vector<TensorDesc> outDescs;
    for (size_t i = 0; i < yoloModel_->GetOutTensorNum(); i++) {
        yoloModel_->GetOutTensorDescByIdx(i, desc);
        outDescs.push_back(desc);
    }

    if (yoloModel_->Execute(yoloInBufs_, yoloOutBufs_) != SUCCESS) {
        LOG(ERROR) << "YOLO Execute Failed";
        return detectedBoxes;
    }

    Infer::DeepSort::YoloImageInfo imgInfo = {
        frame.cols, frame.rows, config_.yoloConfThres, config_.yoloNmsThres};
    Infer::DeepSort::PostProcessYOLO(yoloOutBufs_, outDescs, detectedBoxes,
                                    imgInfo);
    return detectedBoxes;
}

// 核心工具：构造绝对安全的检测抠图框，防止行人出框导致 OpenCV 内存越界
cv::Rect DeepSortController::GetSafeBoxRect(const DetectBox &box, int imgCols,
                                            int imgRows)
{
    cv::Rect rect(box.x1, box.y1, box.width(), box.height());
    // 左上角原点硬性截断归零约束
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    // 宽高收敛至物理图像边界约束
    rect.width = std::min(rect.width, imgCols - rect.x);
    rect.height = std::min(rect.height, imgRows - rect.y);
    return rect;
}

// 核心工具：针对单个裁切图，执行 ResNet 行人重识别(ReID) 抽取 512 维表观特征
std::vector<float> DeepSortController::ExtractReIDFeature(const cv::Mat &crop)
{
    TensorDesc desc;
    resnetModel_->GetInTensorDescByIdx(0, desc);

    Infer::DeepSort::PreProcessResnetCrop(crop, resnetInBufs_[0], desc);
    static int callCount = 1;
    if (config_.testReidFps) {
        auto start = std::chrono::high_resolution_clock::now();
        resnetInBufs_.emplace_back(4, 0);  // 4 字节用于存储时间戳
        for (int iter = 0; iter < REID_FPS_TEST_ITERS; iter++) {
            if (resnetModel_->Execute(resnetInBufs_, resnetOutBufs_) != SUCCESS) {
                LOG(ERROR) << "ResNet Execute Failed";
                return std::vector<float>(REID_FEATURE_DIM, 0.0f);
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float msDur = *(static_cast<float*>(resnetInBufs_[resnetInBufs_.size() - 1].GetRawPtr())) / (REID_FPS_TEST_ITERS * NS_TO_US_FACTOR * callCount);
        LOG(INFO) << std::fixed << std::setprecision(2) << "execution time: " << msDur << "ms, frame rate: " << (NS_TO_US_FACTOR / msDur) << "fps";
        resnetInBufs_.pop_back();
        callCount++;
    } else {
        if (resnetModel_->Execute(resnetInBufs_, resnetOutBufs_) != SUCCESS) {
            LOG(ERROR) << "ResNet Execute Failed";
            return std::vector<float>(REID_FEATURE_DIM, 0.0f);
        }
    }

    // 获取 Output TensorDesc 用于尺寸推演
    resnetModel_->GetOutTensorDescByIdx(0, desc);

    // 承接 ResNet NPU 输出数据，支持 FP16 无损转换
    std::vector<float> feature(REID_FEATURE_DIM, 0.0f);
    size_t typeSize = desc.typeSize;
    if (typeSize == 16) {  // FP16 半精度
        const uint16_t *fp16Ptr =
            static_cast<const uint16_t *>(resnetOutBufs_[0].GetRawPtr());
        for (int k = 0; k < REID_FEATURE_DIM; k++) {
            feature[k] = Infer::HalfToFloat(fp16Ptr[k]);
        }
    } else if (typeSize == 32) {  // FP32 单精度
        const float *fp32Ptr =
            static_cast<const float *>(resnetOutBufs_[0].GetRawPtr());
        feature.assign(fp32Ptr, fp32Ptr + REID_FEATURE_DIM);
    } else {
        return feature;
    }

    // 极为关键的 L2 特征归一化 (L2 Normalization)：
    // 由于后续计算距离时使用 Cosine 夹角公式，且 NPU 处理量化与反算时范数可能飘移
    // (不为 1.0) 这里必须手动将 512 维向量约束在单位球面上，以屏蔽量化误差对角度比对的干扰
    float l2Norm = 0.0f;
    for (int k = 0; k < REID_FEATURE_DIM; k++) {
        l2Norm += feature[k] * feature[k];
    }
    l2Norm = sqrtf(l2Norm + 1e-12f);  // 加微小量防止除零
    for (int k = 0; k < REID_FEATURE_DIM; k++) {
        feature[k] /= l2Norm;
    }

    return feature;
}

std::vector<std::vector<float>>
DeepSortController::RunResNet(const cv::Mat &frame,
                              const std::vector<DetectBox> &boxes)
{
    std::vector<std::vector<float>> allFeatures;
    for (const auto &box : boxes) {
        cv::Rect rect = GetSafeBoxRect(box, frame.cols, frame.rows);
        if (rect.width <= 0 || rect.height <= 0) {
            allFeatures.push_back(std::vector<float>(REID_FEATURE_DIM, 0.0f));
            continue;
        }
        allFeatures.push_back(ExtractReIDFeature(frame(rect)));
    }
    return allFeatures;
}

std::vector<Track> DeepSortController::ProcessFrame(cv::Mat &frame)
{
    std::vector<DetectBox> yoloBoxes = RunYOLO(frame);
    std::vector<std::vector<float>> features = RunResNet(frame, yoloBoxes);

    std::vector<BoundingBox> measurements;
    std::vector<int> classIds;
    for (const auto &box : yoloBoxes) {
        BoundingBox measure;
        measure.x = (box.x1 + box.x2) / 2.0f;
        measure.y = (box.y1 + box.y2) / 2.0f;
        measure.a = box.width() / std::max(1.0f, box.height());
        measure.h = box.height();

        measurements.push_back(measure);
        classIds.push_back(box.classID);
    }

    if (!tracker_) {
        // 使用用户传入的超参数进行初始化
        NearestNeighborDistanceMetric *metric = new NearestNeighborDistanceMetric(
            config_.maxCosineDistance, config_.nnBudget);
        tracker_ = std::make_unique<Tracker>(
            metric, config_.maxIouDistance, config_.maxAge, config_.nInit);
    }
    tracker_->Predict();
    tracker_->Update(measurements, features, classIds);

    return tracker_->GetConfirmedTracks();
}
