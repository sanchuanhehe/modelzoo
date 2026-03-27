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

#ifndef DEEPSORT_H
#define DEEPSORT_H

#include "deepsort_postprocess.h"
#include "deepsort_preprocess.h"
#include "dev_interface_adapter.h"
#include "tracker.h"
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

struct DeepSORTConfig {
    std::string yoloModelPath;
    std::string resnetModelPath;
    float yoloConfThres = 0.25f;
    float yoloNmsThres = 0.45f;
    float maxCosineDistance = 0.2f;
    int nnBudget = 100;
    float maxIouDistance = 0.7f;
    int maxAge = 70;
    int nInit = 3;
    bool testReidFps = false;
};

class DeepSortController {
public:
    DeepSortController(const DeepSORTConfig &config);
    ~DeepSortController();

    int Init();
    // 输入是从 OpenCV 读取到的 BGR 单帧图像，返回带有追踪 ID 的边框格式数据
    std::vector<Track> ProcessFrame(cv::Mat &frame);

    // DeepSORT 核心追踪逻辑管理器
    std::unique_ptr<Tracker> tracker_;

private:
    DeepSORTConfig config_;

    // NPU 模型句柄
    std::shared_ptr<Infer::MdlBase> yoloModel_;
    std::shared_ptr<Infer::MdlBase> resnetModel_;

    // 持久化预分配的共享 NPU 输入输出显存，防止高频 DevMalloc 导致 NPU 驱动碎片化锁死
    std::vector<Infer::TensorBuf> yoloInBufs_;
    std::vector<Infer::TensorBuf> yoloOutBufs_;
    std::vector<Infer::TensorBuf> resnetInBufs_;
    std::vector<Infer::TensorBuf> resnetOutBufs_;

    // YOLO 推理方法封装
    std::vector<DetectBox> RunYOLO(const cv::Mat &frame);
    // DeepSORT ReID 行人重识别特征提取推理封装
    std::vector<std::vector<float>>
    RunResNet(const cv::Mat &frame, const std::vector<DetectBox> &boxes);

    cv::Rect GetSafeBoxRect(const DetectBox &box, int imgCols, int imgRows);
    std::vector<float> ExtractReIDFeature(const cv::Mat &crop);
};

#endif // DEEPSORT_H
