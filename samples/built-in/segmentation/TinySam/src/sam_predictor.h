/*
 * Copyright (c) ModelZoo. 2025-2025. All rights reserved.
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

#include <memory>
#include <string>
#include <iostream>
#include <vector>
#include <utility>
#include "opencv2/opencv.hpp"
#include "dev_interface_adapter.h"
#include "model.h"

class SAMBoxPredictor {
public:
    SAMBoxPredictor() : imageEncoder_(std::make_unique<Infer::Model>()),
        promptEncoder_(std::make_unique<Infer::Model>()), maskDecoder_(std::make_unique<Infer::Model>()) {}

    bool loadModel(const std::vector<std::string>& ModelPaths);

    bool setImage(const cv::Mat& imgIn);

    std::pair<float, cv::Mat> predict(const std::vector<float>& boxes);

private:
    std::pair<int, int> imageSize_ = {448, 448}; // 448 模型输入尺寸 (H, W)
    std::vector<Infer::TensorBuf> maskInputs_;
    std::vector<Infer::TensorBuf> promptInputs_;
    std::vector<Infer::TensorBuf> imageInputs_;
    float scale_ = 0.0f;                        // 图像缩放比例
    std::pair<int, int> originImgShape_;       // 原始图像尺寸 (H, W)
    std::pair<int, int> targetImgShape_;    // 缩放后图像尺寸 (H, W)
    float scaleH_ = 0.0f;                      // 高度缩放因子（特征图/原始图）
    float scaleW_ = 0.0f;                      // 宽度缩放因子（特征图/原始图）
    std::unique_ptr<Infer::Model> imageEncoder_;
    std::unique_ptr<Infer::Model> promptEncoder_;
    std::unique_ptr<Infer::Model> maskDecoder_;
};