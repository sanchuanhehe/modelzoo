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

#include "hrnet_preprocess.h"
#include "log.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "utils.h"

namespace Infer
{
// COCO图像归一化参数（全局常量）
const std::vector<float> kMean = {0.485f, 0.456f, 0.406f};
const std::vector<float> kStd = {0.229f, 0.224f, 0.225f};

void GetMultiScaleSize(const cv::Mat& image, int inputSize,
                      cv::Size& resizedSize, cv::Point2f& center, std::pair<float, float>& scale) {
    int h = image.rows;
    int w = image.cols;

    center.x = static_cast<float>(cvRound(w / 2.0 + 0.5));
    center.y = static_cast<float>(cvRound(h / 2.0 + 0.5));

    int baseShortSize = static_cast<int>((inputSize + 63) / 64 * 64);

    int wResized, hResized;
    float scaleW, scaleH;

    if (w < h) {
        wResized = baseShortSize;
        hResized = static_cast<int>((static_cast<float>(baseShortSize) / w * h + 63)) / 64 * 64;
        scaleW = w / 200.0f;
        scaleH = static_cast<float>(hResized) / wResized * w / 200.0f;
    } else {
        hResized = baseShortSize;
        wResized = static_cast<int>((static_cast<float>(baseShortSize) / h * w + 63) / 64) * 64; 
        scaleH = h / 200.0f;
        scaleW = static_cast<float>(wResized) / hResized * h / 200.0f;
    }

    resizedSize = cv::Size(wResized, hResized);
    scale = std::make_pair(scaleW, scaleH);
}

static cv::Mat GetAffineMatrix(const cv::Point2f& center, float scaleW, float scaleH, const cv::Size& outputSize) {
    float dstW = outputSize.width;
    float dstH = outputSize.height;
    float srcW = scaleW * 200.0f;
    float srcH = scaleH * 200.0f;

    return (cv::Mat_<float>(2, 3) << 
        dstW / srcW, 0, dstW * (-center.x / srcW + 0.5f),
        0, dstH / srcH, dstH * (-center.y / srcH + 0.5f)
    );
}

static cv::Mat HwcToChw(const cv::Mat& hwcImg)
{
    CV_Assert(!hwcImg.empty() && hwcImg.channels() == 3);
    std::vector<cv::Mat> channels;
    cv::split(hwcImg, channels);

    cv::Mat chwMat(3, hwcImg.rows * hwcImg.cols, hwcImg.depth());
    chwMat = cv::Scalar(0);

    for (int channelIdx = 0; channelIdx < 3; ++channelIdx) {
        cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
        flatChannel.copyTo(chwMat.row(channelIdx));
    }

    return chwMat; // 返回 3 × (高度*宽度) 的C * HW矩阵
}

static cv::Mat NormalizeCHW(const cv::Mat &chwImage, const std::vector<float> &mean, const std::vector<float> &std)
{
    if (chwImage.channels() != 3) {
        LOG(ERROR) << "Expected 3 img channels, but got: " << chwImage.channels();
    }
    cv::Mat normalized;
    chwImage.copyTo(normalized);
    float *data = normalized.ptr<float>();

    // 假设数据布局为 CHW: [channel1_data, channel2_data, channel3_data]
    int channelSize = normalized.total();

    for (int c = 0; c < chwImage.channels() ; ++c) {
        float *channelData = data + c * channelSize;
        float channelMean = mean[c];
        float channelStd = std[c];
        for (int i = 0; i < channelSize; ++i) {
            channelData[i] = (channelData[i] - channelMean) / channelStd;
        }
    }

    return normalized;
}

cv::Mat Preprocess(const std::string& imagePath, int inputSize = 512) {
    // 1. 读取图像
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("无法读取图像: " + imagePath);
    }

    cv::Size resizedSize;
    cv::Point2f center;
    std::pair<float, float> scale;
    GetMultiScaleSize(image, inputSize, resizedSize, center, scale);
    float scaleW = scale.first;
    float scaleH = scale.second;

    // 3. 转换为RGB色彩空间
    cv::Mat imageRgb;
    cv::cvtColor(image, imageRgb, cv::COLOR_BGR2RGB);

    // 4. 应用仿射变换
    cv::Mat trans = GetAffineMatrix(center, scaleW, scaleH, resizedSize);
    cv::Mat imageResized;
    cv::warpAffine(
        imageRgb, 
        imageResized, 
        trans, 
        resizedSize
    );

    cv::Mat chwImg = HwcToChw(imageResized);

    chwImg.convertTo(chwImg, CV_32F, 1.0 / 255.0);

    chwImg = chwImg.reshape(3, resizedSize.height);

    chwImg = NormalizeCHW(chwImg, kMean, kStd);

    return chwImg;
}

bool HRNetPreprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs)
{
    for (size_t i = 0; i < fileList.size(); ++i) {
        cv::Mat image = Preprocess(fileList[i]);
        if (image.empty()) {
            LOG(ERROR) << "fail to preprocess " << fileList[i];
            return false;
        }
        size_t imageSize = image.total() * image.elemSize();
        if (!image.isContinuous()) {
            image = image.clone();
        }  

        if (!PadDataToTensorBuf(image.data, imageSize, tensorDescs[i], tensorBufs[i])) {
            LOG(ERROR) << "fail to pad data";
            return false;
        }
    }
    return true;
}
}