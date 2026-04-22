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

#include "deepsort_preprocess.h"

namespace Infer {
namespace DeepSort {

const int YOLO_INPUT_SIZE = 640;
const int REID_INPUT_HEIGHT = 128;  // ReID 网络输入高度
const int REID_INPUT_WIDTH = 64;    // ReID 网络输入宽度
const float PIXEL_MAX_VALUE = 255.0f;  // 像素值归一化除数
const int LETTERBOX_PAD_VALUE = 114;   // YOLOv5 letterbox 边缘填充灰度值

int PreProcessYOLO(const cv::Mat &frame, TensorBuf &inBuf,
                   const TensorDesc &desc)
{
    // 1. 等比例缩放并保持宽高比（Letterbox 补边）
    float scale = std::min((float)YOLO_INPUT_SIZE / frame.cols,
                           (float)YOLO_INPUT_SIZE / frame.rows);
    int newW = std::round(frame.cols * scale);
    int newH = std::round(frame.rows * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    cv::Mat canvas = cv::Mat::zeros(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, CV_8UC3);
    canvas.setTo(cv::Scalar(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE));

    int xOffset = (YOLO_INPUT_SIZE - newW) / 2;
    int yOffset = (YOLO_INPUT_SIZE - newH) / 2;
    resized.copyTo(canvas(cv::Rect(xOffset, yOffset, newW, newH)));

    // 2. 将 BGR 转换为 RGB
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    // 3. 归一化并将像素排布由 HWC 平铺为 CHW 的一维连续浮点数组
    float *inData = static_cast<float *>(inBuf.GetRawPtr());
    int channelOffset = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;  // 单通道像素数

    for (int y = 0; y < YOLO_INPUT_SIZE; y++) {
        for (int x = 0; x < YOLO_INPUT_SIZE; x++) {
            cv::Vec3b pixel = canvas.at<cv::Vec3b>(y, x);
            inData[0 * channelOffset + y * YOLO_INPUT_SIZE + x] = pixel[0] / PIXEL_MAX_VALUE;
            inData[1 * channelOffset + y * YOLO_INPUT_SIZE + x] = pixel[1] / PIXEL_MAX_VALUE;
            inData[2 * channelOffset + y * YOLO_INPUT_SIZE + x] = pixel[2] / PIXEL_MAX_VALUE;
        }
    }

    return Infer::SUCCESS;
}

int PreProcessResnetCrop(const cv::Mat &crop, TensorBuf &inBuf,
                         const TensorDesc &desc)
{
    // 1. 缩放到 (64, 128) -> 即宽=64，高=128
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(REID_INPUT_WIDTH, REID_INPUT_HEIGHT));

    // 2. 将 BGR 转换为 RGB（模型训练时使用 PIL/ImageFolder 加载图片，期望 RGB 输入）
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // 3. 归一化并将像素排布由 HWC 平铺为 CHW
    // ImageNet 数据集上统计的 RGB 通道均值和标准差，用于 ResNet 的标准归一化
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdDev[3] = {0.229f, 0.224f, 0.225f};

    float *inData = static_cast<float *>(inBuf.GetRawPtr());
    int channelOffset = REID_INPUT_HEIGHT * REID_INPUT_WIDTH;  // 单通道像素数

    for (int y = 0; y < REID_INPUT_HEIGHT; y++) {
        for (int x = 0; x < REID_INPUT_WIDTH; x++) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);
            inData[0 * channelOffset + y * REID_INPUT_WIDTH + x] =
                ((pixel[0] / PIXEL_MAX_VALUE) - mean[0]) / stdDev[0];
            inData[1 * channelOffset + y * REID_INPUT_WIDTH + x] =
                ((pixel[1] / PIXEL_MAX_VALUE) - mean[1]) / stdDev[1];
            inData[2 * channelOffset + y * REID_INPUT_WIDTH + x] =
                ((pixel[2] / PIXEL_MAX_VALUE) - mean[2]) / stdDev[2];
        }
    }
    return Infer::SUCCESS;
}

} // namespace DeepSort
} // namespace Infer
