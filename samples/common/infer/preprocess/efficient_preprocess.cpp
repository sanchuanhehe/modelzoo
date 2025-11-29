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

#include "dev_interface_adapter.h"
#include "efficient_preprocess.h"
#include "log.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <nlohmann/json.hpp>
#include "PillowResize/PillowResize.hpp"

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
constexpr int COLOR = 3;
constexpr int TARGET_IMAGE_SIZE = 288;
constexpr double NORMALIZED = 255.0;
namespace Infer
{
    static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
    {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        if (desc.defaultStride == 0)
        {
            desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
            inBuf.stride = desc.defaultStride;
        }
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++)
        {
            loopTimes *= desc.dims[loop];
        }

        int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = inBuf.stride / dataSize;
        size_t lineSize = width;
        const float *srcData = chwImg.ptr<float>();
        for (int64_t loop = 0; loop < loopTimes; loop++) {
            // 添加循环边界检查
            float *destPtr = static_cast<float *>(inBuf.GetRawPtr()) + loop * strideElemNum;
            const float *srcPtr = srcData + loop * width;
            // 检查指针有效性
            if (destPtr == nullptr || srcPtr == nullptr) {
                LOG(ERROR) << "Null pointer detected";
                return FAILED;
            }
            // 拷贝数据
            memcpy(destPtr, srcPtr, width * dataSize);
        }
        return SUCCESS;
    }

    static cv::Mat HwcToChw(const cv::Mat &hwcImg)
    {
        CV_Assert(!hwcImg.empty() && hwcImg.channels() == COLOR);
        std::vector<cv::Mat> channels;
        cv::split(hwcImg, channels);

        cv::Mat chwMat(COLOR, hwcImg.rows * hwcImg.cols, hwcImg.depth());
        chwMat = cv::Scalar(0);

        for (int channelIdx = 0; channelIdx < COLOR; ++channelIdx) {
            cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
            flatChannel.copyTo(chwMat.row(channelIdx));
        }
        return chwMat; // 返回 3 × (高度*宽度) 的CHW矩阵
    }

    static cv::Mat NormalizeHWC(const cv::Mat &img, float scale, cv::Mat meanMat, cv::Mat stdMat)
    {
        cv::Mat normalized;
        cv::Mat imgFloat;

        // 确保输入图像是浮点类型
        if (img.type() != CV_32FC3) {
            img.convertTo(imgFloat, CV_32FC3);
        } else {
            imgFloat = img.clone();
        }

        // 确保meanMat和stdMat是正确类型和尺寸
        cv::Mat meanFloat, stdFloat;
        meanMat.convertTo(meanFloat, CV_32FC3);
        stdMat.convertTo(stdFloat, CV_32FC3);

        // 先乘以 scale
        cv::multiply(imgFloat, scale, normalized);

        // 使用更安全的方式减去均值
        // 确保meanMat可以正确广播到图像尺寸
        if (meanFloat.size() == cv::Size(1, 1)) {
            // 对于1x1的均值矩阵，使用标量方式
            cv::Scalar meanVal = meanFloat.at<cv::Vec3f>(0, 0);
            normalized -= meanVal;
        } else {
            // 对于其他尺寸，使用矩阵运算
            cv::subtract(normalized, meanFloat, normalized);
        }

        // 除以标准差
        if (stdFloat.size() == cv::Size(1, 1)) {
            cv::Scalar stdVal = stdFloat.at<cv::Vec3f>(0, 0);
            normalized /= stdVal;
        } else {
            cv::divide(normalized, stdFloat, normalized);
        }

        return normalized;
    }

    static cv::Mat NormalizeCHW(const cv::Mat &chwImage, float scale, const std::vector<float> &mean, const std::vector<float> &std)
    {
        // 将 HWC 转换为 CHW
        if (chwImage.channels() != COLOR) {
            LOG(ERROR) << "Expected 3 img channels, but got: " << chwImage.channels();
        }
        // 创建输出矩阵 (与输入同尺寸、同类型)

        cv::Mat normalized;
        chwImage.copyTo(normalized);
        float *data = normalized.ptr<float>();

        // 假设数据布局为 CHW: [channel1_data, channel2_data, channel3_data]
        int channelSize = normalized.total();
        LOG(INFO) << "channelSize " << channelSize;

        for (int c = 0; c < COLOR; ++c) {
            LOG(INFO) << "cc " << c;
            float *channelData = data + c * channelSize;
            float channelMean = mean[c];
            float channelStd = std[c];
            LOG(INFO) << "channelMean " << channelMean;
            LOG(INFO) << "channelStd " << channelStd;
            for (int i = 0; i < channelSize; ++i) {
                channelData[i] = (channelData[i] - channelMean) / channelStd;
            }
        }

        return normalized;
    }

    static cv::Mat ResizeImage(cv::Mat &image, std::vector<int> imgSize)
    {
        LOG(INFO) << "ResizeImage ";
        int target_height = imgSize[0];
        int target_width = imgSize[1];
        cv::Mat resized = PillowResize::resize(image, cv::Size(target_width, target_height), PillowResize::INTERPOLATION_BICUBIC);
        return resized;
    }

    static cv::Mat NormalizeImage(const cv::Mat &img,
                                  float scale = 1.0f / 255.0f,
                                  const std::vector<float> &mean = {0.485f, 0.456f, 0.406f},
                                  const std::vector<float> &std = {0.229f, 0.224f, 0.225f},
                                  const std::string &order = "chw")
    {
        cv::Mat meanMat;
        cv::Mat stdMat;

        LOG(INFO) << "NormalizeImage : ";
        if (order == "chw") {
           
            meanMat = cv::Mat(mean).reshape(1, 3);
            stdMat = cv::Mat(std).reshape(1, 3);
        } else {
            // HWC 格式: (1, 1, 3)
            cv::Mat meanHwc(1, 3, CV_32FC1, const_cast<float *>(mean.data()));
            cv::Mat stdHwc(1, 3, CV_32FC1, const_cast<float *>(std.data()));
            meanMat = meanHwc.reshape(3, 1).clone(); // 转为(1,1,3)
            stdMat = stdHwc.reshape(3, 1).clone();   // 转为(1,1,3)
        }
        cv::Mat result;

        LOG(INFO) << "NormalizeImage convertTo: ";

        if (order == "chw") {
            result = NormalizeCHW(img, scale, mean, std);
        } else {
            LOG(INFO) << "NormalizeHWC ";
            result = NormalizeHWC(img, scale, meanMat, stdMat);
        }

        return result;
    }

    static cv::Mat ResizeKeepRatio(const cv::Mat &img, int targetSize, int interpolation = PillowResize::INTERPOLATION_BICUBIC)
    {
        if (targetSize <= 0) {
            return img;
        }
        int width = img.cols;  // 图像宽度
        int height = img.rows; // 图像高度
        // 检查是否已满足目标尺寸条件
        if ((width <= height && width == targetSize) || (height <= width && height == targetSize)) {
            return img;
        }

        cv::Mat targetImg;
        if (width < height) {
            int newWidth = targetSize;
            int newHeight = static_cast<int>(height * static_cast<float>(targetSize) / width);
            targetImg = PillowResize::resize(img, cv::Size(newWidth, newHeight), interpolation);
        } else {
            int newHeight = targetSize;
            int newWidth = static_cast<int>(width * static_cast<float>(targetSize) / height);
            targetImg = PillowResize::resize(img, cv::Size(newWidth, newHeight), interpolation);
        }

        return targetImg;
    }

    static cv::Mat CenterCrop(const cv::Mat &img, int cropSize)
    {
        if (cropSize <= 0) {
            return img;
        }
        int width = img.cols;
        int height = img.rows;

        // 计算裁剪区域
        int startX = std::max(0, (width - cropSize) / 2);
        int startY = std::max(0, (height - cropSize) / 2);
        int cropWidth = std::min(cropSize, width - startX);
        int cropHeight = std::min(cropSize, height - startY);

        return img(cv::Rect(startX, startY, cropWidth, cropHeight)).clone();
    }

    bool EfficientNetPreprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs)
    {
        // 进度显示（简化版）
        LOG(INFO) << "Processing " << fileList.size();
        // 处理每个图像
        std::vector<int> imgSize = {TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE};
        for (size_t i = 0; i < fileList.size(); ++i) {

            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;

            cv::Mat im0 = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
            cv::Mat imgRgb;
            cv::cvtColor(im0, imgRgb, cv::COLOR_BGR2RGB);
            cv::Mat resizeImage = ResizeKeepRatio(imgRgb, TARGET_IMAGE_SIZE);
            cv::Mat cropImage = CenterCrop(resizeImage, TARGET_IMAGE_SIZE);

            cv::Mat chwImg = HwcToChw(cropImage);
            cv::Mat doubleTemp;
            chwImg.convertTo(doubleTemp, CV_64FC3); // 先转为double
            doubleTemp /= NORMALIZED;                    // double精度除法
            doubleTemp.convertTo(chwImg, CV_32FC3); // 截断为float32

            cv::Mat chw = chwImg.reshape(COLOR, TARGET_IMAGE_SIZE);
            cv::Mat normalized = NormalizeImage(chw);
            ReadImgFileToBuf(normalized, tensorDescs[i], tensorBufs[i]);
            LOG(INFO) << "read end: " << imgPath;
        }

        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}