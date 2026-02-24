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

#include "PillowResize/PillowResize.hpp"
#include "dev_interface_adapter.h"
#include "image_process.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <unordered_map>

namespace Infer {
const double TARGET_HEIGHT_1 = 32.0;
const double TARGET_WIDTH_2 = 160.0;
const double SOURCE_WIDTH_2 = 280.0;
const float MAX_PIXEL_VALUE = 255.0f;
const float NORMALIZATION_MEAN = 0.588f; // 归一化均值
const float NORMALIZATION_STD = 0.193f; // 归一化标准差

static Infer::Result ReadImgFileToBuf(const std::string& fileName, Infer::TensorDesc desc, Infer::TensorBuf inBuf)
{
    cv::Mat image = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        LOG(ERROR) << "Failed to read img file: " << fileName;
        return Infer::FAILED;
    }

    // 第一次缩放
    int height = image.rows;
    // 第一次缩放：将图像高度缩放到32像素
    double scale_factor1 = TARGET_HEIGHT_1 / static_cast<double>(height);
    cv::resize(image, image, cv::Size(0, 0), scale_factor1, scale_factor1, cv::INTER_CUBIC);

    // 第二次缩放：将图像宽度缩放到160像素（保持高度不变）
    double scale_factor2 = TARGET_WIDTH_2 / SOURCE_WIDTH_2;
    cv::resize(image, image, cv::Size(0, 0), scale_factor2, 1.0, cv::INTER_CUBIC);

    // 转换为浮点数并归一化
    cv::Mat floatImg;
    image.convertTo(floatImg, CV_32F);

    // 使用精确的数值运算避免精度误差
    // 公式：(像素值 / 255 - 均值) / 标准差
    floatImg = (floatImg / MAX_PIXEL_VALUE - NORMALIZATION_MEAN) / NORMALIZATION_STD;

    // 确保数据是连续的
    if (!floatImg.isContinuous()) {
        floatImg = floatImg.clone();
    }

    // 将数据复制到TensorBuf
    if (inBuf.stride == 0) {
        // 直接拷贝整个数据块
        size_t dataSize = floatImg.total() * floatImg.elemSize();
        memcpy(inBuf.GetRawPtr(), floatImg.data, dataSize);
    } else {
        // 按stride逐行拷贝
        size_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        size_t width = desc.dims[desc.dimCount - 1];
        size_t lineSize = width * sizeof(float);

        const float* srcPtr = reinterpret_cast<const float*>(floatImg.data);
        char* dstPtr = static_cast<char*>(inBuf.GetRawPtr());

        for (size_t loop = 0; loop < loopTimes; loop++) {
            memcpy(dstPtr + loop * inBuf.stride, srcPtr + loop * width, lineSize);
        }
    }

    return Infer::SUCCESS;
}

bool CRNNPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    int ret = SUCCESS;
    for (size_t i = 0; i < fileList.size(); ++i) {
        /* read img to inBuf */
        ret = ReadImgFileToBuf(fileList[i], tensorDescs[i], tensorBufs[i]);
        if (ret != SUCCESS) {
            LOG(ERROR) << "read img file to buf failed";
            return false;
        }
    }
    return true;
}
}