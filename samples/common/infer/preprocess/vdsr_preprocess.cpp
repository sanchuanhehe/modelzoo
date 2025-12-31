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
#include "imresize_cpp.h"
#include "infer_utils.h"
#include "log.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <unordered_map>

namespace Infer {
constexpr int INPUT_SIZE = 516;
constexpr float MAX_PIXEL_VALUE = 255.0f;
cv::Mat PadImage(const cv::Mat& image, const std::vector<int>& pad)
{
    // 提取填充值
    int padTop = pad[0];
    int padBottom = pad[1];
    int padLeft = pad[2];
    int padRight = pad[3];

    cv::Mat paddedImage;

    // 使用反射101边界进行填充（类似于镜像填充）
    cv::copyMakeBorder(
        image,
        paddedImage,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_REFLECT_101);

    return paddedImage;
}

static Infer::Result ReadImgFileToBuf(const std::string& fileName, Infer::TensorDesc desc, Infer::TensorBuf inBuf)
{
    // 使用OpenCV无损读取PNG文件
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    if (img.empty()) {
        LOG(ERROR) << "Failed to read PGM file: " << fileName;
        return Infer::FAILED;
    }

    int r = stoi(SplitStringByChar(fileName, '/')[SplitStringByChar(fileName, '/').size() - 3].substr(1, 1));

    std::vector<int> preprocessSize = { INPUT_SIZE, INPUT_SIZE };
    cv::Size targetSize(
        static_cast<int>(preprocessSize[0] / static_cast<double>(r)),
        static_cast<int>(preprocessSize[1] / static_cast<double>(r)));

    int lrOriginImgH = img.rows;
    int lrOriginImgW = img.cols;

    // 计算填充
    int padTop = (targetSize.height - lrOriginImgH) / 2;
    int padBottom = targetSize.height - lrOriginImgH - padTop;
    int padLeft = (targetSize.width - lrOriginImgW) / 2;
    int padRight = targetSize.width - lrOriginImgW - padLeft;

    std::vector<int> pad = { padTop, padBottom, padLeft, padRight };
    cv::Mat padImg = PadImage(img, pad);
    padImg = MatlabImresize(padImg, r);

    cv::Mat floatImg;
    padImg.convertTo(floatImg, CV_32F, 1.0 / MAX_PIXEL_VALUE);

    // 将BGR转换为YCbCr颜色空间
    cv::Mat yCbCrImage;
    yCbCrImage = MatlabBgr2ycbcr(floatImg);

    // 如果只需要Y通道，返回单通道图像
    std::vector<cv::Mat> channels;
    cv::split(yCbCrImage, channels);
    cv::Mat yChannelImg = channels[0]; // Y通道

    // 确保数据是连续的
    if (!yChannelImg.isContinuous()) {
        yChannelImg = yChannelImg.clone();
    }
    floatImg = yChannelImg;

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

bool VDSRPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
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