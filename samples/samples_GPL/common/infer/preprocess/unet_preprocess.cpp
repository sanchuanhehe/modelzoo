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

#include "image_process.h"
#include "unet_preprocess.h"
#include "log.h"
#include "dev_interface_adapter.h"
#include <opencv2/opencv.hpp>
#include "PillowResize/PillowResize.hpp"
#include <string>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <fstream>

namespace Infer
{
    constexpr int BYTE_BIT_NUM = 8;
    constexpr double NORMALIZED = 255.0;

    static cv::Mat PillowResizeNoRatio(const cv::Mat &img, int targetSize, int32_t interpolation = PillowResize::INTERPOLATION_BILINEAR)
    {
        if (targetSize <= 0) {
            return img;
        }
        int width = img.cols;  // 图像宽度
        int height = img.rows; // 图像高度
        cv::Mat targetImg = PillowResize::resize(img, cv::Size(targetSize, targetSize), interpolation);
        return targetImg;
    }

    static cv::Mat hwcToChw(const cv::Mat &hwcImg)
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

        return chwMat; // 返回 3 × (高度*宽度) 的CHW矩阵
    }

    static cv::Mat ProcessSingleImage(const std::string &imgPath, const ImageprocessOptions &options)
    {
        cv::Mat rgbMat;
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
        cv::cvtColor(img, rgbMat, cv::COLOR_BGR2RGB);
        cv::Mat image = PillowResizeNoRatio(rgbMat, options.resizeValue);
        
        image.convertTo(image, CV_32FC3);
        cv::Mat doubleTemp;
        image.convertTo(doubleTemp, CV_64FC3); // 先转为double
        doubleTemp /= NORMALIZED;                    // double精度除法
        doubleTemp.convertTo(image, CV_32FC3); // 截断为float32
        image = hwcToChw(image);
        return image;                    // 确保内存连续
    }

    static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
    {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        if (desc.defaultStride == 0) {
            desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
            inBuf.stride = desc.defaultStride;
        }
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = inBuf.stride / dataSize;
        size_t lineSize = width;

        const float *srcData = chwImg.ptr<float>();

        LOG(INFO) << "ReadImgFileToBuf: loopTimes " << loopTimes << " linesize: " << lineSize << " sizeof(float): " << sizeof(float);

        for (int64_t loop = 0; loop < loopTimes; loop++) {
            // 添加循环边界检查

            float *destPtr = static_cast<float *>(inBuf.GetRawPtr()) + loop * strideElemNum;
            // const float *srcPtr = chwData.data() + loop * width;
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

    bool UnetPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs, const ImageprocessOptions &options)
    {
        LOG(INFO) << "start UnetPreprocess ";
        for (size_t i = 0; i < fileList.size(); ++i) {
            cv::Mat image = ProcessSingleImage(fileList[i], options);
            if (image.empty()) {
                LOG(ERROR) << "fail to preprocess " << fileList[i];
                return false;
            }
            
            size_t imageSize = image.total() * image.elemSize();
            if (tensorBufs[i].size != imageSize) {
                tensorBufs[i] = TensorBuf(imageSize, 0);
            }
            if (!image.isContinuous()) {
                image = image.clone();
            }

            if (ReadImgFileToBuf(image, tensorDescs[i], tensorBufs[i])) {
                LOG(ERROR) << "fail to memcpy";
                return false;
            }
        }
        return true;
    }
}