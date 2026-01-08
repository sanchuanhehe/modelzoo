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

#include "ocr_det_preprocess.h"
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

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
namespace Infer
{
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
            float *destPtr = static_cast<float *>(inBuf.GetRawPtr()) + loop * strideElemNum;
            const float *srcPtr = srcData + loop * width;
            if (destPtr == nullptr || srcPtr == nullptr) {
                LOG(ERROR) << "Null pointer detected";
                return FAILED;
            }
            memcpy(destPtr, srcPtr, width * dataSize);
        }
        return SUCCESS;
    }

    static cv::Mat ResizeAndPadding(const cv::Mat &img, const cv::Size &targetSize, bool scaleup)
    {
        cv::Size shape = img.size();
        double r = std::min(static_cast<double>(targetSize.width) / shape.width,
                            static_cast<double>(targetSize.height) / shape.height);
        size_t resizeW = static_cast<int>(std::round(shape.width * r));
        size_t resizeH = static_cast<int>(std::round(shape.height * r));
        resizeW = std::max(static_cast<int>(std::round(resizeW / 32.0)) * 32, 32);
        resizeH = std::max(static_cast<int>(std::round(resizeH / 32.0)) * 32, 32);
        cv::Size unpad(resizeW, resizeH);
        cv::Mat resized;
        cv::resize(img, resized, unpad, 0, 0, cv::INTER_LINEAR);
        int top = 0;
        int bottom = targetSize.height - resizeH;
        int left = 0;
        int right = targetSize.width - resizeW;
        LOG(INFO) << "copyMakeBorder w h : " << bottom << " " << right << "  " <<  resized.type();
        cv::Mat padded;
        if (resized.empty()) {
            LOG(INFO) << "RESIZE EMPTY : ";
            return resized;
        }
        cv::copyMakeBorder(resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
        return padded;
    }

    static cv::Mat HwcToChw(const cv::Mat &hwcImg)
    {
        CV_Assert(!hwcImg.empty() && hwcImg.channels() == 3);
        std::vector<cv::Mat> channels;
        cv::split(hwcImg, channels);

        cv::Mat chwMat(3, hwcImg.rows * hwcImg.cols, hwcImg.depth());
        chwMat = cv::Scalar(0);

        for (int channelIdx = 0; channelIdx < 3; ++channelIdx)
        {
            cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
            flatChannel.copyTo(chwMat.row(channelIdx));
        }

        return chwMat; // 返回 3 × (高度*宽度) 的CHW矩阵
    }

    
    static cv::Mat NormalizeHWC(const cv::Mat& img, float scale, cv::Mat meanMat, cv::Mat stdMat) {
        // 校验输入图像类型
        if (img.type() != CV_32FC3) {
            LOG(INFO) << "img CV_32FC3类型（3通道32位浮点）";
        }
        // 校验mean/std类型和形状
        if (meanMat.type() != CV_32FC3 || stdMat.type() != CV_32FC3) {
            LOG(INFO) << "meanMat and stdMat ";
        }
        if (meanMat.size() != cv::Size(1, 1) || meanMat.channels() != 3) {
            LOG(INFO) << "meanMat必须是(1,1,3)的3通道矩阵";
        }
           
        // 确保所有矩阵都是相同的类型
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
    
    static cv::Mat NormalizeCHW(const cv::Mat& img, float scale, const std::vector<float>& mean , const std::vector<float>& std) {
        // 将 HWC 转换为 CHW
        std::vector<cv::Mat> channels;
        cv::split(img, channels);
        
        // 对每个通道分别归一化
        for (int i = 0; i < 3; ++i) {
            channels[i].convertTo(channels[i], CV_32F);
            channels[i] = (channels[i] * scale - mean[i]) / std[i];
        }
        
        // 合并通道
        cv::Mat result;
        cv::merge(channels, result);
        
        return result;
    }

    static cv::Mat NormalizeImage(const cv::Mat& img,
                   float scale = 1.0f / 255.0f, 
                   const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
                   const std::vector<float>& std = {0.229f, 0.224f, 0.225f},
                   const std::string& order = "hwc") {
        cv::Mat meanMat;
        cv::Mat stdMat;
        LOG(INFO) << "normalizeImage";
        if (order == "chw") {
            // CHW 格式: (3, 1, 1)
            meanMat = cv::Mat(3, 1, CV_32FC1, const_cast<float*>(mean.data())).clone();
            stdMat = cv::Mat(3, 1, CV_32FC1, const_cast<float*>(std.data())).clone();
           
        } else {
            LOG(INFO) << "normalizeImage hwc: ";
            // HWC 格式: (1, 1, 3)
            cv::Mat meanHwc(1, 3, CV_32FC1, const_cast<float*>(mean.data()));
            cv::Mat stdHwc(1, 3, CV_32FC1, const_cast<float*>(std.data()));
            meanMat = meanHwc.reshape(3, 1).clone();  // 转为(1,1,3)
            stdMat = stdHwc.reshape(3, 1).clone();    // 转为(1,1,3)
        }
        cv::Mat result;
         // 应用归一化: (img * scale - mean) / std
        cv::Mat imgFloat;
        img.convertTo(imgFloat, CV_32FC3);
        meanMat.convertTo(meanMat, CV_32FC3);
        stdMat.convertTo(stdMat, CV_32FC3);
        if (order == "chw") {
            // CHW 格式需要特殊处理
            result = NormalizeCHW(imgFloat, scale, mean, std);
        } else {
            // HWC 格式可以直接广播
             LOG(INFO) << "NormalizeHWC ";
            result = NormalizeHWC(imgFloat, scale, meanMat, stdMat);
        }
        
        return result;
    }

    bool OcrDetPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
    {
        // 处理每个图像
        std::vector<int> img_size = {960, 960};
        for (size_t i = 0; i < fileList.size(); ++i)
        {

            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;
            cv::Mat im0 = cv::imread(imgPath);

            // 获取原始尺寸
            int h0 = im0.rows;
            int w0 = im0.cols;
            // 应用letterbox
            cv::Mat processed = ResizeAndPadding(im0, cv::Size(img_size[0], img_size[1]), false);
            cv::Mat normalized = NormalizeImage(processed);
            cv::Mat chwImg = HwcToChw(normalized);
            
            // 假设processed是需要处理的cv::Mat
            if (!chwImg.isContinuous())
            {
                chwImg = chwImg.clone(); // 克隆矩阵以获得连续内存
            }
            ReadImgFileToBuf(chwImg, inDescs[i], inBufs[i]);
            LOG(INFO) << "read end: " << imgPath;
        }

        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}