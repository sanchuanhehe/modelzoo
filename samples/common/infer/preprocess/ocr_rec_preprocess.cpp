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

#include "ocr_rec_preprocess.h"
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
static bool isDpicoGlobal = true;
namespace Infer
{
    static Result ReadImgFileToBufDpico(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
    {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }
        int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = inBuf.stride / dataSize;
        size_t lineSize = width;

        // 获取数据指针
        const float *srcData = chwImg.ptr<float>();

        LOG(INFO) << "ReadImgFileToBuf: loopTimes " << loopTimes << " linesize: " << lineSize << " sizeof(float): " << sizeof(float);
       
        for (int64_t loop = 0; loop < loopTimes; loop++) {
            // 添加循环边界检查
            float *destPtr = static_cast<float *>(inBuf.GetRawPtr()) + loop * strideElemNum;
            // const float *srcPtr = chwData.data() + loop * width;
            const float *srcPtr = srcData + loop * width;
            // 检查指针有效性
            if (destPtr == nullptr || srcPtr == nullptr)
            {
                LOG(ERROR) << "Null pointer detected";
                return FAILED;
            }
            // 拷贝数据
            memcpy(destPtr, srcPtr, width * dataSize);
        }
        return SUCCESS;
    }

    static Result ReadImgFileToBufDlite(const cv::Mat& mat, const TensorDesc& desc,
    TensorBuf& inBuf)
    {
        size_t matTotalBytes = mat.total() * mat.elemSize();
        memcpy(inBuf.GetRawPtr(), mat.data, matTotalBytes);
        return SUCCESS;
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

    static cv::Mat ResizeNormlImgOptimized(const cv::Mat &img)
    {
        int imgC = 3;
        int imgH = 48;
        int imgW = 320;
        // 检查图像是否成功加载
        if (img.empty()) {
            // std::cerr << "错误: 无法加载图像: " << imgPath << std::endl;
            LOG(ERROR) << "img is empty ";
            cv::Mat output = cv::Mat::zeros(imgC, imgH * imgW, CV_32F);
            return output;
        }
        cv::Size shape = img.size();

        cv::Mat resized;
        int resizedWidth;

        float ratio = static_cast<float>(img.cols) / img.rows;
        resizedWidth = std::min(imgW, static_cast<int>(std::ceil(imgH * ratio)));
        cv::resize(img, resized, cv::Size(resizedWidth, imgH));
        cv::Mat chwMat = hwcToChw(resized);
        cv::Mat doubleTemp;
        chwMat.convertTo(doubleTemp, CV_64FC3); // 先转为double
        doubleTemp /= 255.0;                    // double精度除法
        doubleTemp.convertTo(chwMat, CV_32FC3); // 截断为float32
        chwMat = (chwMat - 0.5f) / 0.5f;

        cv::Mat output = cv::Mat::zeros(imgC, imgH * imgW, CV_32F);
        cv::Mat padding_im = cv::Mat::zeros(imgC, imgH * imgW, CV_32F);

        float valid_ratio = std::min(1.0f, static_cast<float>(resizedWidth) / imgW);
        // 多通道图像
        std::vector<cv::Mat> channels;
        for (int c = 0; c < imgC; c++) {
            cv::Mat srcChannel = chwMat.row(c);
            cv::Mat dstRoi = padding_im.row(c).colRange(0, resizedWidth);
            cv::Mat dstChannel = padding_im.row(c);
            for (int h = 0; h < imgH; h++) {
                cv::Mat srcRow = srcChannel.colRange(h * resizedWidth, (h + 1) * resizedWidth);
                cv::Mat dstRow = dstChannel.colRange(h * imgW, h * imgW + resizedWidth);
                srcRow.copyTo(dstRow);
            }
        }
        return padding_im;
    }

    static bool fileExists(const std::string &imagePath)
    {
        std::ifstream file(imagePath);
        return file.good();
    }

    bool OcrRecPreprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &inBufs, std::vector<TensorDesc> &inDescs, bool isDpico)
    {
        isDpicoGlobal = isDpico;
        // 进度显示（简化版）
        LOG(INFO) <<  "PreProcessing ";
        for (size_t i = 0; i < fileList.size(); ++i) {
            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;
            cv::Mat im0 = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
            cv::Mat processed = ResizeNormlImgOptimized(im0);
            if (isDpicoGlobal) {
                ReadImgFileToBufDpico(processed, inDescs[i], inBufs[i]);
            } else {
                ReadImgFileToBufDlite(processed, inDescs[i], inBufs[i]);
            }
            LOG(INFO) << "read end: " << imgPath;
        }
        return true;
    }
}