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

#include "yolo3_preprocess.h"
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

        for (int64_t loop = 0; loop < loopTimes; loop++)
        {
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
            if (width * dataSize > inBuf.size)
            {
                LOG(ERROR) << "out of size";
                return FAILED;
            }
            memcpy(destPtr, srcPtr, width * dataSize);
        }
        return SUCCESS;
    }

// Letterbox函数实现
static cv::Mat letterbox(const cv::Mat &img, const cv::Size &target_size, bool scaleup)
    {
        cv::Size shape = img.size();

        // 计算缩放比例
        double r = std::min(static_cast<double>(target_size.width) / shape.width,
                            static_cast<double>(target_size.height) / shape.height);

        LOG(INFO) << "read r: " << r;
        // 计算未填充的尺寸
        cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
                       static_cast<int>(std::round(shape.height * r)));

        LOG(INFO) << "unpad r: " << unpad.width << "  " << unpad.height;
        // 计算填充量
        double w = (target_size.width - unpad.width) / 2.0f;
        double h = (target_size.height - unpad.height) / 2.0f;
        LOG(INFO) << "w r: " << w << "  " << h;
        cv::Mat resized;
        if (shape != unpad) {
            cv::resize(img, resized, unpad, 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }

        // 计算填充边界
        int top = static_cast<int>(std::round(h - 0.1));
        int bottom = static_cast<int>(std::round(h + 0.1));
        int left = static_cast<int>(std::round(w - 0.1));
        int right = static_cast<int>(std::round(w + 0.1));

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        return padded;
    }

static cv::Mat hwcToChw(const cv::Mat &hwcImg)
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

  static  Result ReadImgToBufDlite(const cv::Mat &mat, const TensorDesc &desc,
                             TensorBuf &inBuf)
    {
        cv::Mat chwImg_f16;
        if (mat.type() == CV_32F)
        {
            mat.convertTo(chwImg_f16, CV_16F);
        }
        else
        {
            // 如果不是 32 位浮点，先转换到 32 位浮点再转 16 位
            cv::Mat temp;
            mat.convertTo(temp, CV_32F);
            temp.convertTo(chwImg_f16, CV_16F);
        }
        size_t matTotalBytes = chwImg_f16.total() * chwImg_f16.elemSize();

        memcpy(inBuf.GetRawPtr(), chwImg_f16.data, matTotalBytes);

        return SUCCESS;
    }

bool Yolo3Preprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs, bool isDpico)
    {
        // 进度显示（简化版）
        LOG(INFO) << "Processing " << fileList.size();
        // 处理每个图像
        std::vector<int> img_size = {640, 640};
        for (size_t i = 0; i < fileList.size(); ++i)
        {

            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;
            cv::Mat im0 = cv::imread(imgPath);

            // 获取原始尺寸
            int h0 = im0.rows;
            int w0 = im0.cols;
            // 应用letterbox
            cv::Mat processed = letterbox(im0, cv::Size(img_size[0], img_size[1]), false);

            // BGR到RGB：反转通道顺序
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

            cv::Mat chwImg = hwcToChw(processed);

            // 假设processed是需要处理的cv::Mat
            if (!chwImg.isContinuous())
            {
                chwImg = chwImg.clone(); // 克隆矩阵以获得连续内存
            }
            // 转换为浮点数并归一化，使用更精确的方式
            cv::Mat double_temp;
            chwImg.convertTo(double_temp, CV_64FC3); // 先转为double
            double_temp /= 255.0;                    // double精度除法
            double_temp.convertTo(chwImg, CV_32FC3); // 截断为float32
            if(isDpico) {
                ReadImgFileToBuf(chwImg, tensorDescs[i], tensorBufs[i]);
            } else {
                ReadImgToBufDlite(chwImg, tensorDescs[i], tensorBufs[i]);
            }
            LOG(INFO) << "read end: " << imgPath;
        }

        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}