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
#include "log.h"
#include "dev_interface_adapter.h"
#include <opencv2/opencv.hpp>
#include "PillowResize/PillowResize.hpp"
#include <string>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <fstream>
#include "utils.h"

namespace Infer {

constexpr size_t BYTE_BIT_NUM = 8; // 1 byte = 8 bit
constexpr int INPUT_SIZE = 100;
constexpr float NORMAL_VALUE = 255.0f;

static Infer::Result ReadImgFileToBuf(const std::string& fileName, Infer::TensorDesc desc, Infer::TensorBuf inBuf) {
    try {
        // 使用OpenCV读取PGM文件
        cv::Mat img = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            LOG(ERROR) << "Failed to read PGM file: " << fileName;
            return Infer::FAILED;
        }

        // 调整大小为100x100
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_LINEAR);

        // 转换为float32并归一化到[0,1]
        cv::Mat floatImg;
        resized.convertTo(floatImg, CV_32F, 1.0/NORMAL_VALUE);

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
    catch (const cv::Exception& e) {
        LOG(ERROR) << "OpenCV exception: " << e.what();
        return Infer::FAILED;
    }
    catch (const std::exception& e) {
        LOG(ERROR) << "Standard exception: " << e.what();
        return Infer::FAILED;
    }
}

bool SiameseNetworkPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
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