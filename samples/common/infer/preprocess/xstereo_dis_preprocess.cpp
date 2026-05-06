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

#include "xstereo_dis_preprocess.h"
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
#include "utils.h"
#include <nlohmann/json.hpp>
#include "PillowResize/PillowResize.hpp"

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
namespace Infer
{
    static Result ReadTxtToBuf(float val, TensorDesc &desc, TensorBuf &inBuf)
    {
        LOG(INFO) << "ReadTxtToBuf: desc.dimCount " << desc.dimCount;
        if (desc.defaultStride == 0) {
            desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
            inBuf.stride = desc.defaultStride;
        }
        float *destPtr = static_cast<float *>(inBuf.GetRawPtr());
        destPtr[0] = val;
        return SUCCESS;
    }

    static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
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
        const float *srcData = chwImg.ptr<float>();
        LOG(INFO) << "ReadImgFileToBuf: loopTimes " << loopTimes << " linesize: " << lineSize << " sizeof(float): "
            << sizeof(float);
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

    bool XStereoDisPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs,
        std::vector<TensorDesc>& inDescs)
    {
        LOG(INFO) << " xstereo dis Processing ";
        auto cfg = ReadCfgFile("../data/cfg.txt");
        int imgHeight = std::stoi(cfg["img_height"]);
        int imgWidth = std::stoi(cfg["img_width"]);
        float line = std::stof(cfg["line"]);
        float distance = std::stof(cfg["distance"]);
        if(inBufs.size() < 4) {
            LOG(ERROR) << "error processing  buf size " << inBufs.size();
            return false;
        }
        cv::Mat inferenceResult(imgHeight, imgWidth, CV_32FC1, static_cast<float *>(inBufs[3].GetRawPtr()));
        ReadImgFileToBuf(inferenceResult, inDescs[0], inBufs[0]);
        ReadTxtToBuf(line, inDescs[1], inBufs[1]);
        ReadTxtToBuf(distance, inDescs[2], inBufs[2]);

        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}