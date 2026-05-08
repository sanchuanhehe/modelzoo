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

#include "PillowResize/PillowResize.hpp"
#include "log.h"
#include "ocr_det_preprocess.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using json = nlohmann::json;

constexpr int BYTE_BIT_NUM = 8;
constexpr int COLOR_NUM = 3;
constexpr float MEAN[3] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float STD[3] = {0.26862954f, 0.26130258f, 0.27577711f};
constexpr int IMAGE_SIZE = 224;

namespace Infer {
static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc,
                               TensorBuf &inBuf)
{
    LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
    if (desc.defaultStride == 0) {
        desc.defaultStride =
            desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
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
    LOG(INFO) << "ReadImgFileToBuf: loopTimes " << loopTimes
                << " linesize: " << lineSize << " sizeof(float): " << sizeof(float);
    for (int64_t loop = 0; loop < loopTimes; loop++) {
        float *destPtr =
            static_cast<float *>(inBuf.GetRawPtr()) + loop * strideElemNum;
        const float *srcPtr = srcData + loop * width;
        if (destPtr == nullptr || srcPtr == nullptr) {
            LOG(ERROR) << "Null pointer detected";
            return FAILED;
        }
        memcpy(destPtr, srcPtr, width * dataSize);
    }
    return SUCCESS;
}

cv::Mat TensorAndNormalize(const cv::Mat &rgbImg)
{
    int H = rgbImg.rows;
    int W = rgbImg.cols;
    int totalPixels = H * W;
    cv::Mat chwTensor(COLOR_NUM, totalPixels, CV_32FC1);
    float *chwPtr = chwTensor.ptr<float>();
    memset(chwPtr, 0, COLOR_NUM * totalPixels * sizeof(float));
    for (int h = 0; h < H; h++) {
        const uchar *rgbRow = rgbImg.ptr<uchar>(h);
        for (int w = 0; w < W; w++) {
        // 读取RGB像素（0-255）
        uchar r = rgbRow[COLOR_NUM * w + 0];
        uchar g = rgbRow[COLOR_NUM * w + 1];
        uchar b = rgbRow[COLOR_NUM * w + 2];

        // 转float + 除以255
        float rFloat = static_cast<float>(r) / 255.0f;
        float gFloat = static_cast<float>(g) / 255.0f;
        float bFloat = static_cast<float>(b) / 255.0f;

        // Normalize：(x - mean) / std
        rFloat = (rFloat - MEAN[0]) / STD[0];
        gFloat = (gFloat - MEAN[1]) / STD[1];
        bFloat = (bFloat - MEAN[2]) / STD[2];

        // 写入CHW矩阵（C0=R, C1=G, C2=B）
        int idx = h * W + w;
        chwPtr[0 * totalPixels + idx] = rFloat;
        chwPtr[1 * totalPixels + idx] = gFloat;
        chwPtr[2 * totalPixels + idx] = bFloat;
        }
    }

    return chwTensor;
}

bool ClipImgPreprocess(std::vector<std::string> &fileList,
                       std::vector<TensorBuf> &inBufs,
                       std::vector<TensorDesc> &inDescs)
{
    // 处理每个图像
    for (size_t i = 0; i < fileList.size(); ++i) {

        std::string imgPath = fileList[i];
        LOG(INFO) << "imgPath: " << imgPath;
        cv::Mat im0 = cv::imread(imgPath);

        // 应用letterbox
        cv::Mat processed = im0.clone();
        cv::Mat rgbImg;
        cv::cvtColor(processed, rgbImg, cv::COLOR_BGR2RGB);
        // 1. Resize - 使用双三次插值
        cv::Mat resizeImg;
        resizeImg = PillowResize::resize(rgbImg, cv::Size(IMAGE_SIZE, IMAGE_SIZE),
            PillowResize::INTERPOLATION_BICUBIC);
        cv::Mat normalized = TensorAndNormalize(resizeImg);
        if (!normalized.isContinuous()) {
            normalized = normalized.clone(); // 克隆矩阵以获得连续内存
        }
        ReadImgFileToBuf(normalized, inDescs[i], inBufs[i]);
        LOG(INFO) << "read end: " << imgPath;
    }
    LOG(INFO) << "PreProcessing completed successfully!";
    return true;
}

} // namespace Infer