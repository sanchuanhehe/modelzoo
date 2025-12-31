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

#include "crowdcount_preprocess.h"
#include "log.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace Infer {
namespace CrowdCountNS {

constexpr int DEFAULT_TARGET_WIDTH = 224;
constexpr int DEFAULT_TARGET_HEIGHT = 224;
constexpr int BYTE_BIT_NUM = 8;
constexpr float PIXEL_MAX_VALUE = 255.0f;
constexpr int CHANNEL_NUM = 3;
constexpr int BATCH_SIZE = 1;

static cv::Mat PreprocessImage(const cv::Mat& img, int targetWidth, int targetHeight) {
    // 调整尺寸
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

    // 归一化到[0, 1]
    cv::Mat imgNorm;
    imgResized.convertTo(imgNorm, CV_32FC3, 1.0f / PIXEL_MAX_VALUE);

    return imgNorm;
}

static cv::Mat HwcToChw(const cv::Mat& hwcImg) {
    CV_Assert(!hwcImg.empty() && hwcImg.channels() == CHANNEL_NUM);
    std::vector<cv::Mat> channels;
    cv::split(hwcImg, channels);

    cv::Mat chwMat(CHANNEL_NUM, hwcImg.rows * hwcImg.cols, CV_32F);
    chwMat = cv::Scalar(0);

    for (int channelIdx = 0; channelIdx < CHANNEL_NUM; ++channelIdx) {
        cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
        flatChannel.convertTo(flatChannel, CV_32F);
        flatChannel.copyTo(chwMat.row(channelIdx));
    }

    return chwMat;
}

    static Result ReadImageToTensorBuf(const cv::Mat& chwImg, TensorDesc& desc, TensorBuf& inBuf) {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        const float *srcData = chwImg.ptr<float>();
        // dlite核，不需要处理字节对齐，直接读取数据
        if (desc.defaultStride == 0) {
            memcpy(static_cast<float *>(inBuf.GetRawPtr()), srcData, desc.defaultSize);
            return SUCCESS;
        }

        // dpico核，需要处理字节对齐
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = inBuf.stride / dataSize;

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

bool CrowdCountPreprocess(std::vector<std::string>& fileList,
                         std::vector<TensorBuf>& tensorBufs,
                         std::vector<TensorDesc>& tensorDescs) {
    auto cfg = ReadCfgFile("../data/cfg.txt");
    std::string saveBin = cfg["save_preprocess_bin"];

    if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
        LOG(ERROR) << "Create preprocess bin dir failed: " << saveBin;
        return false;
    }

    LOG(INFO) << "Processing " << fileList.size() << " images...";

    for (size_t i = 0; i < fileList.size(); ++i) {
        std::string imgPath = fileList[i];
        LOG(INFO) << "Processing image: " << imgPath;

        // 提取文件名（不含扩展名）
        size_t start = imgPath.find_last_of("/");
        size_t end = imgPath.find_last_of(".");
        std::string outputName = imgPath.substr(start + 1, end - start - 1);

        // 读取图像并转换为RGB
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            LOG(ERROR) << "Read image failed: " << imgPath;
            continue;
        }

        // 预处理图像
        cv::Mat processedImg = PreprocessImage(img, DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT);

        // 转换为CHW格式
        cv::Mat chwImg = HwcToChw(processedImg);

        if (!chwImg.isContinuous()) {
            chwImg = chwImg.clone();
        }

        // 填充tensor缓冲区
        if (ReadImageToTensorBuf(chwImg, tensorDescs[i], tensorBufs[i]) != SUCCESS) {
            LOG(ERROR) << "Failed to fill tensor buffer for: " << imgPath;
            continue;
        }

        // 保存预处理结果
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + outputName + ".bin";
            std::ofstream binFile(binPath, std::ios::binary);
            if (binFile.is_open()) {
                size_t dataSize = DEFAULT_TARGET_WIDTH * DEFAULT_TARGET_HEIGHT * CHANNEL_NUM * sizeof(float);
                binFile.write(reinterpret_cast<char*>(chwImg.data), dataSize);
                binFile.close();
                if (chmod(binPath.c_str(), 0755) != 0) {
                    LOG(WARNING) << "Failed to set permissions for: " << binPath;
                }
                LOG(INFO) << "Saved preprocess bin: " << binPath;
            } else {
                LOG(WARNING) << "Failed to open bin file: " << binPath;
            }
        }
    }

    LOG(INFO) << "PreProcessing completed successfully!";
    return true;
}

}  // namespace CrowdCountNS
}  // namespace Infer