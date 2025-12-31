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

#include "pfld_preprocess.h"
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
namespace PFLDNS {

// 常量定义
constexpr int DEFAULT_TARGET_WIDTH = 112;
constexpr int DEFAULT_TARGET_HEIGHT = 112;
constexpr int BATCH_SIZE = 1;
constexpr int CHANNEL_NUM = 3;
constexpr float PIXEL_MAX_VALUE = 255.0f;
constexpr int BYTE_BIT_NUM = 8;

/**
 * @brief 图像预处理（缩放+归一化）
 * @param img 输入图像
 * @param targetWidth 目标宽度
 * @param targetHeight 目标高度
 * @return 预处理后的图像
 */
static cv::Mat PreprocessImage(const cv::Mat& img, int targetWidth, int targetHeight) {
    // 调整尺寸
    cv::Mat imgResized;
    cv::resize(img, imgResized, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_LINEAR);

    // 归一化到[0, 1]
    cv::Mat imgNorm;
    imgResized.convertTo(imgNorm, CV_32FC3, 1.0f / PIXEL_MAX_VALUE);

    return imgNorm;
}

/**
 * @brief 转换通道格式（HWC->CHW）
 * @param hwcImg HWC格式图像
 * @return CHW格式图像
 */
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

/**
 * @brief 添加批次维度（CHW->NCHW）
 * @param chwImg CHW格式图像
 * @return NCHW格式图像（批次大小为1）
 */
static cv::Mat AddBatchDimension(const cv::Mat& chwImg) {
    CV_Assert(!chwImg.empty() && chwImg.rows == CHANNEL_NUM);
    int height = chwImg.cols / DEFAULT_TARGET_WIDTH;
    int width = DEFAULT_TARGET_WIDTH;

    cv::Mat nchwMat(BATCH_SIZE, CHANNEL_NUM * height * width, CV_32F);
    memcpy(nchwMat.data, chwImg.data, CHANNEL_NUM * height * width * sizeof(float));
    return nchwMat;
}

/**
 * @brief 将图像数据写入张量缓冲区
 * @param nchwImg NCHW格式图像
 * @param desc 张量描述符
 * @param inBuf 输入张量缓冲区
 * @return 处理结果
 */
static Result WriteImageToTensorBuf(const cv::Mat& nchwImg, TensorDesc& desc, TensorBuf& inBuf) {
    LOG(INFO) << "WriteImageToTensorBuf: desc.dimCount " << desc.dimCount;
    const float* srcData = nchwImg.ptr<float>();

    // for dlite
    if (desc.defaultStride == 0) {
        // 计算数据量
        size_t dataCount = nchwImg.total();

        // 转换为FP16
        std::vector<uint16_t> fp16Data(dataCount);
        for (size_t i = 0; i < dataCount; ++i) {
            fp16Data[i] = FloatToHalf(srcData[i]);
        }

        memcpy(static_cast<void*>(inBuf.GetRawPtr()), fp16Data.data(), desc.defaultSize);
        return SUCCESS;
    }

    // for dpico
    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    int64_t width = desc.dims[desc.dimCount - 1];
    size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
    size_t strideElemNum = inBuf.stride / dataSize;

    for (int64_t loop = 0; loop < loopTimes; loop++) {
        float* destPtr = static_cast<float*>(inBuf.GetRawPtr()) + loop * strideElemNum;
        const float* srcPtr = srcData + loop * width;

        if (destPtr == nullptr || srcPtr == nullptr) {
            LOG(ERROR) << "Null pointer detected";
            return FAILED;
        }

        memcpy(destPtr, srcPtr, width * dataSize);
    }
    return SUCCESS;
}

/**
 * @brief PFLD模型前处理主函数
 * @param fileList 图像文件路径列表
 * @param tensorBufs 输出张量缓冲区列表
 * @param tensorDescs 张量描述符列表
 * @return 处理是否成功
 */
bool PFLDPreprocess(std::vector<std::string>& fileList,
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
        std::string fileName = imgPath.substr(start + 1, end - start - 1);

        // 读取图像
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            LOG(ERROR) << "Read image failed: " << imgPath;
            continue;
        }

        // 预处理图像
        cv::Mat processedImg = PreprocessImage(img, DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT);

        // 转换通道格式
        cv::Mat chwImg = HwcToChw(processedImg);

        // 添加批次维度
        cv::Mat nchwImg = AddBatchDimension(chwImg);

        if (!nchwImg.isContinuous()) {
            nchwImg = nchwImg.clone();
        }

        // 填充张量缓冲区
        if (WriteImageToTensorBuf(nchwImg, tensorDescs[i], tensorBufs[i]) != SUCCESS) {
            LOG(ERROR) << "Failed to fill tensor buffer for: " << imgPath;
            continue;
        }

        // 保存预处理结果
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + fileName + ".bin";
            std::ofstream binFile(binPath, std::ios::binary);
            if (binFile.is_open()) {
                size_t dataSize = BATCH_SIZE * CHANNEL_NUM * DEFAULT_TARGET_HEIGHT * DEFAULT_TARGET_WIDTH * sizeof(float);
                binFile.write(reinterpret_cast<char*>(nchwImg.data), dataSize);
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

}  // namespace PfldNS
}  // namespace Infer