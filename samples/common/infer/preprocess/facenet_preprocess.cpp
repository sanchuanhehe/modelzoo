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

#include "facenet_preprocess.h"
#include "log.h"
#include "utils.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

using json = nlohmann::json;

namespace Infer {
namespace FaceNetNS {

    // 常量定义
    constexpr int TARGET_WIDTH = 160;
    constexpr int TARGET_HEIGHT = 160;
    constexpr int CHANNEL_NUM = 3;
    constexpr int BYTE_BIT_NUM = 8;
    constexpr float PIXEL_MAX_VALUE = 255.0f;
    constexpr float STANDARDIZATION_MEAN = 127.5f;
    constexpr float STANDARDIZATION_SCALE = 128.0f;

    /**
     * 预处理图像： resize和标准化
     */
    static cv::Mat PreprocessImage(const cv::Mat& img)
    {
        // 调整尺寸
        cv::Mat imgResized;
        cv::resize(img, imgResized, cv::Size(TARGET_WIDTH, TARGET_HEIGHT), 0, 0, cv::INTER_LINEAR);

        // 转换为浮点型并标准化 (x - 127.5) / 128.0
        cv::Mat imgNorm;
        imgResized.convertTo(imgNorm, CV_32FC3);
        imgNorm = (imgNorm - STANDARDIZATION_MEAN) / STANDARDIZATION_SCALE;

        return imgNorm;
    }

    /**
     * 将HWC格式转换为CHW格式
     */
    static cv::Mat HwcToChw(const cv::Mat& hwcImg)
    {
        CV_Assert(!hwcImg.empty() && hwcImg.channels() == CHANNEL_NUM);
        std::vector<cv::Mat> channels;
        cv::split(hwcImg, channels);

        cv::Mat chwMat(CHANNEL_NUM, hwcImg.rows * hwcImg.cols, CV_32F);

        for (int channelIdx = 0; channelIdx < CHANNEL_NUM; ++channelIdx) {
            cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
            flatChannel.convertTo(flatChannel, CV_32F);
            flatChannel.copyTo(chwMat.row(channelIdx));
        }

        return chwMat;
    }

    /**
     * 将处理后的图像数据写入Tensor缓冲区
     */
    static Result ReadImageToTensorBuf(const cv::Mat& chwImg, TensorDesc& desc, TensorBuf& inBuf)
    {
        LOG(INFO) << "Read image to tensor buffer, dim count: " << desc.dimCount;
        const float* srcData = chwImg.ptr<float>();

        // for dlite
        if (desc.defaultStride == 0) {
            // 计算数据量
            size_t dataCount = chwImg.total();

            // 转换为FP16
            std::vector<uint16_t> fp16Data(dataCount);
            for (size_t i = 0; i < dataCount; ++i) {
                fp16Data[i] = FloatToHalf(srcData[i]);
            }

            memcpy(static_cast<void*>(inBuf.GetRawPtr()), fp16Data.data(), desc.defaultSize);
            return SUCCESS;
        }

        // for dpico, 对齐存储处理
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
                LOG(ERROR) << "Null pointer detected in tensor buffer copy";
                return FAILED;
            }

            memcpy(destPtr, srcPtr, width * dataSize);
        }
        return SUCCESS;
    }

    /**
     * 人脸网络预处理主函数
     */
    bool FaceNetPreprocess(std::vector<std::string>& fileList,
        std::vector<TensorBuf>& tensorBufs,
        std::vector<TensorDesc>& tensorDescs)
    {
        LOG(INFO) << "====== Start FaceNet preprocessing ======";

        auto cfg = ReadCfgFile("../data/cfg.txt");
        std::string saveBinDir = cfg["save_preprocess_bin"];

        // 创建保存目录
        if (!saveBinDir.empty() && !CreateDirectoryRecursive(saveBinDir)) {
            LOG(ERROR) << "Create preprocess bin directory failed: " << saveBinDir;
            return false;
        }

        LOG(INFO) << "Start processing " << fileList.size() << " images";

        for (size_t i = 0; i < fileList.size(); ++i) {
            std::string imgPath = fileList[i];
            LOG(INFO) << "Processing image: " << imgPath;

            // 提取文件名（不含扩展名）
            size_t lastSep = imgPath.find_last_of("/");
            size_t lastDot = imgPath.find_last_of(".");
            std::string fileName = imgPath.substr(lastSep + 1, lastDot - lastSep - 1);

            // 读取图像（假设为RGB格式）
            cv::Mat img = cv::imread(imgPath);
            if (img.empty()) {
                LOG(ERROR) << "Read image failed: " << imgPath;
                continue;
            }

            // 转换为RGB格式（OpenCV默认读取为BGR）
            cv::Mat imgRgb;
            cv::cvtColor(img, imgRgb, cv::COLOR_BGR2RGB);

            // 预处理图像
            cv::Mat processedImg = PreprocessImage(imgRgb);

            // 转换为CHW格式
            cv::Mat chwImg = HwcToChw(processedImg);

            if (!chwImg.isContinuous()) {
                chwImg = chwImg.clone();
            }

            // 填充tensor缓冲区
            if (ReadImageToTensorBuf(chwImg, tensorDescs[i], tensorBufs[i]) != SUCCESS) {
                LOG(ERROR) << "Fill tensor buffer failed for: " << imgPath;
                continue;
            }

            // 保存预处理结果
            if (!saveBinDir.empty()) {
                std::string binPath = saveBinDir + "/" + fileName + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                if (binFile.is_open()) {
                    size_t dataSize = TARGET_WIDTH * TARGET_HEIGHT * CHANNEL_NUM * sizeof(float);
                    binFile.write(reinterpret_cast<char*>(chwImg.data), dataSize);
                    binFile.close();

                    if (chmod(binPath.c_str(), 0755) != 0) {
                        LOG(WARNING) << "Set permissions failed for: " << binPath;
                    }
                    LOG(INFO) << "Saved preprocess bin: " << binPath;
                } else {
                    LOG(WARNING) << "Open bin file failed: " << binPath;
                }
            }
        }

        LOG(INFO) << "FaceNet preprocessing completed successfully";
        return true;
    }

} // namespace FaceNetNS
} // namespace Infer