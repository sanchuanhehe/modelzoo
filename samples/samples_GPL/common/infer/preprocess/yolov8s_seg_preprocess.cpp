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

#include "yolov8s_seg_preprocess.h"
#include "dev_interface_adapter.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
constexpr int FP16_BIT_NUM = 16;
constexpr int FP32_BIT_NUM = 32;
namespace Infer {
namespace Yolov8sSegNS {
    // 新增像素值归一化系数常量，替代魔鬼数字255.0
    constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
    constexpr int CHANNEL_COUNT = 3;

    static Result ReadImgFileToBuf(cv::Mat& chwImg, TensorDesc& desc, TensorBuf& inBuf)
    {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        const float* srcData = chwImg.ptr<float>();
        // for dlite
        if (desc.defaultStride == 0) {
            if (desc.typeSize == FP32_BIT_NUM) {
                memcpy(static_cast<float*>(inBuf.GetRawPtr()), srcData, desc.defaultSize);
                return SUCCESS;
            } else if (desc.typeSize == FP16_BIT_NUM) {
                // 计算数据量
                size_t dataCount = chwImg.total();

                // 转换为FP16
                std::vector<uint16_t> fp16Data(dataCount);
                for (size_t i = 0; i < dataCount; ++i) {
                    fp16Data[i] = FloatToHalf(srcData[i]);
                }

                memcpy(static_cast<void*>(inBuf.GetRawPtr()), fp16Data.data(), desc.defaultSize);
                return SUCCESS;
            } else {
                LOG(ERROR) << "dlite core unsupported desc.typeSize: " << desc.typeSize;
                return FAILED;
            }
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

    // Letterbox函数实现
    static cv::Mat Letterbox(const cv::Mat& img, const cv::Size& targetSize, bool scaleup)
    {
        cv::Size shape = img.size();
        const float minValue = 0.1f;

        double r = std::min(static_cast<double>(targetSize.width) / shape.width,
            static_cast<double>(targetSize.height) / shape.height);

        LOG(INFO) << "read r: " << r;
        cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
            static_cast<int>(std::round(shape.height * r)));

        LOG(INFO) << "unpad r: " << unpad.width << "  " << unpad.height;
        double w = (targetSize.width - unpad.width) / 2.0f;
        double h = (targetSize.height - unpad.height) / 2.0f;
        LOG(INFO) << "w r: " << w << "  " << h;

        cv::Mat resized;
        if (shape != unpad) {
            cv::resize(img, resized, unpad, 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }

        int top = static_cast<int>(std::round(h - minValue));
        int bottom = static_cast<int>(std::round(h + minValue));
        int left = static_cast<int>(std::round(w - minValue));
        int right = static_cast<int>(std::round(w + minValue));
        const int padValue = 114;

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
            cv::BORDER_CONSTANT, cv::Scalar(padValue, padValue, padValue));

        return padded;
    }

    static cv::Mat HwcToChw(const cv::Mat& hwcImg)
    {
        CV_Assert(!hwcImg.empty() && hwcImg.channels() == CHANNEL_COUNT);
        std::vector<cv::Mat> channels;
        cv::split(hwcImg, channels);

        cv::Mat chwMat(CHANNEL_COUNT, hwcImg.rows * hwcImg.cols, hwcImg.depth());
        chwMat = cv::Scalar(0);

        for (int channelIdx = 0; channelIdx < CHANNEL_COUNT; ++channelIdx) {
            cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
            flatChannel.copyTo(chwMat.row(channelIdx));
        }

        return chwMat;
    }

    bool Yolov8sSegPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
    {
        // 读取配置文件
        auto cfg = ReadCfgFile("../data/cfg.txt");
        int imgHeight = std::stoi(cfg["img_height"]);
        int imgWidth = std::stoi(cfg["img_width"]);
        std::string saveBin = cfg["save_preprocess_bin"];

        // 创建保存目录
        if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
            LOG(ERROR) << "Create preprocess bin dir failed";
            return false;
        }

        LOG(INFO) << "Processing " << fileList.size() << " images...";

        // 处理每个图像
        std::vector<int> imgSize = { imgWidth, imgHeight };
        for (size_t i = 0; i < fileList.size(); ++i) {
            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;
            // 提取文件名（不含路径和后缀）
            size_t start = imgPath.find_last_of("/");
            size_t end = imgPath.find_last_of(".");
            std::string outputName = imgPath.substr(start + 1, end - start - 1);
            // 读取图片文件
            cv::Mat im0 = cv::imread(imgPath);

            if (im0.empty()) {
                LOG(ERROR) << "Read image failed: " << imgPath;
                return false;
            }

            // 应用letterbox
            cv::Mat processed = Letterbox(im0, cv::Size(imgSize[0], imgSize[1]), false);

            // BGR到RGB
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

            cv::Mat chwImg = HwcToChw(processed);

            if (!chwImg.isContinuous()) {
                chwImg = chwImg.clone();
            }

            // 转换为浮点数并归一化
            cv::Mat doubleTemp;
            chwImg.convertTo(doubleTemp, CV_64FC3);
            doubleTemp /= PIXEL_NORMALIZE_FACTOR;
            doubleTemp.convertTo(chwImg, CV_32FC3);

            ReadImgFileToBuf(chwImg, tensorDescs[i], tensorBufs[i]);
            LOG(INFO) << "read end: " << imgPath;

            // 保存预处理结果
            if (!saveBin.empty()) {
                std::string binPath = saveBin + "/" + outputName + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                if (binFile.is_open()) {
                    const int floatSize = sizeof(float);
                    size_t dataSize = imgWidth * imgHeight * CHANNEL_COUNT * floatSize;
                    binFile.write((char*)chwImg.data, dataSize);
                    binFile.close();
                    // 设置文件权限
                    const int filePermission = 0755;
                    if (chmod(binPath.c_str(), filePermission) != 0) {
                        LOG(WARNING) << "Failed to set permissions for bin file: " << binPath;
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
}
}