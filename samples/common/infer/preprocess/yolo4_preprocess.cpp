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

#include "yolo4_preprocess.h"
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

namespace Infer {
// 新增子命名空间：将YOLO4相关代码全部放入Infer::Yolo4
namespace Yolo4 {
    // 新增像素值归一化系数常量，替代魔鬼数字255.0
    constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
    constexpr int BYTE_BIT_NUM = 8;
    constexpr int FP16_BIT_NUM = 16;
    constexpr int FP32_BIT_NUM = 32;

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
    static cv::Mat Letterbox(const cv::Mat& img, const cv::Size& target_size, bool scaleup)
    {
        cv::Size shape = img.size();

        double r = std::min(static_cast<double>(target_size.width) / shape.width,
            static_cast<double>(target_size.height) / shape.height);

        LOG(INFO) << "read r: " << r;
        cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
            static_cast<int>(std::round(shape.height * r)));

        LOG(INFO) << "unpad r: " << unpad.width << "  " << unpad.height;
        double w = (target_size.width - unpad.width) / 2.0f;
        double h = (target_size.height - unpad.height) / 2.0f;
        LOG(INFO) << "w r: " << w << "  " << h;

        cv::Mat resized;
        if (shape != unpad) {
            cv::resize(img, resized, unpad, 0, 0, cv::INTER_LINEAR);
        } else {
            resized = img.clone();
        }

        int top = static_cast<int>(std::round(h - 0.1));
        int bottom = static_cast<int>(std::round(h + 0.1));
        int left = static_cast<int>(std::round(w - 0.1));
        int right = static_cast<int>(std::round(w + 0.1));

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        return padded;
    }

    static cv::Mat HwcToChw(const cv::Mat& hwcImg)
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

        return chwMat;
    }

    bool Yolov4Preprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
    {
        // 读取配置文件
        auto cfg = ReadCfgFile("../data/cfg.txt");
        int imgHeight = std::stoi(cfg["img_height"]);
        int imgWidth = std::stoi(cfg["img_width"]);
        std::string saveBin = cfg["save_preprocess_bin"];
        std::string saveTxt = cfg["save_preprocess_txt"];

        // 创建保存目录
        if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
            LOG(ERROR) << "Create preprocess bin dir failed";
            return false;
        }
        if (!saveTxt.empty() && !CreateDirectoryRecursive(saveTxt)) {
            LOG(ERROR) << "Create preprocess txt dir failed";
            return false;
        }

        LOG(INFO) << "Processing " << fileList.size() << " images...";

        // 处理每个图像
        std::vector<int> img_size = { imgWidth, imgHeight };
        for (size_t i = 0; i < fileList.size(); ++i) {
            std::string imgPath = fileList[i];
            LOG(INFO) << "imgPath: " << imgPath;
            // 提取文件名（不含路径和后缀）
            size_t start = imgPath.find_last_of("/");
            size_t end = imgPath.find_last_of(".");
            std::string outputName = imgPath.substr(start, end - start);
            // 读取图片文件
            cv::Mat im0 = cv::imread(imgPath);

            if (im0.empty()) {
                LOG(ERROR) << "Read image failed: " << imgPath;
                return false;
            }

            // 应用letterbox
            cv::Mat processed = Letterbox(im0, cv::Size(img_size[0], img_size[1]), false);

            // BGR到RGB
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

            cv::Mat chwImg = HwcToChw(processed);

            if (!chwImg.isContinuous()) {
                chwImg = chwImg.clone();
            }

            // 转换为浮点数并归一化
            cv::Mat double_temp;
            chwImg.convertTo(double_temp, CV_64FC3);
            double_temp /= PIXEL_NORMALIZE_FACTOR;
            double_temp.convertTo(chwImg, CV_32FC3);

            ReadImgFileToBuf(chwImg, tensorDescs[i], tensorBufs[i]);
            LOG(INFO) << "read end: " << imgPath;

            // 保存预处理结果
            if (!saveBin.empty()) {
                std::string binPath = saveBin + "/" + outputName + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                if (binFile.is_open()) {
                    size_t dataSize = imgWidth * imgHeight * 3 * sizeof(float);
                    binFile.write((char*)chwImg.data, dataSize);
                    binFile.close();
                    // 设置文件权限
                    if (chmod(binPath.c_str(), 0755) != 0) {
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