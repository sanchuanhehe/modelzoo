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

#include "yolov8s_obb_preprocess.h"
#include "log.h"
#include "dev_interface_adapter.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
constexpr double LETTERBOX_HALF = 2.0;

namespace Infer {
namespace Yolo8sObb {
    // 像素值归一化系数常量
    constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
    constexpr int CHANNEL_COUNT = 3;
    constexpr float LETTERBOX_EPSILON = 0.1f;
    constexpr int LETTERBOX_PAD_VALUE = 114;
    constexpr size_t FLOAT_SIZE = sizeof(float);

    // ReadImgFileToBuf: 将内存中的数据拷贝到模型输入缓冲区
    // 逻辑与模型类型无关，只与硬件平台（如是否需要对齐）有关
    static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
    {
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

    // Letterbox: YOLO系列标准的等比例缩放和填充方法
    static cv::Mat Letterbox(const cv::Mat &img, const cv::Size &targetSize, bool scaleup)
    {
        cv::Size shape = img.size();
        double r = std::min(static_cast<double>(targetSize.width) / shape.width,
                            static_cast<double>(targetSize.height) / shape.height);

        cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
                       static_cast<int>(std::round(shape.height * r)));

        double w = (targetSize.width - unpad.width) / LETTERBOX_HALF;
        double h = (targetSize.height - unpad.height) / LETTERBOX_HALF;

        cv::Mat resized;
        if (shape != unpad) {
            cv::resize(img, resized, unpad, 0, 0, cv::INTER_LINEAR);
        }
        else {
            resized = img.clone();
        }

        int top = static_cast<int>(std::round(h - LETTERBOX_EPSILON));
        int bottom = static_cast<int>(std::round(h + LETTERBOX_EPSILON));
        int left = static_cast<int>(std::round(w - LETTERBOX_EPSILON));
        int right = static_cast<int>(std::round(w + LETTERBOX_EPSILON));

        cv::Mat padded;
        cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE, LETTERBOX_PAD_VALUE));

        return padded;
    }

    // HwcToChw: 将OpenCV的HWC格式转换为模型输入所需的CHW格式
    static cv::Mat HwcToChw(const cv::Mat &hwcImg)
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

    bool Yolov8sObbPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
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
        std::vector<int> imgSize = {imgWidth, imgHeight};
        for (size_t i = 0; i < fileList.size(); ++i) {
            std::string imgPath = fileList[i];
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

            // 1. 应用letterbox进行等比例缩放和填充
            cv::Mat processed = Letterbox(im0, cv::Size(imgSize[0], imgSize[1]), false);

            // 2. BGR到RGB颜色空间转换
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

            // 3. HWC到CHW格式转换
            cv::Mat chwImg = HwcToChw(processed);

            if (!chwImg.isContinuous()) {
                chwImg = chwImg.clone();
            }

            // 转换为浮点数并归一化
            cv::Mat doubleTemp;
            chwImg.convertTo(doubleTemp, CV_64FC3);
            doubleTemp /= PIXEL_NORMALIZE_FACTOR;
            doubleTemp.convertTo(chwImg, CV_32FC3);

            // 将处理后的数据拷贝到模型输入缓冲区
            ReadImgFileToBuf(chwImg, tensorDescs[i], tensorBufs[i]);

            // 保存预处理结果
            if (!saveBin.empty()) {
                std::string binPath = saveBin + "/" + outputName + ".bin";
                std::ofstream binFile(binPath, std::ios::binary);
                if (binFile.is_open()) {
                    size_t dataSize = imgWidth * imgHeight * CHANNEL_COUNT * FLOAT_SIZE;
                    binFile.write((char*)chwImg.data, dataSize);
                    binFile.close();
                    // 设置文件权限
                    constexpr int filePermission = 0755;
                    if (chmod(binPath.c_str(), filePermission) != 0) {
                        LOG(WARNING) << "Failed to set permissions for bin file: " << binPath;
                    }
                }
                else {
                    LOG(WARNING) << "Failed to open bin file: " << binPath;
                }
            }
        }

        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}
}
