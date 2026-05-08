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

#include "xstereo_preprocess.h"
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
    static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
    {
        LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
        if (desc.defaultStride == 0) {
            desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
            inBuf.stride = desc.defaultStride;
        }
        int64_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }

        int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
        size_t dataSize = desc.typeSize / BYTE_BIT_NUM;
        size_t strideElemNum = inBuf.stride / dataSize;

        // 获取数据指针
        char *srcData = chwImg.ptr<char>();
      
        // 计算总字节数
        size_t totalBytes = chwImg.total() * chwImg.elemSize();

        for (int64_t loop = 0; loop < loopTimes; loop++) {
            char *destPtr = static_cast<char *>(inBuf.GetRawPtr()) + loop * strideElemNum;
            char *srcPtr = srcData + loop * width;
           
            if (srcData + loop * width > srcData + totalBytes) {
                LOG(ERROR) << "Source buffer overflow at loop " << loop;
                return FAILED;
            }

            if (static_cast<char *>(inBuf.GetRawPtr()) + loop * strideElemNum > static_cast<char *>(inBuf.GetRawPtr()) +
                inBuf.size) {
                LOG(ERROR) << "Source inbuf overflow at loop " << loop;
                return FAILED;
            }

            if (destPtr == nullptr || srcPtr == nullptr) {
                LOG(ERROR) << "Null pointer detected";
                return FAILED;
            }
            // 拷贝数据
            memcpy(destPtr, srcPtr, width * dataSize);
        }
        LOG(INFO) << "ReadImgFileToBuf sucess------";
        return SUCCESS;
    }

    static cv::Mat hwcToChw(const cv::Mat &hwcImg)
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

        return chwMat; // 返回 3 × (高度*宽度) 的CHW矩阵
    }
       

    // 安全的图像读取（处理KITTI PNG的alpha通道/16位深度问题）
    static cv::Mat SafeImread(const std::string& imgPath)
    {
        // 1. 强制以3通道BGR读取（自动丢弃alpha通道/转换16位→8位）
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR); 
        if (img.empty()) {
            throw std::runtime_error("Image read failed: " + imgPath);
        }
        // 2. 强制转换为8位无符号（避免16位深度）
        if (img.depth() != CV_8U) {
            img.convertTo(img, CV_8UC3);
        }
        // 3. 确保通道数为3（KITTI PNG可能含alpha通道）
        if (img.channels() != 3) {
            cv::cvtColor(img, img, cv::COLOR_BGRA2BGR); // 4通道→3通道
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        return img;
    }

    static cv::Mat loadImage(const std::string &imgPath, int target_w = 1248, int target_h = 384, bool rect_mode = true) 
    {
        // 1. 安全读取图像（处理KITTI PNG的特殊格式）
        cv::Mat img = SafeImread(imgPath);
        int h = img.rows, w = img.cols;

        // 2. 中心剪裁（严格边界校验）
        int cropLeft = std::max(0, (w - target_w) / 2);
        int cropTop = std::max(0, (h - target_h) / 2);
        int cropRight = std::min(w, cropLeft + target_w);
        int cropBottom = std::min(h, cropTop + target_h);
        
        // 校验剪裁ROI合法性
        if (cropRight <= cropLeft || cropBottom <= cropTop) {
            throw std::runtime_error("Invalid crop ROI: " + std::to_string(cropLeft) + "," + std::to_string(cropTop) 
                                    + "," + std::to_string(cropRight) + "," + std::to_string(cropBottom));
        }
        cv::Rect cropRoi(cropLeft, cropTop, cropRight - cropLeft, cropBottom - cropTop);
        cv::Mat imgCropped = img(cropRoi).clone(); // 强制clone避免ROI引用问题
        int cropH = imgCropped.rows, cropW = imgCropped.cols;

        // 3. 零填充（严格边界校验）
        cv::Mat imgPadded = cv::Mat::zeros(target_h, target_w, CV_8UC3);
        cv::Rect pasteRoi(0, 0, cropW, cropH);
        // 防止剪裁尺寸超过目标尺寸（极端情况）
        pasteRoi.width = std::min(pasteRoi.width, imgPadded.cols - pasteRoi.x);
        pasteRoi.height = std::min(pasteRoi.height, imgPadded.rows - pasteRoi.y);
        if (pasteRoi.width <= 0 || pasteRoi.height <= 0) {
            throw std::runtime_error("Invalid paste ROI: " + std::to_string(pasteRoi.width) + "," +
                std::to_string(pasteRoi.height));
        }
    
        int bottom = target_h - cropH;
        int right = target_w - cropW;
        
        cv::Mat padded;
        cv::copyMakeBorder(imgCropped, padded, 0, bottom, 0, right, cv::BORDER_REPLICATE);
        //imgCropped(pasteRoi).copyTo(imgPadded(pasteRoi));
        cv::Mat chwMat = hwcToChw(padded);
        return chwMat;
    } 

    bool XStereoPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs,
        std::vector<TensorDesc>& inDescs)
    {
        LOG(INFO) << " xstereo Processing " << fileList.size();

        // 读取配置文件
        auto cfg = ReadCfgFile("../data/cfg.txt");
        int imgHeight = std::stoi(cfg["img_height"]);
        int imgWidth = std::stoi(cfg["img_width"]);
        bool type = std::stoi(cfg["type"]) == 1;

        LOG(INFO) << "Processing " << fileList.size() << " images...";
        std::vector<std::string> imgPaths;
        if (type) {
            imgPaths = GetInputList(fileList[0]);
        } else {
            imgPaths = fileList;
        }
        
        for (size_t i = 0; i < imgPaths.size(); ++i) {
            std::string imgPath = imgPaths[i];
            LOG(INFO) << "imgPath-------: " << imgPath;
            cv::Mat chwImg = loadImage(imgPath, imgWidth, imgHeight);
            ReadImgFileToBuf(chwImg, inDescs[i], inBufs[i]);
            LOG(INFO) << "read end: " << imgPath;
        }

        LOG(INFO) << "PreProcessing completed successfully!";
        return true;
    }
}