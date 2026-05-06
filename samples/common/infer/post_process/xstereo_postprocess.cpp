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

#include "xstereo_postprocess.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <climits>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include "utils.h"
#include "log.h"
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <map>

using namespace std;
using json = nlohmann::json;

namespace Infer
{
    using namespace std;
    constexpr float MIN_VAL = 0.0f;
    constexpr float MAX_VAL = 128.0f;
    constexpr int BYTE_BIT_NUM = 8;

    static Result GetOutputWithBin(Infer::TensorBuf &outBuf, Infer::TensorDesc &outDesc, std::string outputBinFileName,
        std::vector<float> &noStrideBuf)
    {
        LOG(INFO) << "GetOutputWithBin: " << outputBinFileName;
        int64_t lastDim = outDesc.dims[outDesc.dimCount - 1];
        size_t dataSize = outDesc.typeSize / BYTE_BIT_NUM;
        size_t lastDimSize = dataSize * lastDim;
        size_t loopNum = outBuf.size / outBuf.stride;
        size_t strideElemNum = outBuf.stride / dataSize;
        size_t invalidSize = outBuf.stride - lastDimSize;
        float *outData = static_cast<float *>(outBuf.GetRawPtr());
        size_t outSize = outBuf.size / dataSize;
        if (invalidSize == 0) {
            // not stride invalid space, return directly.
            LOG(INFO) << "not stride invalid space, return directly.";
            // malloc temp buffer to clear stride useless temp buffer to help output.bin compare
            std::vector<float> tempBuf(outData, outData + outSize);
            noStrideBuf.assign(tempBuf.begin(), tempBuf.end());
        } else {
            /** */
            std::vector<float> tempBuf(outData, outData + outSize);
            for (size_t i = 0; i < loopNum; ++i) {
                size_t offset = i * strideElemNum;
                for (int64_t index = 0; index < lastDim; index++) {
                    // int8_t* outOffset = outData + ;
                    noStrideBuf.push_back(tempBuf[offset + index]);
                }
            }
        }
        if (noStrideBuf.empty()) {
            LOG(ERROR) << "noStrideBuf malloc fail";
            return FAILED;
        }
        std::ofstream fout(outputBinFileName, std::ios::out | std::ios::binary);
        if (fout.good() == false) {
            LOG(ERROR) << "create output file failed";
            return FAILED;
        }
        fout.write((char *)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
        fout.close();
        return SUCCESS;
    }

    static void SaveResultBin(Infer::TensorBuf &outBufs, Infer::TensorDesc &outDescs, const std::string &filePath,
        std::vector<float> &temp)
    {
        LOG(INFO) << "save bin to ==========" << filePath;
        // 获取保存文件路径和文件名
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");

        std::string fileName = filePath.substr(start, end - start);
        std::string resultPath = "../out/result";
        struct stat info;
        if (stat(resultPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(resultPath.c_str(), 0777);
            LOG(INFO) << "create file success";
        }
        std::string jpgPath = resultPath + "/jpg";
        if (stat(jpgPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(jpgPath.c_str(), 0777);
        }
        std::string binPath = resultPath + "/bin";
        if (stat(binPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(binPath.c_str(), 0777);
        }

        // 保存bin文件
        std::string binFile = binPath + fileName + "_0.bin";
        GetOutputWithBin(outBufs, outDescs, binFile, temp);
    };

    /**
     * @brief 深度图后处理（归一化+颜色映射+无效值处理）
     * @param depth 输入深度图，单通道 (H, W)，数据类型 CV_32F/CV_64F
     * @param file_path 图像路径（用于打印日志，对应 Python 的 file_path）
     * @return cv::Mat 可视化彩色图，3通道 BGR (H, W, 3)，CV_8UC3
     */
    static cv::Mat Postprocess(const cv::Mat &depth, const std::string &file_path)
    {
        LOG(INFO) << "postprocess img_path:"<< file_path;

        // 检查输入深度图有效性
        if (depth.empty()) {
            throw std::runtime_error("Input depth map is empty!");
        }
        if (depth.channels() != 1) {
            throw std::runtime_error("Input depth map must be single-channel!");
        }
        if (depth.type() != CV_32F && depth.type() != CV_64F) {
            throw std::runtime_error("Input depth map must be CV_32F or CV_64F!");
        }

        int H = depth.rows;                                 // 深度图高度（对应 Python 的 ny）
        int W = depth.cols;                                 // 深度图宽度（对应 Python 的 nx）

        // 1. 生成无效值掩码（depth >= inf）
        cv::Mat invalidMask = (depth >= std::numeric_limits<float>::infinity());
        // 处理 CV_64F 类型（double 精度）
        if (depth.type() == CV_64F) {
            invalidMask = (depth >= std::numeric_limits<double>::infinity());
        }

        // 2. 深度归一化：(depth - min_val) / (max_val - min_val) → [0, 1]，再缩放到 [0, 255]
        cv::Mat vis_normalized;
        float scale = 1.0f / (MAX_VAL - MIN_VAL);
        if (depth.type() == CV_32F) {
            // 对 CV_32F 类型直接计算
            depth.convertTo(vis_normalized, CV_32F, scale, -MIN_VAL * scale);
        } else { // CV_64F
            depth.convertTo(vis_normalized, CV_32F, scale, -MIN_VAL * scale);
        }

        // 裁剪到 [0, 1] 范围（对应 Python 的 clip(0, 1)）
        cv::threshold(vis_normalized, vis_normalized, 1.0f, 1.0f, cv::THRESH_TRUNC);
        cv::threshold(vis_normalized, vis_normalized, 0.0f, 0.0f, cv::THRESH_TOZERO);

        // 缩放到 [0, 255] 并转为 8位无符号整数
        cv::Mat vis8u;
        vis_normalized.convertTo(vis8u, CV_8UC1, 255.0f); // CV_8UC1: 单通道 8位

        // 3. 应用颜色映射（JET 色表，对应 Python 的 cv2.applyColorMap）
        cv::Mat visColor;
        cv::applyColorMap(vis8u, visColor, cv::COLORMAP_JET); // 输出为 3通道 BGR（CV_8UC3）

        // 4. 无效区域（invalidMask）置黑
        if (cv::countNonZero(invalidMask) > 0) {
            // 将掩码转为 CV_8UC1 以匹配图像尺寸
            cv::Mat mask_8u;
            invalidMask.convertTo(mask_8u, CV_8UC1);
            // 遍历所有像素，无效区域设为黑色（B=0, G=0, R=0）
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    if (mask_8u.at<uchar>(y, x) != 0) { // 无效像素（mask为真）
                        visColor.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // 置黑
                    }
                }
            }
        }

        return visColor;
    }

    // 安全的图像读取（处理KITTI PNG的alpha通道/16位深度问题）
    static cv::Mat SafeImread(const std::string& imgPath) {
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

    bool XStereoPostprocess(std::vector<std::string>& inputFileList, std::vector<TensorBuf>& outBufs,
        std::vector<TensorDesc>& outDescs)
    {
        LOG(INFO) << "XStereoPostprocess" ;
        std::vector<std::vector<cv::Point>> bbox;
         // 读取配置文件
        auto cfg = ReadCfgFile("../data/cfg.txt");
        int imgHeight = std::stoi(cfg["img_height"]);
        int imgWidth = std::stoi(cfg["img_width"]);
        LOG(INFO) << "save jpg to ==========" << inputFileList[0];
        bool type = std::stoi(cfg["type"]) == 1;

        LOG(INFO) << "Processing " << inputFileList.size() << " images...";
        std::vector<std::string> imgPaths;
        if (type) {
            imgPaths = GetInputList(inputFileList[0]);
        } else {
            imgPaths = inputFileList;
        }
        for (size_t i = 0; i < outBufs.size(); i++) {
            TensorDesc desc = outDescs[i];
            TensorBuf buf = outBufs[i];
            if (desc.defaultStride == 0) {
                desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
                buf.stride = desc.defaultStride;
            }
            std::vector<float> temp;
            
            SaveResultBin(buf, desc, imgPaths[0], temp);
            cv::Mat inferenceResult(imgHeight, imgWidth, CV_32FC1, temp.data()); 
            
            cv::Mat depthColored = Postprocess(inferenceResult, imgPaths[0]);
            cv::Mat img = SafeImread(imgPaths[0]);
            int h = img.rows, w = img.cols;

            // 2. 中心剪裁（严格边界校验）
            int cropLeft = std::max(0, (w - imgWidth) / 2);
            int cropTop = std::max(0, (h - imgHeight) / 2);
            int cropRight = std::min(w, cropLeft + imgWidth);
            int cropBottom = std::min(h, cropTop + imgHeight);
            cv::Rect cropRoi(0, 0, cropRight - cropLeft, cropBottom - cropTop);
            cv::Mat imgCropped = depthColored(cropRoi).clone(); // 强制clone避免ROI引用问题
            // 获取保存文件路径和文件名
            size_t start = imgPaths[0].find_last_of("/");
            size_t end = imgPaths[0].find_last_of(".");
            std::string fileName = imgPaths[0].substr(start + 1, end - start - 1);
            std::string jpgName = "../out/result/jpg/" + fileName + ".jpg";
            if (!jpgName.empty()) {
                cv::imwrite(jpgName, imgCropped);
                LOG(INFO) << "save jpg to " << jpgName;
            }
            LOG(INFO) << "dump final data success " << inputFileList[0];
        }
        return true;
    }
}