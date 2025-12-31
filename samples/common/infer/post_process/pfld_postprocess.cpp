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

#include "pfld_postprocess.h"
#include "log.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <cmath>

namespace Infer {
namespace PFLDNS {

// 常量定义
constexpr int DEFAULT_TARGET_WIDTH = 112;
constexpr int DEFAULT_TARGET_HEIGHT = 112;
constexpr int COORDINATE_DIMENSION = 2;
constexpr int FLOAT_PRECISION = 6;
constexpr int BYTE_BIT_NUM = 8;
constexpr int FP16_BIT_NUM = 16;

/**
 * @brief 从文件路径提取文件名（不含扩展名）
 * @param path 文件路径
 * @return 文件名
 */
std::string GetFileName(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ?
           path.substr(start) : path.substr(start, end - start);
}

/**
 * @brief 解码 landmarks 并缩放到目标尺寸
 * @param preds 推理输出数据
 * @param pointCount 关键点个数
 * @param targetWidth 目标宽度
 * @param targetHeight 目标高度
 * @return 解码后的关键点坐标
 */
std::vector<std::pair<float, float>> DecodeLandmarks(const float* preds,
                                                    size_t pointCount,
                                                    int targetWidth,
                                                    int targetHeight) {
    std::vector<std::pair<float, float>> landmarks;

    for (size_t i = 0; i < pointCount; ++i) {
        float x = preds[COORDINATE_DIMENSION * i] * targetWidth;
        float y = preds[COORDINATE_DIMENSION * i + 1] * targetHeight;
        landmarks.emplace_back(x, y);
    }

    return landmarks;
}

/**
 * @brief 保存关键点坐标到TXT文件
 * @param landmarks 关键点坐标
 * @param txtPath 保存路径
 * @return 是否保存成功
 */
bool SaveLandmarksToTxt(const std::vector<std::pair<float, float>>& landmarks,
                       const std::string& txtPath) {
    std::ofstream txtFile(txtPath);
    if (!txtFile.is_open()) {
        LOG(ERROR) << "Open txt file failed: " << txtPath;
        return false;
    }

    for (const auto& point : landmarks) {
        txtFile.precision(FLOAT_PRECISION);
        txtFile << std::fixed << point.first << " " << point.second << "\n";
    }

    txtFile.close();
    return true;
}

/**
 * @brief PFLD模型后处理主函数
 * @param fileList 图像文件路径列表
 * @param outBufs 输出张量缓冲区列表
 * @param outDescs 输出张量描述符列表
 * @return 处理是否成功
 */
bool PFLDPostprocess(std::vector<std::string>& fileList,
                    std::vector<TensorBuf>& outBufs,
                    std::vector<TensorDesc>& outDescs) {
    LOG(INFO) << "====== Start PFLD postprocessing ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    std::string outputTxtDir = cfg["save_result_txt"];
    std::string saveBin = cfg["save_result_bin"];

    // 创建输出目录
    if (!outputTxtDir.empty() && !CreateDirectoryRecursive(outputTxtDir)) {
        LOG(ERROR) << "Create output txt directory failed: " << outputTxtDir;
        return false;
    }
    if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
        LOG(ERROR) << "创建原始输出bin目录失败: " << saveBin;
        return false;
    }

    // 处理每个图像结果
    int outBufsIdx = 1;
    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const std::string& imgPath = fileList[imgIdx];
        std::string fileName = GetFileName(imgPath);
        LOG(INFO) << "Processing file: " << fileName;

        // 保存原始输出bin
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + fileName + "_output_0.bin";
            std::ofstream binFile(binPath, std::ios::binary);
            if (binFile.is_open()) {
                binFile.write(reinterpret_cast<char*>(outBufs[outBufsIdx].data.get()),
                             outBufs[outBufsIdx].size);
                LOG(INFO) << "保存原始输出bin至: " << binPath;
            } else {
                LOG(WARNING) << "无法保存原始输出bin至: " << binPath;
            }
        }

        // 获取推理结果数据并转换为FP32
        size_t dataSize = outBufs[outBufsIdx].size;
        size_t typeSize = outDescs[outBufsIdx].typeSize; // 16 or 32 bit
        size_t floatCount = dataSize / (typeSize / BYTE_BIT_NUM); // 98 points, 196 num

        std::vector<float> preds(floatCount);
        if (dataSize == 0 || outBufs[outBufsIdx].data.get() == nullptr) {
            LOG(ERROR) << "Invalid output data for: " << fileName;
            continue;
        }
        if (typeSize == FP16_BIT_NUM) {
            const uint16_t* fp16Preds = reinterpret_cast<const uint16_t*>(outBufs[outBufsIdx].data.get());
            for (size_t i = 0; i < floatCount; ++i) {
                preds[i] = HalfToFloat(fp16Preds[i]);
            }
        } else {
            const float* fp32Preds = reinterpret_cast<const float*>(outBufs[outBufsIdx].data.get());
            for (size_t i = 0; i < floatCount; ++i) {
                preds[i] = fp32Preds[i];
            }
        }

        // 解码关键点时使用FP32数据
        size_t pointCount = floatCount / COORDINATE_DIMENSION;
        auto landmarks = DecodeLandmarks(preds.data(), pointCount, DEFAULT_TARGET_WIDTH, DEFAULT_TARGET_HEIGHT);

        if (landmarks.empty()) {
            LOG(WARNING) << "No landmarks detected for: " << fileName;
            continue;
        }

        // 保存结果到TXT文件
        std::string txtPath = outputTxtDir + "/" + fileName + ".txt";
        if (SaveLandmarksToTxt(landmarks, txtPath)) {
            LOG(INFO) << "Saved landmarks to: " << txtPath;
        } else {
            LOG(WARNING) << "Failed to save landmarks to: " << txtPath;
        }
    }

    LOG(INFO) << "====== PFLD postprocessing completed ======";
    return true;
}

}  // namespace PfldNS
}  // namespace Infer