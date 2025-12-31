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

#include "crowdcount_postprocess.h"
#include "log.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <algorithm>
#include <numeric>

namespace Infer {
namespace CrowdCountNS {

// 常量定义
constexpr int CHANNEL_NUM = 1;
constexpr int TARGET_HEIGHT = 56;
constexpr int TARGET_WIDTH = 56;
constexpr float NORMALIZE_SCALE = 255.0f;

std::string GetFileName(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ?
           path.substr(start) : path.substr(start, end - start);
}

int CalculateRawCount(const cv::Mat& rawDensityMap) {
    float sum = 0.0f;
    for (int i = 0; i < rawDensityMap.rows; ++i) {
        const float* rowPtr = rawDensityMap.ptr<float>(i);
        for (int j = 0; j < rawDensityMap.cols; ++j) {
            sum += rowPtr[j];
        }
    }
    return static_cast<int>(sum);
}

std::pair<cv::Mat, int> PostprocessDensityMap(const float* preds, int targetHeight, int targetWidth,
                             int originalHeight, int originalWidth, float threshold) {
    // 转换为单通道密度图 (H, W) - 模型原始输出尺寸
    cv::Mat rawDensityMap(targetHeight, targetWidth, CV_32F);
    memcpy(rawDensityMap.data, preds, targetHeight * targetWidth * sizeof(float));

    // 应用阈值过滤
    rawDensityMap.setTo(0.0f, rawDensityMap < threshold);

    // 计算人数（在原始尺寸密度图上，与Python一致）
    int count = CalculateRawCount(rawDensityMap);

    // 为可视化准备缩放后的密度图
    cv::Mat visDensityMap = rawDensityMap.clone();
    if (originalHeight > 0 && originalWidth > 0 &&
        (originalHeight != targetHeight || originalWidth != targetWidth)) {
        cv::resize(visDensityMap, visDensityMap, cv::Size(originalWidth, originalHeight),
                  0, 0, cv::INTER_LINEAR);
    }

    return {visDensityMap, count};  // 返回可视化用图和计数结果
}

cv::Mat VisualizeDensityMap(const cv::Mat& densityMap) {
    // 归一化到0-255
    cv::Mat densityNorm;
    double maxVal;
    cv::minMaxLoc(densityMap, nullptr, &maxVal);

    if (maxVal > 0.0f) {
        densityMap.convertTo(densityNorm, CV_8UC1, NORMALIZE_SCALE / maxVal);
    } else {
        densityNorm = cv::Mat::zeros(densityMap.size(), CV_8UC1);
    }

    // 应用颜色映射
    cv::Mat coloredMap;
    cv::applyColorMap(densityNorm, coloredMap, cv::COLORMAP_JET);
    return coloredMap;
}

bool CrowdCountPostprocess(std::vector<std::string>& fileList,
                            std::vector<TensorBuf>& outBufs,
                            std::vector<TensorDesc>& outDescs) {
    LOG(INFO) << "====== 开始人群计数后处理 ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    float threshold = std::stof(cfg["threshold"]);
    std::string outputDir = cfg["save_result_dir"];
    std::string saveBin = cfg["save_result_bin"];
    std::string densityMapDir = cfg["save_density_maps_dir"];
    std::string txtDir = outputDir + "/txt";

    // 创建输出目录结构
    if (!outputDir.empty()) {
        if (!CreateDirectoryRecursive(outputDir)) {
            LOG(ERROR) << "创建输出目录失败: " << outputDir;
            return false;
        }
        if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
            LOG(ERROR) << "创建原始输出bin目录失败: " << saveBin;
            return false;
        }
        if (!densityMapDir.empty() && !CreateDirectoryRecursive(densityMapDir)) {
            LOG(ERROR) << "创建密度图目录失败: " << densityMapDir;
            return false;
        }
        if (!CreateDirectoryRecursive(txtDir)) {
            LOG(ERROR) << "创建TXT计数目录失败: " << txtDir;
            return false;
        }
    }

    // 处理每个图像
    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const std::string& imgPath = fileList[imgIdx];
        std::string fileName = GetFileName(imgPath);
        LOG(INFO) << "处理文件: " << fileName;

        // 保存原始输出bin
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + fileName + "_output_0.bin";
            std::ofstream binFile(binPath, std::ios::binary);
            if (binFile.is_open()) {
                binFile.write(reinterpret_cast<char*>(outBufs[imgIdx].data.get()),
                             outBufs[imgIdx].size);
                LOG(INFO) << "保存原始输出bin至: " << binPath;
            } else {
                LOG(WARNING) << "无法保存原始输出bin至: " << binPath;
            }
        }

        // 验证输出数据尺寸
        const float* preds = reinterpret_cast<const float*>(outBufs[imgIdx].data.get());
        size_t expectedSize = CHANNEL_NUM * TARGET_HEIGHT * TARGET_WIDTH * sizeof(float);
        if (outBufs[imgIdx].size != expectedSize) {
            LOG(ERROR) << "输出尺寸不匹配，预期: " << expectedSize
                      << "，实际: " << outBufs[imgIdx].size;
            continue;
        }

        // 获取原始图像尺寸（即使读取失败也继续，使用模型输出尺寸可视化）
        int originalHeight = TARGET_HEIGHT;
        int originalWidth = TARGET_WIDTH;
        cv::Mat originalImg = cv::imread(imgPath);
        if (!originalImg.empty()) {
            originalHeight = originalImg.rows;
            originalWidth = originalImg.cols;
        } else {
            LOG(WARNING) << "无法读取原始图像: " << imgPath << "，使用模型输出尺寸可视化";
        }

        // 执行后处理（获取可视化图和计数结果）- 修复结构化绑定语法警告
        std::pair<cv::Mat, int> postResult = PostprocessDensityMap(
            preds, TARGET_HEIGHT, TARGET_WIDTH,
            originalHeight, originalWidth,
            threshold
        );
        cv::Mat visDensityMap = postResult.first;
        int count = postResult.second;

        // 输出人数计数（来自原始尺寸密度图的统计）
        LOG(INFO) << "人数计数: " << count;

        // 保存计数结果到单独的TXT文件（与Python逻辑一致）
        std::string txtPath = txtDir + "/" + fileName + ".txt";
        std::ofstream txtFile(txtPath);
        if (txtFile.is_open()) {
            txtFile << count;
            txtFile.close();
            LOG(INFO) << "已保存计数结果至: " << txtPath;
        } else {
            LOG(WARNING) << "无法保存计数结果至: " << txtPath;
        }

        // 可视化并保存密度图（使用缩放后的图）
        if (!densityMapDir.empty()) {
            cv::Mat densityMapVis = VisualizeDensityMap(visDensityMap);
            std::string imgSavePath = densityMapDir + "/" + fileName + "_density.jpg";
            if (!cv::imwrite(imgSavePath, densityMapVis)) {
                LOG(WARNING) << "保存密度图失败: " << imgSavePath;
            } else {
                LOG(INFO) << "已保存密度图至: " << imgSavePath;
            }
        }
    }

    LOG(INFO) << "所有后处理结果已保存至: " << outputDir;
    LOG(INFO) << "计数TXT文件已保存至: " << txtDir;
    LOG(INFO) << "====== 人群计数后处理完成 ======";
    return true;
}

}  // namespace CrowdCountNS
}  // namespace Infer