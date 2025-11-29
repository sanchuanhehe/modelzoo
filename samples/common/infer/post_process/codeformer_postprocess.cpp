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

#include "codeformer_postprocess.h"
#include "log.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <algorithm>

namespace Infer {
namespace CodeFormerNS {

constexpr int TARGET_WIDTH = 512;
constexpr int TARGET_HEIGHT = 512;
constexpr int CHANNEL_NUM = 3;
constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
constexpr float CLAMP_MIN = -1.0f;
constexpr float CLAMP_MAX = 1.0f;
constexpr float DENORM_OFFSET = 1.0f;
constexpr float DENORM_SCALE = 2.0f;
constexpr float CONVERT_OFFSET = 0.5f;

std::string GetFileName(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ?
           path.substr(start) : path.substr(start, end - start);
}

cv::Mat Postprocess(const float* preds, int originalHeight, int originalWidth) {
    // 转换为CHW格式Mat (3, H, W)
    cv::Mat chwMat(CHANNEL_NUM, TARGET_HEIGHT * TARGET_WIDTH, CV_32F);
    memcpy(chwMat.data, preds, CHANNEL_NUM * TARGET_HEIGHT * TARGET_WIDTH * sizeof(float));

    // 裁剪到[-1, 1]并映射到[0, 1]
    for (size_t i = 0; i < chwMat.total(); ++i) {
        float val = chwMat.at<float>(i);
        val = std::max(CLAMP_MIN, std::min(CLAMP_MAX, val));
        chwMat.at<float>(i) = (val + DENORM_OFFSET) / DENORM_SCALE;
    }

    // CHW转HWC
    std::vector<cv::Mat> channels(CHANNEL_NUM);
    for (int c = 0; c < CHANNEL_NUM; ++c) {
        channels[c] = cv::Mat(TARGET_HEIGHT, TARGET_WIDTH, CV_32F,
                             chwMat.ptr<float>(c)).clone();
    }
    cv::Mat hwcMat;
    cv::merge(channels, hwcMat);

    // RGB转BGR
    cv::Mat bgrMat;
    cv::cvtColor(hwcMat, bgrMat, cv::COLOR_RGB2BGR);

    // 转换为uint8
    cv::Mat uint8Mat;
    bgrMat.convertTo(uint8Mat, CV_8UC3, PIXEL_NORMALIZE_FACTOR, CONVERT_OFFSET);

    // 调整回原始尺寸
    if (originalHeight != TARGET_HEIGHT || originalWidth != TARGET_WIDTH) {
        cv::resize(uint8Mat, uint8Mat, cv::Size(originalWidth, originalHeight),
                  0, 0, cv::INTER_LINEAR);
    }

    return uint8Mat;
}

bool CodeFormerPostprocess(std::vector<std::string>& fileList,
                          std::vector<TensorBuf>& outBufs,
                          std::vector<TensorDesc>& outDescs) {
    LOG(INFO) << "====== 开始后处理 ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    std::string saveBin = cfg["save_result_bin"];
    std::string saveImg = cfg["save_result_img"];

    if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
        LOG(ERROR) << "创建bin目录失败: " << saveBin;
        return false;
    }
    if (!saveImg.empty() && !CreateDirectoryRecursive(saveImg)) {
        LOG(ERROR) << "创建图像目录失败: " << saveImg;
        return false;
    }

    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const std::string& imgPath = fileList[imgIdx];
        std::string fileName = GetFileName(imgPath);
        LOG(INFO) << "处理文件: " << fileName;

        // 保存原始输出bin
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + fileName + "_output_0.bin";
            std::ofstream binFile(binPath, std::ios::binary);
            binFile.write(reinterpret_cast<char*>(outBufs[imgIdx].data.get()),
                         outBufs[imgIdx].size);
            LOG(INFO) << "保存原始输出bin to: " << binPath;
        }

        // 验证输出数据
        const float* preds = reinterpret_cast<const float*>(outBufs[imgIdx].data.get());
        size_t expectedSize = CHANNEL_NUM * TARGET_HEIGHT * TARGET_WIDTH * sizeof(float);
        if (outBufs[imgIdx].size != expectedSize) {
            LOG(ERROR) << "输出尺寸不匹配，预期: " << expectedSize
                      << "，实际: " << outBufs[imgIdx].size;
            continue;
        }

        // 获取原始图像尺寸
        cv::Mat originalImg = cv::imread(imgPath);
        if (originalImg.empty()) {
            LOG(WARNING) << "无法读取原始图像: " << imgPath;
            continue;
        }
        int originalHeight = originalImg.rows;
        int originalWidth = originalImg.cols;

        // 执行后处理
        cv::Mat resultImg = Postprocess(preds, originalHeight, originalWidth);

        // 保存结果图像
        if (!saveImg.empty()) {
            std::string imgSavePath = saveImg + "/" + fileName + "_restored.jpg";
            if (!cv::imwrite(imgSavePath, resultImg)) {
                LOG(WARNING) << "保存结果图像失败: " << imgSavePath;
            } else {
                LOG(INFO) << "已保存修复结果: " << imgSavePath;
            }
        }
    }

    LOG(INFO) << "====== 后处理完成 ======";
    return true;
}

}  // namespace CodeFormer
}  // namespace Infer