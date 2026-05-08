/*
 * Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

#include "clip_postprocess.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace Infer {
namespace Clip {

constexpr int TARGET_WIDTH = 512;
constexpr int TARGET_HEIGHT = 512;
constexpr int CHANNEL_NUM = 3;
constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
constexpr float CLAMP_MIN = -1.0f;
constexpr float CLAMP_MAX = 1.0f;
constexpr float DENORM_OFFSET = 1.0f;
constexpr float DENORM_SCALE = 2.0f;
constexpr float CONVERT_OFFSET = 0.5f;
constexpr int FEATDIM = 512;     // 特征维度（对应Python的512）
constexpr int TEMPLATE_NUM = 11; // 每个类别对应11个模板
constexpr float EPS = 1e-8f;     // 数值稳定性阈值

std::string GetFileName(const std::string &path)
{
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ? path.substr(start) : path.substr(start, end - start);
}

cv::Mat Postprocess(const float *preds, int originalHeight, int originalWidth)
{
    // 转换为CHW格式Mat (3, H, W)
    cv::Mat chwMat(CHANNEL_NUM, TARGET_HEIGHT * TARGET_WIDTH, CV_32F);
    memcpy(chwMat.data, preds,
            CHANNEL_NUM * TARGET_HEIGHT * TARGET_WIDTH * sizeof(float));

    // 裁剪到[-1, 1]并映射到[0, 1]
    for (size_t i = 0; i < chwMat.total(); ++i) {
        float val = chwMat.at<float>(i);
        val = std::max(CLAMP_MIN, std::min(CLAMP_MAX, val));
        chwMat.at<float>(i) = (val + DENORM_OFFSET) / DENORM_SCALE;
    }

    // CHW转HWC
    std::vector<cv::Mat> channels(CHANNEL_NUM);
    for (int c = 0; c < CHANNEL_NUM; ++c) {
        channels[c] = cv::Mat(TARGET_HEIGHT, TARGET_WIDTH, CV_32F, chwMat.ptr<float>(c)).clone();
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
        cv::resize(uint8Mat, uint8Mat, cv::Size(originalWidth, originalHeight), 0, 0, cv::INTER_LINEAR);
    }

    return uint8Mat;
}

void L2Normalize(std::vector<std::vector<float>> &feats)
{
    for (auto &row : feats) {
        // 使用double进行累加以提高精度
        double normSq = 0.0;
        for (float val : row) {
            normSq += static_cast<double>(val) * static_cast<double>(val);
        }
        double norm = std::sqrt(normSq); // double开方

        if (norm > EPS) {
            float normF = static_cast<float>(norm); // 最后转回float
            for (float &val : row) {
                val /= normF;
            }
        }
    }
}

/**
 * @brief 按列平均（完全对齐PyTorch: mean(dim=0)）
 * @param feats 输入特征矩阵（shape: [M, FEATDIM]，每行长度必须=FEATDIM）
 * @return 平均后的特征（shape: [FEATDIM]）
 */
std::vector<float> MeanByCol(std::vector<std::vector<float>> &feats)
{
    std::vector<float> avgFeat(FEATDIM, 0.0f);
    int M = feats.size();
    if (M == 0) {
        LOG(ERROR) << "Warning: feats is empty, return zero vector";
        return avgFeat;
    }

    // 使用高精度累加
    for (int d = 0; d < FEATDIM; ++d) {
        double sum = 0.0;

        for (const auto &row : feats) {
            if (row.size() != FEATDIM) {
                throw std::invalid_argument("All rows must have length " + std::to_string(FEATDIM));
            }
            sum += static_cast<double>(row[d]);
        }

        avgFeat[d] = static_cast<float>(sum / M);
    }

    return avgFeat;
}

/**
 * @brief 单个向量L2归一化
 * @param vec 输入向量（shape: [FEATDIM]）
 */
void NormalizeSingleVec(std::vector<float> &vec)
{
    // 严格校验：向量长度必须=FEATDIM
    if (vec.size() != FEATDIM) {
        LOG(ERROR) << "Error: 向量长度=" << vec.size() << "，必须=" << FEATDIM;
        return;
    }

    // 计算L2范数
    double normSq = 0.0;
    for (float val : vec) {
        normSq += val * val;
    }
    double norm = std::sqrt(normSq);

    // PyTorch逻辑：范数>EPS时归一化，否则保留原值
    if (norm > EPS) {
        float normF = static_cast<float>(norm); // 最后转回float
        for (float &val : vec) {
            val /= normF;
        }
    }
}

std::vector<std::vector<float>>
StackDim(std::vector<std::vector<float>> &vectors) // 每个vector是[C]
{
    int n = vectors.size();    // 向量个数
    int C = vectors[0].size(); // 每个向量的维度

    // 结果形状：[C, n]
    std::vector<std::vector<float>> result(C, std::vector<float>(n));

    // 填充数据：按照PyTorch的内存布局
    for (int i = 0; i < n; ++i) {     // 遍历每个向量
        for (int j = 0; j < C; ++j) {   // 遍历每个维度
            result[j][i] = vectors[i][j]; // result[j][i] = 第i个向量的第j维
        }
    }

    return result;
}

const float TEMPERATURE = 100.0f; // 温度系数

/**
 * @brief 矩阵乘法 + 温度缩放
 * @param imageFeats 图像特征（shape: [batchSize, featDim]，行优先存储）
 * @param classifier 分类器权重（shape: [featDim, numClasses]，行优先存储）
 * @param batchSize 批次大小（必须显式传入，不能忽略）
 * @param featDim 特征维度（如512）
 * @param numClasses 类别数（如100）
 * @return 未归一化的logits（shape: [batchSize, numClasses]，行优先存储）
 */
std::vector<float>
MatmulWithScale(const std::vector<float> &imageFeats,
                const std::vector<std::vector<float>> &classifier,
                int batchSize, int featDim, int numClasses)
{
    // 初始化输出logits：batchSize * numClasses 个元素，初始值0
    std::vector<float> logits(batchSize * numClasses, 0.0f);

    // 矩阵乘法核心：C[bs][j] = sum_k (A[bs][k] * B[k][j]) * TEMPERATURE
    for (int bs = 0; bs < batchSize; ++bs) { // 遍历每个样本（批次维度）
        // 当前样本的imageFeats起始索引：bs * featDim
        int imgStartIdx = bs * featDim;

        for (int k = 0; k < featDim; ++k) { // 遍历特征维度（公共维度）
            float a_val = imageFeats[imgStartIdx + k];
            if (fabs(a_val) < EPS)
                continue; // 优化：跳过0值，不影响结果

            for (int j = 0; j < numClasses; ++j) { // 遍历类别维度
                float b_val = classifier[k][j];
                // 当前logits的索引：bs * numClasses + j
                logits[bs * numClasses + j] += a_val * b_val;
            }
        }

        // 温度缩放：先求和再乘100.0（和Python完全一致）
        for (int j = 0; j < numClasses; ++j) {
            logits[bs * numClasses + j] *= TEMPERATURE;
        }
    }

    return logits;
}

/**
 * @brief Softmax归一化（完全对齐Python：softmax(dim=-1)）
 * @param logits 输入logits（shape: [batchSize, numClasses]，行优先存储）
 * @param batchSize 批次大小
 * @param numClasses 类别数
 * @return 归一化后的概率分布（shape: [batchSize, numClasses]，行优先存储）
 */
std::vector<float> Softmax(const std::vector<float> &logits, int batchSize, int numClasses)
{
    std::vector<float> probs(logits.size(), 0.0f);

    // 遍历每个样本，单独对其numClasses维度做Softmax
    for (int bs = 0; bs < batchSize; ++bs) {
        // 当前样本的logits/probs起始索引
        int startIdx = bs * numClasses;
        const float *logitPtr = &logits[startIdx];
        float *probPtr = &probs[startIdx];

        // 步骤1：找当前样本logits的最大值（数值稳定化，避免exp溢出）
        float maxVal = logitPtr[0];
        for (int j = 1; j < numClasses; ++j) {
            if (logitPtr[j] > maxVal) {
                maxVal = logitPtr[j];
            }
        }

        // 步骤2：计算exp(logit - maxVal)并求和
        float sumExp = 0.0f;
        for (int j = 0; j < numClasses; ++j) {
            probPtr[j] = std::exp(logitPtr[j] - maxVal);
            sumExp += probPtr[j];
        }

        // 步骤3：归一化（避免除0）
        sumExp = (sumExp < EPS) ? EPS : sumExp;
        for (int j = 0; j < numClasses; ++j) {
            probPtr[j] /= sumExp;
        }
    }

    return probs;
}

bool ClipImgPostprocess(std::vector<std::string> &fileList,
                        std::vector<Infer::TensorBuf> &outBufs,
                        std::vector<Infer::TensorDesc> &outDescs)
{
    LOG(INFO) << "====== 开始后处理 ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    std::string saveBin = cfg["img_result"];

    mode_t old = umask(0);
    if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
        LOG(ERROR) << "创建bin目录失败: " << saveBin;
        return false;
    }
    umask(old);

    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const std::string &imgPath = fileList[imgIdx];
        std::string fileName = GetFileName(imgPath);
        LOG(INFO) << "处理文件: " << fileName;

        // 保存原始输出bin
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + fileName + "_output_0.bin";
            std::ofstream binFile(binPath, std::ios::binary);
            binFile.write(reinterpret_cast<char *>(outBufs[imgIdx].data.get()), outBufs[imgIdx].size);
            LOG(INFO) << "保存原始输出bin to: " << binPath;
        }
    }
    return true;
}

bool ClipTxtPostprocess(std::vector<std::string> &fileList,
                        std::vector<TensorBuf> &outBufs,
                        std::vector<TensorDesc> &outDescs)
{
    LOG(INFO) << "====== 开始后处理 ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    std::string saveBin = cfg["txt_result"];
    mode_t old = umask(0);
    if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
        LOG(ERROR) << "创建bin目录失败: " << saveBin;
        return false;
    }
    umask(old);
    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const std::string &imgPath = fileList[imgIdx];
        std::string fileName = GetFileName(imgPath);
        LOG(INFO) << "处理文件: " << fileName;

        // 保存原始输出bin
        if (!saveBin.empty()) {
            std::string binPath = saveBin + "/" + fileName + "_output_0.bin";
            std::ofstream binFile(binPath, std::ios::binary);
            binFile.write(reinterpret_cast<char *>(outBufs[imgIdx].data.get()), outBufs[imgIdx].size);
            LOG(INFO) << "保存原始输出bin to: " << binPath;
        }
    }

    LOG(INFO) << "====== 后处理完成 ======";
    return true;
}

} // namespace Clip
} // namespace Infer