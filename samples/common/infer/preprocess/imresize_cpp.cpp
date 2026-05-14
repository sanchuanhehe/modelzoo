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

#include "imresize_cpp.h"
#include "PillowResize/PillowResize.hpp"
#include "dev_interface_adapter.h"
#include "image_process.h"
#include "log.h"
#include "string"
#include "utils.h"
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <regex>
#include <string>
#include <unordered_map>

using namespace std;

namespace Infer {
using Image = std::vector<std::vector<std::vector<double>>>; // [c][h][w]
using Image2D = std::vector<std::vector<double>>; // [h][w]
constexpr double ZERO_TOLERANCE = 1e-12;

double Cubic(double x)
{
    // 定义常量
    const double SMOOTH_RANGE_END = 1.0; // 第一段平滑区间的终点
    const double RAMP_RANGE_END = 2.0; // 第二段斜坡区间的终点

    // 第一段多项式系数 (|x| <= 1.0)
    const double COEFF_A1 = 1.5; // x³项系数
    const double COEFF_B1 = -2.5; // x²项系数
    const double COEFF_C1 = 1.0; // 常数项

    // 第二段多项式系数 (1.0 < |x| <= 2.0)
    const double COEFF_A2 = -0.5; // x³项系数
    const double COEFF_B2 = 2.5; // x²项系数
    const double COEFF_C2 = -4.0; // x项系数
    const double COEFF_D2 = 2.0; // 常数项

    double absx = std::abs(x);
    double absx2 = absx * absx;
    double absx3 = absx2 * absx;

    if (absx <= SMOOTH_RANGE_END) {
        return COEFF_A1 * absx3 + COEFF_B1 * absx2 + COEFF_C1;
    } else if (absx <= RAMP_RANGE_END) {
        return COEFF_A2 * absx3 + COEFF_B2 * absx2 + COEFF_C2 * absx + COEFF_D2;
    } else {
        return 0.0;
    }
}

// 计算权重和索引的核心函数
struct WeightsIndicesResult {
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<int>> indices;
    int symLenS;
    int symLenE;
};

WeightsIndicesResult CalculateWeightsIndices(
    int inLength,
    int outLength,
    double scale,
    int kernelWidth,
    bool antialiasing)
{
    double adjustedKernelWidth = static_cast<double>(kernelWidth);

    // 如果下采样且启用抗锯齿，调整核宽度
    if (scale < 1.0 && antialiasing) {
        adjustedKernelWidth = adjustedKernelWidth / scale;
    }

    // 生成输出空间坐标
    std::vector<double> x(outLength);
    for (int i = 0; i < outLength; ++i) {
        x[i] = static_cast<double>(i + 1); // 从1开始，与Matlab一致
    }

    // 计算输入空间坐标
    std::vector<double> u(outLength);
    for (int i = 0; i < outLength; ++i) {
        u[i] = x[i] / scale + 0.5 * (1.0 - 1.0 / scale);
    }

    // 计算左边界
    std::vector<double> left(outLength);
    for (int i = 0; i < outLength; ++i) {
        left[i] = std::floor(u[i] - adjustedKernelWidth / 2.0);
    }

    // 计算参与计算的像素数
    int p = static_cast<int>(std::ceil(adjustedKernelWidth)) + 2;

    // 创建索引矩阵
    std::vector<std::vector<int>> indices(outLength, std::vector<int>(p));
    for (int i = 0; i < outLength; ++i) {
        for (int j = 0; j < p; ++j) {
            indices[i][j] = static_cast<int>(left[i] + static_cast<double>(j));
        }
    }

    // 计算到中心的距离
    std::vector<std::vector<double>> distanceToCenter(outLength, std::vector<double>(p));
    for (int i = 0; i < outLength; ++i) {
        for (int j = 0; j < p; ++j) {
            distanceToCenter[i][j] = u[i] - static_cast<double>(indices[i][j]);
        }
    }

    // 应用三次卷积核计算权重
    std::vector<std::vector<double>> weights(outLength, std::vector<double>(p));
    if (scale < 1.0 && antialiasing) {
        for (int i = 0; i < outLength; ++i) {
            for (int j = 0; j < p; ++j) {
                weights[i][j] = scale * Cubic(distanceToCenter[i][j] * scale);
            }
        }
    } else {
        for (int i = 0; i < outLength; ++i) {
            for (int j = 0; j < p; ++j) {
                weights[i][j] = Cubic(distanceToCenter[i][j]);
            }
        }
    }

    // 归一化权重
    for (int i = 0; i < outLength; ++i) {
        double weightsSum = 0.0;
        for (int j = 0; j < p; ++j) {
            weightsSum += weights[i][j];
        }

        // 避免除零错误
        if (std::abs(weightsSum) > ZERO_TOLERANCE) {
            for (int j = 0; j < p; ++j) {
                weights[i][j] /= weightsSum;
            }
        }
    }

    // 移除全零列（只考虑第一列和最后一列）
    bool removeFirst = false;
    bool removeLast = false;

    // 检查第一列
    double firstColSum = 0.0;
    for (int i = 0; i < outLength; ++i) {
        firstColSum += std::abs(weights[i][0]);
    }
    if (std::abs(firstColSum) < ZERO_TOLERANCE) {
        removeFirst = true;
    }

    // 检查最后一列
    double lastColSum = 0.0;
    for (int i = 0; i < outLength; ++i) {
        lastColSum += std::abs(weights[i][p - 1]);
    }
    if (std::abs(lastColSum) < ZERO_TOLERANCE) {
        removeLast = true;
    }

    // 调整权重和索引矩阵
    int newP = p;
    if (removeFirst)
        newP--;
    if (removeLast)
        newP--;

    std::vector<std::vector<double>> finalWeights(outLength, std::vector<double>(newP));
    std::vector<std::vector<int>> finalIndices(outLength, std::vector<int>(newP));

    for (int i = 0; i < outLength; ++i) {
        int newJ = 0;
        for (int j = 0; j < p; ++j) {
            if ((removeFirst && j == 0) || (removeLast && j == p - 1)) {
                continue;
            }
            finalWeights[i][newJ] = weights[i][j];
            finalIndices[i][newJ] = indices[i][j];
            newJ++;
        }
    }

    // 计算对称填充长度
    int minIndex = finalIndices[0][0];
    int maxIndex = finalIndices[0][0];
    for (int i = 0; i < outLength; ++i) {
        for (int j = 0; j < newP; ++j) {
            if (finalIndices[i][j] < minIndex)
                minIndex = finalIndices[i][j];
            if (finalIndices[i][j] > maxIndex)
                maxIndex = finalIndices[i][j];
        }
    }

    int symLenS = -minIndex + 1;
    int symLenE = maxIndex - inLength;

    // 调整索引
    for (int i = 0; i < outLength; ++i) {
        for (int j = 0; j < newP; ++j) {
            finalIndices[i][j] += symLenS - 1;
        }
    }

    return { finalWeights, finalIndices, symLenS, symLenE };
}

// 辅助函数：安全的向量访问
template <typename T>
T SafeGet(const std::vector<T>& vec, size_t index, const T& defaultVal = T())
{
    return (index < vec.size()) ? vec[index] : defaultVal;
}

template <typename T>
T SafeGet2d(const std::vector<std::vector<T>>& vec, size_t i, size_t j, const T& defaultVal = T())
{
    if (i < vec.size() && j < vec[i].size()) {
        return vec[i][j];
    }
    return defaultVal;
}

// 向量点积
double DotProduct(const std::vector<double>& a, const std::vector<double>& b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// 安全的图像创建函数
Image CreateImage(int channels, int height, int width, double initVal = 0.0)
{
    Image img;
    img.reserve(channels);
    for (int c = 0; c < channels; ++c) {
        Image2D channel;
        channel.reserve(height);
        for (int h = 0; h < height; ++h) {
            std::vector<double> row(width, initVal);
            channel.push_back(std::move(row));
        }
        img.push_back(std::move(channel));
    }
    return img;
}

// 图像缩放函数
Image Imresize(const Image& image, double scaleFactor, bool antialiasing = true)
{
    int inC = image.size();
    int inH = image[0].size();
    int inW = image[0][0].size();

    // 计算输出尺寸
    int outH = static_cast<int>(std::ceil(inH * scaleFactor));
    int outW = static_cast<int>(std::ceil(inW * scaleFactor));

    int kernelWidth = 4;

    // 获取权重和索引
    auto weightsIndicesH = CalculateWeightsIndices(inH, outH, scaleFactor, kernelWidth, antialiasing);
    auto weightsIndicesW = CalculateWeightsIndices(inW, outW, scaleFactor, kernelWidth, antialiasing);

    const auto& weightsH = weightsIndicesH.weights;
    const auto& indicesH = weightsIndicesH.indices;
    int symLenHs = weightsIndicesH.symLenS;
    int symLenHe = weightsIndicesH.symLenE;

    const auto& weightsW = weightsIndicesW.weights;
    const auto& indicesW = weightsIndicesW.indices;
    int symLenWs = weightsIndicesW.symLenS;
    int symLenWe = weightsIndicesW.symLenE;

    // 验证权重和索引的尺寸
    const size_t outHSize = static_cast<size_t>(outH);
    const size_t outWSize = static_cast<size_t>(outW);
    if (weightsH.size() != outHSize || indicesH.size() != outHSize ||
        weightsW.size() != outWSize || indicesW.size() != outWSize) {
        throw std::runtime_error("Weights and indices size mismatch");
    }

    // 处理H维度 - 对称填充
    int augmentedH = inH + symLenHs + symLenHe;
    if (augmentedH <= 0) {
        throw std::runtime_error("Invalid augmented height");
    }

    auto imgAug = CreateImage(inC, augmentedH, inW, 0.0);

    // 复制原始图像到中间位置
    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < inH; ++i) {
            for (int j = 0; j < inW; ++j) {
                imgAug[c][i + symLenHs][j] = SafeGet2d(image[c], i, j);
            }
        }
    }

    // 上边界对称填充
    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < symLenHs; ++i) {
            int srcRow = symLenHs - 1 - i;
            if (srcRow < 0)
                srcRow = 0; // 边界保护
            if (srcRow >= inH)
                srcRow = inH - 1;

            for (int j = 0; j < inW; ++j) {
                imgAug[c][i][j] = SafeGet2d(image[c], srcRow, j);
            }
        }
    }

    // 下边界对称填充
    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < symLenHe; ++i) {
            int srcRow = inH - 1 - i;
            if (srcRow < 0)
                srcRow = 0; // 边界保护
            if (srcRow >= inH)
                srcRow = inH - 1;

            for (int j = 0; j < inW; ++j) {
                imgAug[c][symLenHs + inH + i][j] = SafeGet2d(image[c], srcRow, j);
            }
        }
    }

    // H维度插值
    auto out1 = CreateImage(inC, outH, inW, 0.0);

    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < outH; ++i) {
            if (static_cast<size_t>(i) >= indicesH.size())
                continue;

            int idx = SafeGet(indicesH[i], 0, 0);
            const auto& currentWeights = weightsH[i];
            int kernelWidthH = currentWeights.size();

            for (int j = 0; j < inW; ++j) {
                // 提取kernel_width_h个像素
                std::vector<double> pixels(kernelWidthH, 0.0);
                for (int k = 0; k < kernelWidthH; ++k) {
                    int rowIdx = idx + k;
                    if (rowIdx >= 0 && rowIdx < augmentedH) {
                        pixels[k] = SafeGet2d(imgAug[c], rowIdx, j);
                    }
                    // 如果越界，保持为0（已初始化）
                }
                // 计算加权和
                out1[c][i][j] = DotProduct(pixels, currentWeights);
            }
        }
    }

    // 处理W维度 - 对称填充
    int augmentedW = inW + symLenWs + symLenWe;
    if (augmentedW <= 0) {
        throw std::runtime_error("Invalid augmented width");
    }

    auto out1Aug = CreateImage(inC, outH, augmentedW, 0.0);

    // 复制中间结果
    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < inW; ++j) {
                out1Aug[c][i][j + symLenWs] = SafeGet2d(out1[c], i, j);
            }
        }
    }

    // 左边界对称填充
    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < symLenWs; ++j) {
                int srcCol = symLenWs - 1 - j;
                if (srcCol < 0)
                    srcCol = 0;
                if (srcCol >= inW)
                    srcCol = inW - 1;

                out1Aug[c][i][j] = SafeGet2d(out1[c], i, srcCol);
            }
        }
    }

    // 右边界对称填充
    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < outH; ++i) {
            for (int j = 0; j < symLenWe; ++j) {
                int srcCol = inW - 1 - j;
                if (srcCol < 0)
                    srcCol = 0;
                if (srcCol >= inW)
                    srcCol = inW - 1;

                out1Aug[c][i][symLenWs + inW + j] = SafeGet2d(out1[c], i, srcCol);
            }
        }
    }

    // W维度插值
    auto out2 = CreateImage(inC, outH, outW, 0.0);

    for (int c = 0; c < inC; ++c) {
        for (int i = 0; i < outW; ++i) {
            if (static_cast<size_t>(i) >= indicesW.size())
                continue;

            int idx = SafeGet(indicesW[i], 0, 0);
            const auto& currentWeights = weightsW[i];
            int kernelWidthW = currentWeights.size();

            for (int j = 0; j < outH; ++j) {
                // 提取kernel_width_w个像素
                std::vector<double> pixels(kernelWidthW, 0.0);
                for (int k = 0; k < kernelWidthW; ++k) {
                    int colIdx = idx + k;
                    if (colIdx >= 0 && colIdx < augmentedW) {
                        pixels[k] = SafeGet2d(out1Aug[c], j, colIdx);
                    }
                    // 如果越界，保持为0（已初始化）
                }
                // 计算加权和
                out2[c][j][i] = DotProduct(pixels, currentWeights);
            }
        }
    }

    return out2;
}

Image MatToVector(const cv::Mat& mat)
{
    // 检查输入矩阵是否有效
    if (mat.empty()) {
        return Image();
    }

    // 确保矩阵是连续的内存块
    cv::Mat continuousMat = mat.isContinuous() ? mat : mat.clone();

    // 获取矩阵尺寸
    int channels = continuousMat.channels();
    int height = continuousMat.rows;
    int width = continuousMat.cols;

    // 创建结果向量
    Image result(channels,
        std::vector<std::vector<double>>(height,
            std::vector<double>(width)));

    // 根据数据类型进行转换
    switch (continuousMat.depth()) {
    case CV_8U: {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    result[c][h][w] = static_cast<double>(
                        continuousMat.ptr<cv::Vec3b>(h)[w][c]);
                }
            }
        }
        break;
    }
    case CV_32F: {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    result[c][h][w] = static_cast<double>(
                        continuousMat.ptr<cv::Vec3f>(h)[w][c]);
                }
            }
        }
        break;
    }
    case CV_64F: {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    result[c][h][w] = continuousMat.ptr<cv::Vec3d>(h)[w][c];
                }
            }
        }
        break;
    }
    default: {
        // 对于其他数据类型，先转换为CV_64F
        cv::Mat doubleMat;
        continuousMat.convertTo(doubleMat, CV_64F);
        return MatToVector(doubleMat);
    }
    }

    return result;
}

cv::Mat VectorToMat(const Image& image)
{
    if (image.empty() || image[0].empty() || image[0][0].empty()) {
        return cv::Mat();
    }

    int channels = image.size();
    int height = image[0].size();
    int width = image[0][0].size();

    cv::Mat result(height, width, CV_64FC(channels));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                if (channels == 1) {
                    result.at<double>(h, w) = image[c][h][w];
                } else {
                    result.at<cv::Vec3d>(h, w)[c] = image[c][h][w];
                }
            }
        }
    }

    return result;
}

cv::Mat MatlabImresize(const cv::Mat& mat, const int& r)
{
    Image testImage3d = MatToVector(mat);
    Image resultUpscale = Imresize(testImage3d, r, true);
    return VectorToMat(resultUpscale);
}
}
