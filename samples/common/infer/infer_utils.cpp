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

#include "infer_utils.h"
#include <algorithm>
#include <errno.h>
#include <fstream>
#include <getopt.h>
#include <limits.h>
#include <log.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

namespace Infer {
using namespace std;
constexpr float MAX_PIXEL_VALUE = 255.0f;

std::vector<std::string> SplitStringByChar(const std::string& s, char delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = s.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(s.substr(start, end - start));
        start = end + 1;
        end = s.find(delimiter, start);
    }
    result.push_back(s.substr(start));
    return result;
}

bool SaveMatToBin(const cv::Mat& mat, const std::string& filename)
{
    // return true;
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        return false;
    }

    // 写入矩阵基本信息
    int rows = mat.rows;
    int cols = mat.cols;
    int type = mat.type();

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&type), sizeof(int));

    // 写入矩阵数据
    ofs.write(reinterpret_cast<const char*>(mat.data), mat.total() * mat.elemSize());

    ofs.close();
    return true;
}

cv::Mat MatlabBgr2ycbcr(const cv::Mat& bgrFloat)
{
    // 定义 BGR 到 YCbCr 的转换矩阵（针对 BGR 输入）
    const cv::Matx33f bgr2ycbcrMatrix(
        24.966f, 112.0f, -18.214f,
        128.553f, -74.203f, -93.786f,
        65.481f, -37.797f, 112.0f);

    cv::Matx33f bgr2ycbcrMatrixT = bgr2ycbcrMatrix.t(); // 转置

    const cv::Vec3f bias(16.0f, 128.0f, 128.0f);

    // image 是 BGR 顺序的 CV_32FC3 图像
    cv::Mat resultImage(bgrFloat.size(), bgrFloat.type());

    // 对每个像素进行矩阵乘法
    for (int y = 0; y < bgrFloat.rows; ++y) {
        for (int x = 0; x < bgrFloat.cols; ++x) {
            cv::Vec3f pixel = bgrFloat.at<cv::Vec3f>(y, x);

            // 手动矩阵乘法以保持精度
            float b = pixel[0], g = pixel[1], r = pixel[2];
            float yChannel = b * bgr2ycbcrMatrixT(0, 0) + g * bgr2ycbcrMatrixT(0, 1) + r * bgr2ycbcrMatrixT(0, 2);
            float cbChannel = b * bgr2ycbcrMatrixT(1, 0) + g * bgr2ycbcrMatrixT(1, 1) + r * bgr2ycbcrMatrixT(1, 2);
            float crChannel = b * bgr2ycbcrMatrixT(2, 0) + g * bgr2ycbcrMatrixT(2, 1) + r * bgr2ycbcrMatrixT(2, 2);

            // 加上偏置
            yChannel += bias[0];
            cbChannel += bias[1];
            crChannel += bias[2];

            // 除以255
            yChannel /= MAX_PIXEL_VALUE;
            cbChannel /= MAX_PIXEL_VALUE;
            crChannel /= MAX_PIXEL_VALUE;

            resultImage.at<cv::Vec3f>(y, x) = cv::Vec3f(yChannel, cbChannel, crChannel);
        }
    }

    return resultImage;
}

cv::Mat MatlabYcbcr2bgr(const cv::Mat& ycbcrFloat)
{
    // 定义 YCbCr 到 BGR 的转换矩阵（针对 YCbCr 输入）
    ycbcrFloat *= MAX_PIXEL_VALUE;
    const cv::Matx33f ycbcr2bgrMatrix(
        0.00456621f, 0.00456621f, 0.00456621f,
        0.00791071f, -0.00153632f, 0.0f,
        0.0f, -0.00318811f, 0.00625893f);

    cv::Matx33f ycbcr2bgrMatrixT = ycbcr2bgrMatrix.t(); // 转置

    const cv::Vec3f bias(-276.836f, 135.576f, -222.921f);

    // 假设 image 是 RGB 顺序的 CV_32FC3 图像
    cv::Mat result_image(ycbcrFloat.size(), ycbcrFloat.type());

    // 对每个像素进行矩阵乘法
    for (int y = 0; y < ycbcrFloat.rows; ++y) {
        for (int x = 0; x < ycbcrFloat.cols; ++x) {
            cv::Vec3f pixel = ycbcrFloat.at<cv::Vec3f>(y, x);

            float yc = pixel[0], cb = pixel[1], cr = pixel[2];
            float bChannel = yc * ycbcr2bgrMatrixT(0, 0) + cb * ycbcr2bgrMatrixT(0, 1) + cr * ycbcr2bgrMatrixT(0, 2);
            float gChannel = yc * ycbcr2bgrMatrixT(1, 0) + cb * ycbcr2bgrMatrixT(1, 1) + cr * ycbcr2bgrMatrixT(1, 2);
            float rChannel = yc * ycbcr2bgrMatrixT(2, 0) + cb * ycbcr2bgrMatrixT(2, 1) + cr * ycbcr2bgrMatrixT(2, 2);
            bChannel *= MAX_PIXEL_VALUE;
            gChannel *= MAX_PIXEL_VALUE;
            rChannel *= MAX_PIXEL_VALUE;
            // 加上偏置
            bChannel += bias[0];
            gChannel += bias[1];
            rChannel += bias[2];

            // 除以255
            bChannel /= MAX_PIXEL_VALUE;
            gChannel /= MAX_PIXEL_VALUE;
            rChannel /= MAX_PIXEL_VALUE;

            result_image.at<cv::Vec3f>(y, x) = cv::Vec3f(bChannel, gChannel, rChannel);
        }
    }

    return result_image;
}

}