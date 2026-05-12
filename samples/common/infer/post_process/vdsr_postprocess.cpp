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

#include "vdsr_postprocess.h"
#include "infer_utils.h"
#include "log.h"
#include "post_process.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <regex>
#include <sys/stat.h>

namespace Infer {
using namespace std;

constexpr int INPUT_SIZE = 516;
constexpr float MAX_PIXEL_VALUE = 255.0f;
constexpr mode_t DIR_PERMISSIONS = 0777;
constexpr int PNG_COMPRESSION_LEVEL = 6;
constexpr double EPSILON = 1e-8;
constexpr int TOP_INDEX = 0;
constexpr int BOTTOM_INDEX = 1;
constexpr int LEFT_INDEX = 2;
constexpr int RIGHT_INDEX = 3;
constexpr int Y_CHANNEL_INDEX = 0;
constexpr int CB_CHANNEL_INDEX = 1;
constexpr int CR_CHANNEL_INDEX = 2;
constexpr int DEFAULT_BATCH_INDEX = 0;
constexpr const char* LR_PATH_PREFIX = "/LR/";
constexpr const char* GT_PATH_PREFIX = "/GT/";

static Result SaveResultWithTxt(const std::string& filePath, double mse, double pnsr)
{
    // 先获取原文件属性（可选）
    struct stat fileStat;
    stat(filePath.data(), &fileStat);

    // 打开文件并覆盖内容
    std::fstream file;
    file.open(filePath, std::ios::out | std::ios::trunc); // 使用trunc模式覆盖文件

    if (!file.is_open()) {
        // 如果文件不存在，创建它
        file.open(filePath, std::ios::out);
        if (file.is_open()) {
            file.close();
            // 确保文件已创建后再设置权限
            chmod(filePath.c_str(), DIR_PERMISSIONS);
            // 重新以写模式打开
            file.open(filePath, std::ios::out);
        } else {
            return FAILED; // 如果创建文件也失败，返回失败
        }
    }

    std::string printContent;
    printContent += "MSE: " + std::to_string(mse) + "\n";
    printContent += "PNSR: " + std::to_string(pnsr) + "dB\n";
    file << printContent;
    file.close();
    return SUCCESS;
}

cv::Mat ParseToCVMat(Infer::TensorBuf& outBuf, Infer::TensorDesc& outDesc, int batchIndex = DEFAULT_BATCH_INDEX)
{
    auto dims = outDesc.dims;
    int channels = dims[1];
    int height = dims[2];
    int width = dims[3];

    // 获取数据指针
    float* floatData = static_cast<float*>(outBuf.GetRawPtr());

    // 根据数据类型创建OpenCV Mat
    cv::Mat outputImage;

    // 计算当前batch的数据偏移
    int batchOffset = batchIndex * channels * height * width;
    float* batchData = floatData + batchOffset;

    // 创建OpenCV矩阵 (CHW -> HWC转换)
    outputImage = cv::Mat(height, width, CV_32FC1, batchData);

    return outputImage;
}

cv::Mat CropImageMat(const cv::Mat& imgTensor, const std::vector<int>& crop)
{
    int top = crop[TOP_INDEX];
    int bottom = crop[BOTTOM_INDEX];
    int left = crop[LEFT_INDEX];
    int right = crop[RIGHT_INDEX];

    // 获取图像尺寸
    int realH = imgTensor.rows; // 高度
    int realW = imgTensor.cols; // 宽度

    LOG(INFO) << "imgTensor.shape: " << imgTensor.rows << " x "
              << imgTensor.cols << " | [" << top << ", " << bottom << " , " << left << " , " << right << "]";

    // 计算裁剪区域
    int startH = top;
    int endH = realH - bottom;
    int startW = left;
    int endW = realW - right;

    // 执行裁剪 (使用cv::Rect)
    cv::Rect roi(startW, startH, endW - startW, endH - startH);

    return imgTensor(roi).clone();
}

void ProcessAndSaveOutput(Infer::TensorBuf& outBuf, Infer::TensorDesc& outDesc,
    const std::string& resultPath, const std::string& filePath)
{
    auto parts = SplitStringByChar(filePath, '/');
    std::string fileName = parts.back().substr(0, parts.back().find_last_of("."));
    fileName = SplitStringByChar(fileName, '_')[0];
    int scaleRatio = std::stoi(SplitStringByChar(filePath, '/')[SplitStringByChar(filePath, '/').size() - 3].substr(1, 1));

    auto imgPath = resultPath + "/img/X" + std::to_string(scaleRatio);
    auto txtPath = resultPath + "/txt/X" + std::to_string(scaleRatio);
    auto binPath = resultPath + "/bin/X" + std::to_string(scaleRatio);
    struct stat info;

    if (stat(txtPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(txtPath.c_str(), DIR_PERMISSIONS);
    }
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), DIR_PERMISSIONS);
    }
    if (stat(imgPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(imgPath.c_str(), DIR_PERMISSIONS);
    }

    // 解析为OpenCV Mat
    cv::Mat outputImage = ParseToCVMat(outBuf, outDesc);
    if (outputImage.empty()) {
        LOG(ERROR) << "Output image is empty!";
        return;
    }

    SaveMatToBin(outputImage, binPath + "/" + fileName + "_0.bin");

    // 数据后处理（根据模型输出特性）
    cv::Mat processedImage = outputImage;

    cv::Mat hrImage = cv::imread(std::regex_replace(filePath, std::regex(LR_PATH_PREFIX), GT_PATH_PREFIX), cv::IMREAD_COLOR);
    cv::Mat lrImage = cv::imread(filePath, cv::IMREAD_COLOR);

    std::vector<int> compressionParams;
    compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compressionParams.push_back(PNG_COMPRESSION_LEVEL);

    cv::Mat hrImageFloat;
    hrImage.convertTo(hrImageFloat, CV_32FC3);
    hrImageFloat /= MAX_PIXEL_VALUE;

    std::vector<int> preprocessSize = { INPUT_SIZE, INPUT_SIZE };
    cv::Size targetSize(
        static_cast<int>(preprocessSize[0] / static_cast<double>(scaleRatio)),
        static_cast<int>(preprocessSize[1] / static_cast<double>(scaleRatio)));

    int lrOriginImgH = lrImage.rows;
    int lrOriginImgW = lrImage.cols;

    // 计算填充
    int padTop = (targetSize.height - lrOriginImgH) / 2;
    int padBottom = targetSize.height - lrOriginImgH - padTop;
    int padLeft = (targetSize.width - lrOriginImgW) / 2;
    int padRight = targetSize.width - lrOriginImgW - padLeft;

    std::vector<int> cropPad = { padTop * scaleRatio, padBottom * scaleRatio, padLeft * scaleRatio, padRight * scaleRatio };
    cv::Mat cropImage = CropImageMat(processedImage, cropPad);

    LOG(INFO) << "cropImage.shape: " << cropImage.rows << " x "
              << cropImage.cols << " | [" << padTop * scaleRatio << ", " << padBottom * scaleRatio << " , " << padLeft * scaleRatio << " , " << padRight * scaleRatio << "]";

    // 转换颜色空间：BGR -> YCbCr
    cv::Mat ycbcrHrImg;
    ycbcrHrImg = MatlabBgr2ycbcr(hrImageFloat);

    // 分割通道
    std::vector<cv::Mat> channelsHr;
    cv::split(ycbcrHrImg, channelsHr);

    cv::Mat hrYChannel = channelsHr[Y_CHANNEL_INDEX]; // Y通道
    cv::Mat hrCrChannel = channelsHr[CR_CHANNEL_INDEX]; // Cr通道
    cv::Mat hrCbChannel = channelsHr[CB_CHANNEL_INDEX]; // Cb通道

    cv::Mat srYImage = cv::max(0.0f, cv::min(1.0f, cropImage));

    // 计算MSE
    cv::Mat diff = srYImage - hrYChannel;
    cv::Mat squaredDiff;
    cv::multiply(diff, diff, squaredDiff);
    double mse = cv::mean(squaredDiff)[0];

    LOG(INFO) << "MSE: " << mse;

    // 计算PSNR
    double psnr = 10.0 * log10(1.0 / (mse + EPSILON));
    LOG(INFO) << "PSNR: " << psnr << " dB";

    SaveResultWithTxt(txtPath + "/" + fileName + "_0.txt", mse, psnr);

    // 合并三个通道为 YCbCr 图像
    std::vector<cv::Mat> channelsSr;
    channelsSr.push_back(srYImage); // Y通道
    channelsSr.push_back(hrCbChannel); // Cb通道
    channelsSr.push_back(hrCrChannel); // Cr通道

    cv::Mat srYcbcrImage;
    cv::merge(channelsSr, srYcbcrImage); // 合并为三通道图像，类型为 CV_32FC3
    cv::imwrite(imgPath + "/" + fileName + "_result_gray.png", srYImage * MAX_PIXEL_VALUE, compressionParams);

    cv::Mat srImage = MatlabYcbcr2bgr(srYcbcrImage);
    cv::imwrite(imgPath + "/" + fileName + "_result_color.png", srImage * MAX_PIXEL_VALUE, compressionParams);
}

static void PostProcess(std::vector<Infer::TensorBuf>& outBufs, std::vector<Infer::TensorDesc>& outDescs,
    const std::vector<std::string>& fileList)
{
    auto filePath = fileList[0];
    LOG(INFO) << filePath;

    // 获取保存文件路径和文件名
    auto parts = SplitStringByChar(filePath, '/');
    std::string fileName = parts.back().substr(0, parts.back().find_last_of("."));
    std::string resultPath = "../out/result";
    struct stat info;
    mode_t oldUmask = umask(0);

    if (stat(resultPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(resultPath.c_str(), DIR_PERMISSIONS);
        LOG(INFO) << "Create result directory successfully: " << resultPath;
    }

    std::string txtPath = resultPath + "/txt";
    if (stat(txtPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(txtPath.c_str(), DIR_PERMISSIONS);
    }

    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), DIR_PERMISSIONS);
    }

    std::string imgPath = resultPath + "/img";
    if (stat(imgPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(imgPath.c_str(), DIR_PERMISSIONS);
    }

    ProcessAndSaveOutput(outBufs[0], outDescs[0], resultPath, filePath);
    umask(oldUmask);
}

bool VDSRPostProcess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    PostProcess(tensorBufs, tensorDescs, fileList);
    return true;
}

} // namespace Infer