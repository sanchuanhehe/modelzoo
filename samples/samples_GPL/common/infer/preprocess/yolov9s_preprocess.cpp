/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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

#include "yolov9s_preprocess.h"
#include "log.h"
#include "utils.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

static constexpr int BYTE_BIT_NUM = 8;
static constexpr int FP16_BIT_NUM = 16;
static constexpr int FP32_BIT_NUM = 32;
static constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
static constexpr int CHANNEL_COUNT = 3;
static constexpr int PAD_COLOR_VAL = 114;
static constexpr mode_t FILE_PERMISSION_0755 = 0755;

namespace Infer {

// 预处理配置
struct PreprocessConfig {
    cv::Size imgSize;
    std::string saveBin;
    bool isDpico;
};

static void ConvertFp32ToFp16Inplace(const float* srcData, size_t dataCount, uint16_t* dstData)
{
    for (size_t i = 0; i < dataCount; ++i) {
        dstData[i] = FloatToHalf(srcData[i]);
    }
}

static std::vector<uint16_t> ConvertFp32ToFp16Vector(const float* srcData, size_t dataCount)
{
    std::vector<uint16_t> fp16Data(dataCount);
    ConvertFp32ToFp16Inplace(srcData, dataCount, fp16Data.data());
    return fp16Data;
}

static int64_t GetLoopTimes(const TensorDesc& desc)
{
    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; ++loop) {
        loopTimes *= desc.dims[loop];
    }
    return loopTimes;
}

static Result ReadImgToBufDpico(const cv::Mat& mat, const TensorDesc&,
    TensorBuf& inBuf)
{
    size_t matTotalBytes = mat.total() * mat.elemSize();
    memcpy(inBuf.GetRawPtr(), mat.data, matTotalBytes);
    return SUCCESS;
}

static Result ReadImgToBufDlite(const cv::Mat& mat, const TensorDesc& desc,
    TensorBuf& inBuf)
{
    const float* srcData = mat.ptr<float>();
    const bool isFp32 = (desc.typeSize == FP32_BIT_NUM);
    const bool isFp16 = (desc.typeSize == FP16_BIT_NUM);

    if (desc.defaultStride == 0) {
        if (isFp32) {
            memcpy(inBuf.GetRawPtr(), srcData, desc.defaultSize);
        } else {
            std::vector<uint16_t> fp16Data = ConvertFp32ToFp16Vector(srcData, mat.total());
            memcpy(inBuf.GetRawPtr(), fp16Data.data(), desc.defaultSize);
        }
        return SUCCESS;
    }

    int64_t loopTimes = GetLoopTimes(desc);
    int64_t width = desc.dims[desc.dimCount - 1];
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;
    size_t dstStride = inBuf.stride == 0 ? desc.defaultStride : inBuf.stride;
    char* dstBase = static_cast<char*>(inBuf.GetRawPtr());
    for (int64_t loop = 0; loop < loopTimes; loop++) {
        void* destPtr = dstBase + loop * dstStride;
        const float* srcPtr = srcData + loop * width;
        if (isFp32) {
            memcpy(destPtr, srcPtr, lineSize);
        } else {
            ConvertFp32ToFp16Inplace(srcPtr, width, static_cast<uint16_t*>(destPtr));
        }
    }
    return SUCCESS;
}

// Letterbox函数实现
static cv::Mat LetterBox(const cv::Mat& img, const cv::Size& target_size,
    bool scaleup)
{
    cv::Size shape = img.size();

    // 计算缩放比例
    double r = std::min(static_cast<double>(target_size.width) / shape.width,
        static_cast<double>(target_size.height) / shape.height);
    r = std::min(r, 1.0);

    // 计算未填充的尺寸
    cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
        static_cast<int>(std::round(shape.height * r)));

    // 计算填充量
    double w = (target_size.width - unpad.width) / 2.0f;
    double h = (target_size.height - unpad.height) / 2.0f;
    cv::Mat resized;
    if (shape != unpad) {
        cv::resize(img, resized, unpad, 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img.clone();
    }
    // 计算填充边界
    int top = static_cast<int>(std::round(h - 0.1));
    int bottom = static_cast<int>(std::round(h + 0.1));
    int left = static_cast<int>(std::round(w - 0.1));
    int right = static_cast<int>(std::round(w + 0.1));

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(PAD_COLOR_VAL, PAD_COLOR_VAL, PAD_COLOR_VAL));

    return padded;
}

static cv::Mat HwcToChw(const cv::Mat& hwcImg)
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

    return chwMat; // 返回 3 × (高度*宽度) 的CHW矩阵
}

// 提取文件名（不含路径和后缀）
static std::string GetFileName(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ? 
           path.substr(start) : path.substr(start, end - start);
}

// 预缩放：按比例缩放使长边等于目标尺寸
static void PreScaleImage(cv::Mat& img, const cv::Size& targetSize)
{
    int h0 = img.rows;
    int w0 = img.cols;
    // 等比缩放比：取宽/高方向的较小值，确保缩放后不超出目标框（preR == 1.0 时跳过）
    double preR = std::min(static_cast<double>(targetSize.width) / w0,
        static_cast<double>(targetSize.height) / h0);
    if (preR != 1.0) { // 整数相除，仅当维度完全相等时精确为 1.0，直接比较安全
        // ceil 后用 min 截断，防止浮点向上取整导致新尺寸超出目标框 1 个像素
        int newW = std::min(static_cast<int>(std::ceil(w0 * preR)), targetSize.width);
        int newH = std::min(static_cast<int>(std::ceil(h0 * preR)), targetSize.height);
        cv::resize(img, img, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    }
}

// 图像变换流水线：预缩放 → letterbox → BGR2RGB → 归一化 → HWC转CHW
static cv::Mat TransformImage(cv::Mat& img, const cv::Size& targetSize)
{
    // 预缩放
    PreScaleImage(img, targetSize);

    // 应用letterbox
    cv::Mat processed = LetterBox(img, targetSize, false);

    // BGR到RGB：反转通道顺序
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

    // 像素值归一化到 [0, 1] 区间
    cv::Mat floatImg;
    processed.convertTo(floatImg, CV_32FC3, 1.0 / PIXEL_NORMALIZE_FACTOR);

    // 将图像格式从 HWC 转换为 CHW
    cv::Mat chwImg = HwcToChw(floatImg);

    if (!chwImg.isContinuous()) {
        chwImg = chwImg.clone(); // 克隆矩阵以获得连续内存
    }
    return chwImg;
}

// 保存预处理结果为bin文件
static void SavePreprocessBin(const cv::Mat& chwImg, const std::string& binPath, const cv::Size& imgSize)
{
    std::ofstream binFile(binPath, std::ios::binary);
    if (binFile.is_open()) {
        size_t dataSize = imgSize.width * imgSize.height * CHANNEL_COUNT * sizeof(float);
        binFile.write((char*)chwImg.data, dataSize);
        binFile.close();
        chmod(binPath.c_str(), FILE_PERMISSION_0755);
        LOG(INFO) << "Saved preprocess bin: " << binPath;
    } else {
        LOG(WARNING) << "Failed to save preprocess bin: " << binPath;
    }
}

// 统一的数据写入派发
static void ReadImgToBuf(const cv::Mat& mat, const TensorDesc& desc, TensorBuf& buf, bool isDpico)
{
    if (isDpico) {
        ReadImgToBufDpico(mat, desc, buf);
    } else {
        ReadImgToBufDlite(mat, desc, buf);
    }
}

// 处理单张图像：读取 → 变换 → 写入buffer → 可选保存bin
static bool ProcessSingleImage(const std::string& imgPath, TensorDesc& desc,
    TensorBuf& buf, const PreprocessConfig& cfg)
{
    cv::Mat im0 = cv::imread(imgPath);
    if (im0.empty()) {
        LOG(ERROR) << "Read image failed: " << imgPath;
        return false;
    }

    cv::Mat chwImg = TransformImage(im0, cfg.imgSize);
    ReadImgToBuf(chwImg, desc, buf, cfg.isDpico);

    if (!cfg.saveBin.empty()) {
        std::string binPath = cfg.saveBin + "/" + GetFileName(imgPath) + ".bin";
        SavePreprocessBin(chwImg, binPath, cfg.imgSize);
    }
    return true;
}

bool Yolov9sPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs,
    std::vector<TensorDesc>& tensorDescs, bool isDpico)
{
    LOG(INFO) << "Detected " << (isDpico ? "dpico " : "dlite ");
    LOG(INFO) << "Processing file num: " << fileList.size();

    // 从cfg.txt读取配置
    auto cfgMap = ReadCfgFile("../data/cfg.txt");
    PreprocessConfig cfg;
    cfg.imgSize = cv::Size(std::stoi(cfgMap["img_width"]), std::stoi(cfgMap["img_height"]));
    cfg.saveBin = cfgMap["save_preprocess_bin"];
    cfg.isDpico = isDpico;

    // 按需创建目录
    if (!cfg.saveBin.empty() && !CreateDirectoryRecursive(cfg.saveBin)) {
        LOG(ERROR) << "Create preprocess bin dir failed";
        return false;
    }

    // 处理每个图像
    for (size_t i = 0; i < fileList.size(); ++i) {
        LOG(INFO) << "imgPath: " << fileList[i];
        if (!ProcessSingleImage(fileList[i], tensorDescs[i], tensorBufs[i], cfg)) {
            LOG(ERROR) << "PreProcessing failed!";
            return false;
        }
        LOG(INFO) << "read end: " << fileList[i];
    }

    LOG(INFO) << "PreProcessing completed successfully!";
    return true;
}

} // namespace Infer
