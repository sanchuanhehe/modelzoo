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


#include "vit_preprocess.h"
#include "log.h"
#include "dev_interface_adapter.h"
#include <opencv2/opencv.hpp>
#include "PillowResize/PillowResize.hpp"
#include <string>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <fstream>
#include "utils.h"

namespace Infer {
static cv::Mat PillowResizeKeepRatio(const cv::Mat& img, int targetSize, int32_t interpolation = PillowResize::INTERPOLATION_BILINEAR)
{
    if (targetSize <= 0) {
        return img;
    }
    int width = img.cols;   // 图像宽度
    int height = img.rows;  // 图像高度
    // 检查是否已满足目标尺寸条件
    if ((width <= height && width == targetSize) || (height <= width && height == targetSize)) {
        return img;
    }
    
    cv::Mat targetImg;
    if (width < height) {
        int newWidth = targetSize;
        int newHeight = static_cast<int>(height * static_cast<float>(targetSize) / width);
        targetImg = PillowResize::resize(img, cv::Size(newWidth, newHeight), interpolation);
    } else {
        int newHeight = targetSize;
        int newWidth = static_cast<int>(width * static_cast<float>(targetSize) / height);
        targetImg = PillowResize::resize(img, cv::Size(newWidth, newHeight), interpolation);
    }

    return targetImg;
}

static cv::Mat CenterCrop(const cv::Mat& img, int cropSize)
{
    if (cropSize <= 0) {
        return img;
    }
    int width = img.cols;
    int height = img.rows;
    
    // 计算裁剪区域
    int startX = std::max(0, (width - cropSize) / 2);
    int startY = std::max(0, (height - cropSize) / 2);
    int cropWidth = std::min(cropSize, width - startX);
    int cropHeight = std::min(cropSize, height - startY);
    
    return img(cv::Rect(startX, startY, cropWidth, cropHeight)).clone();
}

static cv::Mat HwcToChw(const cv::Mat& hwcImg)
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

static int SelectDtype(const std::string& typeStr)
{
    static const std::unordered_map<std::string, int> typeMap = {
        {"uint8",   CV_8U},   // 无符号8位整数
        {"int8",    CV_8S},   // 有符号8位整数
        {"int16",   CV_16S},  // 有符号16位整数
        {"int32",   CV_32S},  // 有符号32位整数
        {"float32", CV_32F}   // 单精度浮点数
    };
    auto it = typeMap.find(typeStr);
    return (it != typeMap.end()) ? it->second : CV_8U; 
}

static std::string GetFileExtension(const std::string& filename)
{
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos) {
        return "";
    }
    std::string ext = filename.substr(dotPos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

static bool ParseYuvDimensions(const std::string& filename, int& width, int& height)
{
    // 正则表达式匹配 "_数字*数字" 模式
    std::regex pattern(R"(_(\d+)\*(\d+))");
    std::smatch matches;
    
    if (std::regex_search(filename, matches, pattern) && matches.size() == 3) {
        try {
            width = std::stoi(matches[1].str());
            height = std::stoi(matches[2].str());
            return (width > 0 && height > 0);
        } catch (...) {
            return false;
        }
    }
    return false;
}

static bool Yuv420SpToRgb(const unsigned char* yuvData, int width, int height, cv::Mat& rgbMat)
{
    // 输入参数有效性检查
    if (yuvData == nullptr || width <= 0 || height <= 0) {
        return false;
    }
    
    int ySize = width * height;
    int uvSize = ySize / 2;

    // 创建YUV I420格式的Mat
    cv::Mat yuvI420(height + height/2, width, CV_8UC1);
    if (yuvI420.empty()) {
        return false;
    }
    // 复制Y通道
    memcpy(yuvI420.data, yuvData, ySize);
    // 分离VU交错数据为U和V平面
    unsigned char* uPtr = yuvI420.data + ySize;
    unsigned char* vPtr = uPtr + uvSize/2;
    const unsigned char* vuPtr = yuvData + ySize;
    
    for (int i = 0; i < uvSize / 2; ++i) {
        vPtr[i] = vuPtr[2 * i];     // V分量
        uPtr[i] = vuPtr[2 * i + 1]; // U分量
    }

    cv::cvtColor(yuvI420, rgbMat, cv::COLOR_YUV2RGB_I420);
    // 检查输出矩阵是否有效
    return !rgbMat.empty();
}

static bool ConvertImageToRgb(const std::string& filename, cv::Mat& rgbMat)
{
    std::string ext = GetFileExtension(filename);
    // 处理常见图像格式: JPEG/JPG/PNG/BMP
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp") {
        // 使用OpenCV直接读取
        cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
        if (img.empty()) {
            LOG(ERROR) << "fail to imread, path : " << filename;
            return false;
        }
        
        // 转换为RGB格式 (OpenCV默认读取为BGR)
        if (img.channels() == 3) {
            cv::cvtColor(img, rgbMat, cv::COLOR_BGR2RGB);
        } else if (img.channels() == 4) {
            // 处理带Alpha通道的图像
            cv::cvtColor(img, rgbMat, cv::COLOR_BGRA2RGB);
        } else if (img.channels() == 1) {
            // 处理灰度图像
            cv::cvtColor(img, rgbMat, cv::COLOR_GRAY2RGB);
        } else {
            return false; // 不支持的通道数
        }
    } else if (ext == "yuv420sp") {
        int yuvWidth = 0;
        int yuvHeight = 0;
        // 从文件名解析宽高
        if (!ParseYuvDimensions(filename, yuvWidth, yuvHeight)) {
            return false; // 无法解析宽高
        }
        
        // 计算YUV数据大小
        int ySize = yuvWidth * yuvHeight;
        int uvSize = ySize / 2;
        int totalSize = ySize + uvSize;
        
        // 读取YUV数据
        std::vector<unsigned char> yuvData(totalSize);
        std::ifstream file(filename, std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(yuvData.data()), totalSize)) {
            return false;
        }
        
        // 转换为RGB
        Yuv420SpToRgb(yuvData.data(), yuvWidth, yuvHeight, rgbMat);
    }
    else {
        LOG(ERROR) << "fail to convert image to rgb";
        return false; // 不支持的格式
    }
    return true;
}

/**
 * @brief 图像归一化函数（对应Python版本逻辑）
 * @param pic 输入图像（OpenCV Mat，格式为 HWC，通道顺序BGR/RGB均可，数据类型建议float32）
 * @param avgs 各通道的均值数组（顺序需和图像通道对应）
 * @param varis 各通道的方差数组（顺序需和图像通道对应）
 * @return 归一化后的图像（Mat，格式和输入一致）
 */
static void Normalization(cv::Mat& pic, const std::vector<float>& avgs, const std::vector<float>& varis) {
    // 1. 输入合法性校验
    if (pic.empty()) {
        throw std::invalid_argument("Error: Input image is empty!");
    }
    if (avgs.size() != varis.size()) {
        throw std::invalid_argument("Error: avgs and varis size must be equal!");
    }
    const int channelNum = pic.channels();
    if ((int)avgs.size() != channelNum) {
        throw std::invalid_argument(
            "Error: avgs/varis size(" + std::to_string(avgs.size()) + 
            ") not match image channels(" + std::to_string(channelNum) + ")!"
        );
    }
    // 2. 第二步：按通道执行 (val - avg) / vari 归一化（原地修改）
    // 优化：指针遍历，高效操作
    for (int chn = 0; chn < channelNum; ++chn) {
        const float avg = avgs[chn];
        const float vari = varis[chn];
        
        // 遍历每一行
        for (int row = 0; row < pic.rows; ++row) {
            float* rowPtr = pic.ptr<float>(row); // 获取当前行指针
            // 遍历每一列的当前通道
            for (int col = 0; col < pic.cols; ++col) {
                const int idx = col * channelNum + chn; // 计算像素索引
                // 原地修改：(缩放后的值 - 均值) / 方差
                rowPtr[idx] = (rowPtr[idx] - avg) / vari;
            }
        }
    }
}

static cv::Mat ProcessSingleImage(const std::string& imgPath, const VitPrerocessOptions& options) 
{
    cv::Mat rgbMat;
    if (!ConvertImageToRgb(imgPath, rgbMat)) { 
        LOG(ERROR) << "fail to convert image to rgb";
        return {};
    }
    cv::Mat image = CenterCrop(PillowResizeKeepRatio(rgbMat, options.resizeValue, options.interpolationValue), options.centerCropValue);
    // image.convertTo(image, SelectDtype(options.dtypeStr));
    image.convertTo(image, SelectDtype("float32"), 1.0 / 255.0); // 临时修改适配 vit dpico, 转f32, 并做归一化
    std::vector<float> avgs = {0.5f, 0.5f, 0.5f};  
    std::vector<float> varis = {0.5f, 0.5f, 0.5f};
    Normalization(image, avgs, varis);

    if (options.chwFlag) {
        image = HwcToChw(image);
    }
    return image.clone(); // 确保内存连续
}

bool VitPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs, const VitPrerocessOptions& options)
{
    for (size_t i = 0; i < fileList.size(); ++i) {
        cv::Mat image = ProcessSingleImage(fileList[i], options);
        if (image.empty()) {
            LOG(ERROR) << "fail to preprocess " << fileList[i];
            return false;
        }
        size_t imageSize = image.total() * image.elemSize();
        if (!image.isContinuous()) {
            image = image.clone();
        }
        if (!PadDataToTensorBuf(image.data, imageSize, tensorDescs[i], tensorBufs[i])) {
            LOG(ERROR) << "fail to pad data";
            return false;
        }
    }
    return true;
}
}