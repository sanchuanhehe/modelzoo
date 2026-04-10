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

#include "depthanythingv2_postprocess.h"

#include "log.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

using json = nlohmann::json;

namespace Infer {

static constexpr int BYTE_BIT_NUM = 8;
static constexpr int DEPTH_OUTPUT_SIZE = 518;

static Result ExtractTensorData(Infer::TensorBuf &outBuf, Infer::TensorDesc &outDesc, std::vector<float> &noStrideBuf)
{
    int64_t lastDim = outDesc.dims[outDesc.dimCount - 1];
    size_t dataSize = outDesc.typeSize / BYTE_BIT_NUM;
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = outBuf.size / outBuf.stride;
    size_t strideElemNum = outBuf.stride / dataSize;
    size_t invalidSize = outBuf.stride - lastDimSize;
    float *outData = static_cast<float *>(outBuf.GetRawPtr());
    size_t outSize = outBuf.size / dataSize;

    if (invalidSize == 0) {
        LOG(INFO) << "not stride invalid space, return directly";
        std::vector<float> tempBuf(outData, outData + outSize);
        noStrideBuf.assign(tempBuf.begin(), tempBuf.end());
    } else {
        std::vector<float> tempBuf(outData, outData + outSize);
        for (size_t i = 0; i < loopNum; ++i) {
            size_t offset = i * strideElemNum;
            for (int64_t index = 0; index < lastDim; index++) {
                noStrideBuf.push_back(tempBuf[offset + index]);
            }
        }
    }

    if (noStrideBuf.empty()) {
        LOG(ERROR) << "noStrideBuf malloc fail";
        return FAILED;
    }

    return SUCCESS;
}

// GetPadShapeData logic has been removed as we dynamically calculate it now
static void CreateResultDirs()
{
    std::string resultPath = "../out/result";

    mode_t oldUmask = umask(0);
    struct stat info;
    if (stat(resultPath.c_str(), &info) != 0)
        mkdir(resultPath.c_str(), 0777);

    std::string jpgPath = resultPath + "/jpg";
    if (stat(jpgPath.c_str(), &info) != 0)
        mkdir(jpgPath.c_str(), 0777);

    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0)
        mkdir(binPath.c_str(), 0777);
    umask(oldUmask);
}

static cv::Mat VisualizeDepthMap(
    const cv::Mat &depthMap, const std::string &outputPath = "", int colormap = cv::COLORMAP_JET)
{
    CV_Assert(depthMap.type() == CV_32FC1 || depthMap.type() == CV_64FC1);
    cv::Mat processedDepth = depthMap.clone();

    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();

    if (processedDepth.type() == CV_32FC1) {
        for (int i = 0; i < processedDepth.rows; ++i) {
            for (int j = 0; j < processedDepth.cols; ++j) {
                float value = processedDepth.at<float>(i, j);
                if (std::isnan(value) || std::isinf(value))
                    processedDepth.at<float>(i, j) = 0.0f;
                else if (value > 0) {
                    minVal = std::min(minVal, static_cast<double>(value));
                    maxVal = std::max(maxVal, static_cast<double>(value));
                }
            }
        }
    }

    if (minVal > maxVal) {
        minVal = 0;
        maxVal = 1;
    }

    cv::Mat depthNormalized;
    if (maxVal > minVal) {
        processedDepth.convertTo(
            depthNormalized, CV_32F, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    } else {
        depthNormalized = cv::Mat::zeros(processedDepth.size(), CV_32F);
    }

    cv::Mat depth8bit;
    depthNormalized.convertTo(depth8bit, CV_8U);
    cv::Mat depthColored;
    cv::applyColorMap(depth8bit, depthColored, colormap);

    if (!outputPath.empty()) {
        cv::imwrite(outputPath, depthColored);
        LOG(INFO) << "save jpg sucess";
    }
    return depthColored;
}

bool DepthAnythingV2Postprocess(
    std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs)
{
    LOG(INFO) << "start post";
    const std::string &filePath = fileList[0];

    CreateResultDirs();

    std::vector<float> temp;
    if (tensorDescs[0].defaultStride == 0) {
        tensorDescs[0].defaultStride =
            tensorDescs[0].dims[tensorDescs[0].dimCount - 1] * tensorDescs[0].typeSize / BYTE_BIT_NUM;
        tensorBufs[0].stride = tensorDescs[0].defaultStride;
    }
    if (ExtractTensorData(tensorBufs[0], tensorDescs[0], temp) != SUCCESS)
        return false;

    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");
    std::string fileName = filePath.substr(start + 1, end - start - 1);

    cv::Mat im0 = cv::imread(filePath);
    if (im0.empty()) {
        LOG(ERROR) << "Failed to read image in postprocess";
        return false;
    }
    int h = im0.rows;
    int w = im0.cols;

    int inputSize = DEPTH_OUTPUT_SIZE;
    int ensureMultipleOf = 14;

    auto roundHalfToEven = [](double d) {
        double i;
        double f = std::modf(d, &i);
        if (std::abs(f) == 0.5) {
            if (std::fmod(std::abs(i), 2.0) == 1.0) {
                return i + (i > 0 ? 1.0 : -1.0);
            } else {
                return i;
            }
        }
        return std::round(d);
    };

    auto constrainToMultipleOf = [&](double x, int multipleOf, int minVal, int maxVal) {
        int y = static_cast<int>(roundHalfToEven(x / multipleOf) * multipleOf);
        if (maxVal > 0 && y > maxVal) {
            y = static_cast<int>(std::floor(x / multipleOf) * multipleOf);
        }
        if (y < minVal) {
            y = static_cast<int>(std::ceil(x / multipleOf) * multipleOf);
        }
        return y;
    };

    double scaleHeight = static_cast<double>(inputSize) / h;
    double scaleWidth = static_cast<double>(inputSize) / w;

    if (scaleWidth < scaleHeight) {
        scaleHeight = scaleWidth;
    } else {
        scaleWidth = scaleHeight;
    }

    int newH = constrainToMultipleOf(scaleHeight * h, ensureMultipleOf, 0, inputSize);
    int newW = constrainToMultipleOf(scaleWidth * w, ensureMultipleOf, 0, inputSize);

    int padH = inputSize - newH;
    int padW = inputSize - newW;

    int topPad = padH / 2;
    int bottomPad = padH - topPad;
    int leftPad = padW / 2;
    int rightPad = padW - leftPad;

    // (518x518 Float32 matrix)
    cv::Mat inferenceResult(DEPTH_OUTPUT_SIZE, DEPTH_OUTPUT_SIZE, CV_32FC1, temp.data());
    int cropHeight = DEPTH_OUTPUT_SIZE - bottomPad - topPad;
    int cropWidth = DEPTH_OUTPUT_SIZE - rightPad - leftPad;

    cv::Rect roi(leftPad, topPad, cropWidth, cropHeight);
    cv::Mat croppedArray = inferenceResult(roi);

    // Bilinear Interpolation
    cv::Mat resizedDepth;
    cv::resize(croppedArray, resizedDepth, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    std::string binName = "../out/result/bin/" + fileName + "_0.bin";
    std::ofstream fout(binName, std::ios::out | std::ios::binary);
    if (fout.good()) {
        fout.write(reinterpret_cast<const char *>(resizedDepth.data), resizedDepth.total() * resizedDepth.elemSize());
        fout.close();
    } else {
        LOG(ERROR) << "Failed to save bin file";
    }

    std::string jpgName = "../out/result/jpg/" + fileName + ".jpg";
    VisualizeDepthMap(resizedDepth, jpgName);

    return true;
}

}  // namespace Infer
