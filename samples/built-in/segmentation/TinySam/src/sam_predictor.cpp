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

#include "sam_predictor.h"
#include <memory>
#include <tuple>
#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include "PillowResize/PillowResize.hpp"
#include "model.h"
#include "log.h"

using namespace cv;
using namespace std;
using namespace Infer;

constexpr int MASK_H = 448;
constexpr int MASK_W = 448;

// 原有函数：HWC格式Mat转CHW格式Mat（3 × HW）
static cv::Mat HwcToChw(const cv::Mat& hwcImg)
{
    CV_Assert(!hwcImg.empty() && hwcImg.channels() == 3);
    std::vector<cv::Mat> channels;
    cv::split(hwcImg, channels);

    // 创建CHW格式Mat（3行，HW列，与输入同深度）
    cv::Mat chwMat(3, hwcImg.rows * hwcImg.cols, hwcImg.depth());
    chwMat.setTo(cv::Scalar(0));

    for (int channelIdx = 0; channelIdx < 3; ++channelIdx) {
        // 将单通道H×W展平为1×HW，复制到CHW Mat的对应行
        cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
        flatChannel.copyTo(chwMat.row(channelIdx));
    }

    return chwMat; // 返回 3 × (H*W) 的CHW矩阵
}

static tuple<Mat, float, pair<int, int>, pair<int, int>> preprocess(const Mat& imgIn, const pair<int, int>& imageSize) {
    // 1. 获取原始图像尺寸 (H, W)
    int oriH = imgIn.rows;
    int oriW = imgIn.cols;
    pair<int, int> oriImgShape = {oriH, oriW};

    // 2. 计算等比例缩放因子
    float scaleH = static_cast<float>(imageSize.first) / oriH;
    float scaleW = static_cast<float>(imageSize.second) / oriW;
    float scale = min(scaleH, scaleW);

    // 3. 计算目标尺寸（四舍五入）
    int targetH = static_cast<int>(scale * oriH + 0.5f);
    int targetW = static_cast<int>(scale * oriW + 0.5f);
    pair<int, int> targetImgShape = {targetH, targetW};

    // 4. 双线性插值缩放图像
    Mat imgResized = PillowResize::resize(imgIn, Size(targetW, targetH), PillowResize::INTERPOLATION_BILINEAR);

    // 5. 转换为float32并归一化 (img - mean) / std
    Mat imgF32;
    imgResized.convertTo(imgF32, CV_32FC3);
    const Scalar mean(123.675f, 116.28f, 103.53f);  // RGB均值
    const Scalar std(58.395f, 57.12f, 57.375f);     // RGB标准差
    subtract(imgF32, mean, imgF32);
    divide(imgF32, std, imgF32);

    // 6. 创建padding画布（全零矩阵）
    Mat canvas = Mat::zeros(Size(imageSize.second, imageSize.first), CV_32FC3);
    imgF32.copyTo(canvas(Rect(0, 0, targetW, targetH)));  // 复制到画布左上角

    return make_tuple(canvas, scale, targetImgShape, oriImgShape);
}

vector<float> boxesScale(const vector<float>& boxes, float scaleH, float scaleW) {
    if (boxes.size() != 4) {
        throw runtime_error("Boxes vector length must be multiple of 4 (x1,y1,x2,y2 per box)");
    }
    vector<float> boxesScaled = boxes;
    boxesScaled[0] *= scaleW;// x1
    boxesScaled[2] *= scaleW;// x2
    boxesScaled[1] *= scaleH;// y1
    boxesScaled[3] *= scaleH;// y2

    return boxesScaled;
}

bool SAMBoxPredictor::loadModel(const vector<string>& ModelPaths) {
    if (ModelPaths.size() != 3) {
        LOG(ERROR) << "num of model paths should be 3";
        return false;
    }
    if (imageEncoder_->Load(ModelPaths[0], ModelType::TinySam)){
        LOG(ERROR) << "fail to load image encoder";
        return false;
    }
    if (promptEncoder_->Load(ModelPaths[1], ModelType::TinySam)){
        LOG(ERROR) << "fail to load prompt encoder";
        return false;
    }
    if (maskDecoder_->Load(ModelPaths[2], ModelType::TinySam)){
        LOG(ERROR) << "fail to load mask decoder";
        return false;
    }
    imageInputs_.clear();
    promptInputs_.clear();
    maskInputs_.clear();
    auto modelInfo = imageEncoder_->GetModelInfo();
    size_t inputNum = modelInfo.first.size();
    for (size_t i = 0; i < inputNum; ++i) {
        imageInputs_.emplace_back(modelInfo.first[i].defaultSize, modelInfo.first[i].defaultStride);
    }

    modelInfo = promptEncoder_->GetModelInfo();
    inputNum = modelInfo.first.size();
    for (size_t i = 0; i < inputNum; ++i) {
        promptInputs_.emplace_back(modelInfo.first[i].defaultSize, modelInfo.first[i].defaultStride);
    }

    modelInfo = maskDecoder_->GetModelInfo();
    inputNum = modelInfo.first.size();
    for (size_t i = 0; i < inputNum; ++i) {
        maskInputs_.emplace_back(modelInfo.first[i].defaultSize, modelInfo.first[i].defaultStride);
    }
    return true;
}

bool SAMBoxPredictor::setImage(const Mat& imgIn) {
    Mat canvas;
    tie(canvas, scale_, targetImgShape_, originImgShape_) = preprocess(imgIn, imageSize_);
    scaleH_ = static_cast<float>(targetImgShape_.first) / originImgShape_.first;
    scaleW_ = static_cast<float>(targetImgShape_.second) / originImgShape_.second;

    Mat chwMat = HwcToChw(canvas);
    if (!chwMat.isContinuous()) {
        chwMat = chwMat.clone();
    }
    size_t imageSize = chwMat.total() * chwMat.elemSize();
    if (DevMemcpy(imageInputs_[0].GetRawPtr(), imageInputs_[0].size, chwMat.data, imageSize) != 0) {
        LOG(ERROR) << "fail to memcpy image data";
        return false;
    }

    auto imageOutputs = imageEncoder_->Infer(imageInputs_);
    if (imageOutputs.size() != 1 || maskInputs_.size() != 2) {
        LOG(ERROR) << "fail to encode image";
        return false;
    }
    maskInputs_[0] = imageOutputs[0];
    return true;
}

pair<float, Mat> SAMBoxPredictor::predict(const vector<float>& boxes) {
    // 缩放边界框到特征图尺寸
    vector<float> boxesScaled = boxesScale(boxes, scaleH_, scaleW_);
    vector<TensorBuf> inputBoxBufs;
    *(static_cast<float*>(promptInputs_[0].GetRawPtr())) = boxesScaled[0];
    *(static_cast<float*>(promptInputs_[0].GetRawPtr()) + 1) = boxesScaled[1];
    *(static_cast<float*>(promptInputs_[1].GetRawPtr())) = boxesScaled[2];
    *(static_cast<float*>(promptInputs_[1].GetRawPtr()) + 1) = boxesScaled[3];

    // 编码边界框（模型提示编码器推理）
    auto promptOutputs = promptEncoder_->Infer(promptInputs_);
    if (promptOutputs.size() != 1 || maskInputs_.size() != 2) {
        LOG(ERROR) << "fail to encode image";
        return {};
    }
    maskInputs_[1] = promptOutputs[0];

    // 解码掩码（模型掩码解码器推理）
    auto maskOutput = maskDecoder_->Infer(maskInputs_);
    TensorBuf maskTensor = maskOutput[1];
    static int i = 0;
    ofstream fout("./result/" + to_string(i), ios::binary | ios::out);
    if (!fout.is_open()) {
        return {};
    }
    i++;
    fout.write((char*)maskTensor.GetRawPtr(), maskTensor.size);
    cv::Mat maskMat(MASK_H, MASK_W, CV_32FC1);
    std::memcpy(maskMat.data, maskTensor.GetRawPtr(), MASK_H * MASK_W * sizeof(float));
    cv::Mat clipMat = maskMat(cv::Range(0, targetImgShape_.first), cv::Range(0, targetImgShape_.second)).clone();
    cv::Mat resizeMat;
    cv::resize(clipMat, resizeMat, Size(originImgShape_.second, originImgShape_.first), 
                0, 0, cv::INTER_LINEAR);
    cv::Mat mask_binary;
    cv::threshold(resizeMat, resizeMat, 0.0, 255.0, cv::THRESH_BINARY); // 大于0的像素值设置为255 
    resizeMat.convertTo(resizeMat, CV_8UC1);
    float iou = *(static_cast<float*>(maskOutput[0].GetRawPtr()));

    return make_pair(iou, resizeMat);
}