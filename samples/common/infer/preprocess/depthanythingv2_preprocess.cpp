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

#include "depthanythingv2_preprocess.h"

#include <cstring>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include "log.h"

using json = nlohmann::json;

namespace Infer {

static constexpr int DEPTH_INPUT_SIZE = 518;
static constexpr int ENSURE_MULTIPLE_OF = 14;
static constexpr int BYTE_BIT_NUM = 8;

static Result ReadImgToBufDpico(const cv::Mat& mat, const TensorDesc& desc,
    TensorBuf& inBuf)
{
    char* bufPtr = static_cast<char*>(inBuf.GetRawPtr());
    size_t width = desc.dims[desc.dimCount - 1];
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;
    for (size_t i = 0; i < desc.dims[1] * desc.dims[2]; i++) {
        memcpy(bufPtr + i * desc.defaultStride, mat.data + i * lineSize, lineSize);
    }
    return SUCCESS;
}

static Result ReadImgToBufDlite(const cv::Mat& mat, const TensorDesc& desc, TensorBuf& inBuf)
{
    size_t matTotalBytes = mat.total() * mat.elemSize();
    if (desc.defaultStride == 0) {
        memcpy(inBuf.GetRawPtr(), mat.data, matTotalBytes);
    } else {
        size_t loopTimes = 1;
        for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
            loopTimes *= desc.dims[loop];
        }
        size_t width = desc.dims[desc.dimCount - 1];
        size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;

        for (size_t loop = 0; loop < loopTimes; loop++) {
            char* dst = static_cast<char*>(inBuf.GetRawPtr()) + loop * desc.defaultStride;
            const char* src = reinterpret_cast<const char*>(mat.ptr(static_cast<int>(loop)));
            memcpy(dst, src, lineSize);
        }
    }
    return SUCCESS;
}

static double RoundHalfToEven(double d) {
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
}

static int ConstrainToMultipleOf(double x, int multiple_of=14, int min_val=0, int max_val=-1) {
    int y = static_cast<int>(RoundHalfToEven(x / multiple_of) * multiple_of);
    if (max_val > 0 && y > max_val) {
        y = static_cast<int>(std::floor(x / multiple_of) * multiple_of);
    }
    if (y < min_val) {
        y = static_cast<int>(std::ceil(x / multiple_of) * multiple_of);
    }
    return y;
}

// Equivalent to torchvision's Transform & Resize configuration configured in Python
static cv::Mat DepthAnythingResizeInput(const cv::Mat& img, int& top_pad, int& bottom_pad, int& left_pad, int& right_pad, int input_size = DEPTH_INPUT_SIZE)
{
    int h = img.rows;
    int w = img.cols;

    double scale_height = static_cast<double>(input_size) / h;
    double scale_width  = static_cast<double>(input_size) / w;

    if (scale_width < scale_height) {
        scale_height = scale_width;
    } else {
        scale_width = scale_height;
    }

    int new_h = ConstrainToMultipleOf(scale_height * h, ENSURE_MULTIPLE_OF, 0, input_size);
    int new_w = ConstrainToMultipleOf(scale_width * w, ENSURE_MULTIPLE_OF, 0, input_size);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_CUBIC);

    // Calculate padding
    int pad_h = input_size - new_h;
    int pad_w = input_size - new_w;

    top_pad = pad_h / 2;
    bottom_pad = pad_h - top_pad;
    left_pad = pad_w / 2;
    right_pad = pad_w - left_pad;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top_pad, bottom_pad, left_pad, right_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return padded;
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
    return chwMat; // 3 × (H*W) Continuous CHW Matrix
}

bool DepthAnythingV2Preprocess(std::vector<std::string>& fileList,
                               std::vector<TensorBuf>& tensorBufs,
                               std::vector<TensorDesc>& tensorDescs,
                               bool isDpico)
{
    LOG(INFO) << "Detected " << (isDpico ? "dpico " : "dlite ");
    LOG(INFO) << "Processing file num: " << fileList.size();

    for (size_t i = 0; i < fileList.size(); ++i) {
        std::string imgPath = fileList[i];
        LOG(INFO) << "imgPath: " << imgPath;
        cv::Mat im0 = cv::imread(imgPath);

        if (im0.empty()) {
            LOG(ERROR) << "Failed to read image: " << imgPath;
            return false;
        }

        // Resize & Pad
        int top_pad = 0, bottom_pad = 0, left_pad = 0, right_pad = 0;
        cv::Mat processed = DepthAnythingResizeInput(im0, top_pad, bottom_pad, left_pad, right_pad, DEPTH_INPUT_SIZE);

        // BGR -> RGB
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);

        // Convert to Float & Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        cv::Mat floatImg;
        processed.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        cv::split(floatImg, channels);
        channels[0] = (channels[0] - 0.485f) / 0.229f; // R
        channels[1] = (channels[1] - 0.456f) / 0.224f; // G
        channels[2] = (channels[2] - 0.406f) / 0.225f; // B
        cv::merge(channels, floatImg);

        cv::Mat finalImg = HwcToChw(floatImg);
        if (!finalImg.isContinuous()) {
            finalImg = finalImg.clone();
        }

        if (isDpico) {
            ReadImgToBufDpico(finalImg, tensorDescs[i], tensorBufs[i]);
        } else {
            ReadImgToBufDlite(finalImg, tensorDescs[i], tensorBufs[i]);
        }
        LOG(INFO) << "read end: " << imgPath;
    }

    LOG(INFO) << "DepthAnythingV2 PreProcessing completed successfully!";
    return true;
}

} // namespace Infer
