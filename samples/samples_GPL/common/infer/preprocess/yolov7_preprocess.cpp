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
 *
 *
 *
 */

#include "yolov7_preprocess.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "log.h"

using json = nlohmann::json;
constexpr int BYTE_BIT_NUM = 8;
constexpr int yolov7InputSize = 640;
namespace Infer {
Result ReadImgToBufDlite(const cv::Mat& mat, const TensorDesc& desc,
                         TensorBuf& inBuf) {
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
      char* dst =
          static_cast<char*>(inBuf.GetRawPtr()) + loop * desc.defaultStride;
      const char* src =
          reinterpret_cast<const char*>(mat.ptr(static_cast<int>(loop)));
      memcpy(dst, src, lineSize);
    }
  }

  return SUCCESS;
}

Result ReadImgToBufDpico(const cv::Mat& mat, const TensorDesc& desc,
                         TensorBuf& inBuf) {
  size_t matTotalBytes = mat.total() * mat.elemSize();
  size_t bufTotalBytes =
      desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
  for (size_t i = 0; i < desc.dimCount - 1; i++) {
    bufTotalBytes *= desc.dims[i];
  }

  char* bufPtr = static_cast<char*>(inBuf.GetRawPtr());
  memcpy(bufPtr, mat.data, matTotalBytes);
  return SUCCESS;
}

// Letterbox函数实现
cv::Mat LetterBox(const cv::Mat& img, const cv::Size& target_size,
                  bool scaleup) {
  cv::Size shape = img.size();

  // 计算缩放比例
  double r = std::min(static_cast<double>(target_size.width) / shape.width,
                      static_cast<double>(target_size.height) / shape.height);
  r = std::min(r, 1.0);

  LOG(INFO) << "read r: " << r;
  // 计算未填充的尺寸
  cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
                 static_cast<int>(std::round(shape.height * r)));

  LOG(INFO) << "unpad r: " << unpad.width << "  " << unpad.height;
  // 计算填充量
  double w = (target_size.width - unpad.width) / 2.0f;
  double h = (target_size.height - unpad.height) / 2.0f;
  LOG(INFO) << "w r: " << w << "  " << h;
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
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  return padded;
}

cv::Mat HwcToChw(const cv::Mat& hwcImg) {
  CV_Assert(!hwcImg.empty() && hwcImg.channels() == 3);
  std::vector<cv::Mat> channels;
  cv::split(hwcImg, channels);

  cv::Mat chwMat(3, hwcImg.rows * hwcImg.cols, hwcImg.depth());
  chwMat = cv::Scalar(0);

  for (int channelIdx = 0; channelIdx < 3; ++channelIdx) {
    cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
    flatChannel.copyTo(chwMat.row(channelIdx));
  }

  return chwMat;  // 返回 3 × (高度*宽度) 的CHW矩阵
}

bool Yolov7Preprocess(std::vector<std::string>& fileList,
                      std::vector<TensorBuf>& tensorBufs,
                      std::vector<TensorDesc>& tensorDescs, bool isDpico) {
  if (isDpico) {
    LOG(INFO) << "detected dpico ";
  } else {
    LOG(INFO) << "detected dlite ";
  }
  LOG(INFO) << "Processing file num: " << fileList.size();
  // 处理每个图像
  std::vector<int> imgSize = {yolov7InputSize, yolov7InputSize};
  for (size_t i = 0; i < fileList.size(); ++i) {
    std::string imgPath = fileList[i];
    LOG(INFO) << "imgPath: " << imgPath;
    cv::Mat im0 = cv::imread(imgPath);

    // 应用letterbox
    cv::Mat processed = LetterBox(im0, cv::Size(imgSize[0], imgSize[1]), false);

    // BGR到RGB：反转通道顺序
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    cv::Mat chwImg = processed;  // dlite的格式是hwc
    if (isDpico) {
      chwImg = HwcToChw(processed);
    }
    // 假设processed是需要处理的cv::Mat
    if (!chwImg.isContinuous()) {
      chwImg = chwImg.clone();  // 克隆矩阵以获得连续内存
    }
    if (isDpico) {
      // 转换为浮点数并归一化，使用更精确的方式
      cv::Mat doubleTemp;
      chwImg.convertTo(doubleTemp, CV_64FC3);  // 先转为double
      float divide = 255.0f;
      doubleTemp /= divide;                    // double精度除法
      doubleTemp.convertTo(chwImg, CV_32FC3);  // 截断为float32
    }
    if (isDpico) {
      ReadImgToBufDpico(chwImg, tensorDescs[i], tensorBufs[i]);
    } else {
      ReadImgToBufDlite(chwImg, tensorDescs[i], tensorBufs[i]);
    }
    LOG(INFO) << "read end: " << imgPath;
  }

  LOG(INFO) << "PreProcessing completed successfully!";
  return true;
}
}  // namespace Infer