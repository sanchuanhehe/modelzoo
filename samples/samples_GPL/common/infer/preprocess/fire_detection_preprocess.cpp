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
 *
 *
 *
 */

#include "fire_detection_preprocess.h"

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
static constexpr int BYTE_BIT_NUM = 8;
static constexpr int YOLO_INPUT_SIZE = 640;
namespace Infer {

static Result ReadImgToBufDpico(const cv::Mat& mat, const TensorDesc& desc,
    TensorBuf& inBuf)
{
    size_t matTotalBytes = mat.total() * mat.elemSize();
    size_t bufTotalBytes = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
    for (size_t i = 0; i < desc.dimCount - 1; i++) {
        bufTotalBytes *= desc.dims[i];
    }

    char* bufPtr = static_cast<char*>(inBuf.GetRawPtr());
    memcpy(bufPtr, mat.data, matTotalBytes);
    return SUCCESS;
}

// Letterbox函数实现
static cv::Mat LetterBox(const cv::Mat& img, const cv::Size& targetSize,
    bool scaleup)
{
    cv::Size shape = img.size();

    // 计算缩放比例
    double r = std::min(static_cast<double>(targetSize.width) / shape.width,
        static_cast<double>(targetSize.height) / shape.height);
    r = std::min(r, 1.0);

    // 计算未填充的尺寸
    cv::Size unpad(static_cast<int>(std::round(shape.width * r)),
        static_cast<int>(std::round(shape.height * r)));

    // 计算填充量
    double w = (targetSize.width - unpad.width) / 2.0f;
    double h = (targetSize.height - unpad.height) / 2.0f;

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

static cv::Mat BGRToNV21(const cv::Mat& src)
{
    int w = src.cols;
    int h = src.rows;

    cv::Mat nv21(h + h / 2, w, CV_8UC1); /* 2: nv21格式yuv比例为 4:1:1，yuv数据长度为原图的1.5倍 */

    // 将 BGR 转换为 I420 (Planar) 格式
    // I420 布局: [Y (w*h)] [U (w*h/4)] [V (w*h/4)]
    cv::Mat yuvI420;
    cv::cvtColor(src, yuvI420, cv::COLOR_BGR2YUV_I420);

    // 拷贝 Y 分量 (直接拷贝前 h 行)
    // nv21(cv::Rect(0, 0, w, h)) 对应 Y 平面
    yuvI420.rowRange(0, h).copyTo(nv21.rowRange(0, h));

    // 交叉合并 U 和 V
    const uint8_t* uPlane = yuvI420.ptr<uint8_t>(h); // U 在 h 行开始
    const uint8_t* vPlane = yuvI420.ptr<uint8_t>(h + h / 4); /* 4: yuvI420，格式uv数据长度为原图的1/4 */

    // 指向 NV21 的 UV 交叉区起始位置
    uint8_t* uvData = nv21.ptr<uint8_t>(h);

    int uvPixelCount = (w * h) / 4; /* 4: yuvI420，格式uv数据长度为原图的1/4 */
    for (int i = 0; i < uvPixelCount; ++i) {
        // NV21 顺序是 V, U, V, U...
        *uvData++ = vPlane[i];
        *uvData++ = uPlane[i];
    }

    return nv21;
}

bool FireDetectionPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs,
    std::vector<TensorDesc>& tensorDescs)
{
    LOG(INFO) << "Processing file num: " << fileList.size();
    // 处理每个图像
    std::vector<int> imgSize = { YOLO_INPUT_SIZE, YOLO_INPUT_SIZE };
    for (size_t i = 0; i < fileList.size(); ++i) {
        std::string imgPath = fileList[i];
        LOG(INFO) << "imgPath: " << imgPath;
        cv::Mat im0 = cv::imread(imgPath);

        // 应用letterbox
        cv::Mat processed = LetterBox(im0, cv::Size(imgSize[0], imgSize[1]), false);

        // BGR到nv21：反转通道顺序
        cv::Mat yuvImg = BGRToNV21(processed);
        ReadImgToBufDpico(yuvImg, tensorDescs[i], tensorBufs[i]);

        // 保存为nv21
        std::string rawPath = "frame_" + std::to_string(i) + ".nv21";
        std::ofstream ofs(rawPath, std::ios::binary);
        if (ofs.is_open()) {
            // nv21 内存是连续的，直接写入所有数据
            ofs.write(reinterpret_cast<const char*>(yuvImg.data), yuvImg.total() * yuvImg.elemSize());
            ofs.close();
            LOG(INFO) << "Saved raw NV21 data to: " << rawPath;
        }
    }

    LOG(INFO) << "PreProcessing completed successfully!";
    return true;
}
} // namespace Infer
