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

#include "unet_postprocess.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <climits>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include "log.h"
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <map>

using namespace std;

namespace Infer
{
    using namespace std;
    constexpr int BYTE_BIT_NUM = 8;
    constexpr double NORMALIZED = 255.0;

    static void SaveResultBin(Infer::TensorBuf &outBuf, const std::string &filePath)
    {
        // 获取保存文件路径和文件名
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");

        std::string fileName = filePath.substr(start, end - start);
        std::string resultPath = "../out/result";
        struct stat info;
        if (stat(resultPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(resultPath.c_str(), 0777);
            INFO_LOG("create file success");
        }
        std::string jpgPath = resultPath + "/jpg";
        std::string binPath = resultPath + "/bin";
        if (stat(jpgPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(jpgPath.c_str(), 0777);
            INFO_LOG("create file success");
        }
        if (stat(binPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(binPath.c_str(), 0777);
        }
        // 保存bin文件
        std::string binFile = binPath + fileName + "_0.bin";
        std::ofstream file(binFile, std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char *>(outBuf.GetRawPtr()), outBuf.size);
            file.close();
        } else {
            ERROR_LOG("open %s result bin failed\n", filePath.c_str());
        }
    }

    inline static float Sigmod(float a)
    {
        return 1.0f / (1.0f + exp(-a));
    }

    static bool Postprocess(Infer::TensorBuf &outBuf, Infer::TensorDesc &outDesc, const std::string &filePath)
    {
        float *data = static_cast<float *>(outBuf.GetRawPtr());

        // 创建OpenCV矩阵并应用sigmoid函数
        size_t size = outDesc.dims[outDesc.dimCount - 1];
        cv::Mat mask(size, size, CV_32FC1, data);
        cv::Mat maskSigmoid;

        // 创建一个变换矩阵（单位矩阵，不做实际变换）
        cv::Mat transformMat = cv::Mat::eye(1, 1, CV_32F);

        // 使用cv::transform进行矩阵变换
        cv::transform(mask, maskSigmoid, transformMat);

        // 然后手动应用sigmoid函数（仍然需要）
        maskSigmoid.forEach<float>([](float &pixel, const int *position) -> void
                                    { pixel = 1.0f / (1.0f + exp(-pixel)); });
        // 二值化处理
        cv::Mat maskBinary;
        cv::threshold(maskSigmoid, maskBinary, 0.5, 1.0, cv::THRESH_BINARY);
        maskBinary.convertTo(maskBinary, CV_8UC1, NORMALIZED); // 转换为8位无符号整数

        // 文件名类似：~/img/00001.bin
        // 获取保存文件路径和文件名
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");

        std::string binPath = "../out/result/jpg";
        std::string fileName = filePath.substr(start, end - start);

        // 保存jpg文件
        std::string imgName = binPath + fileName + ".jpg";
        LOG(INFO) << "imgName " << imgName;
        cv::imwrite(imgName, maskBinary);
        SaveResultBin(outBuf, filePath);
        return true;
    }

    bool UnetPostprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs)
    {
        if (tensorBufs.size() == 0 || fileList.size() == 0) {
            LOG(INFO) << " tensorBufs.size() == 0 || fileList.size() == 0  NO OUT";
            return false;
        }
        for (size_t i = 0; i < tensorBufs.size(); i++) {
            TensorDesc desc = tensorDescs[i];
            TensorBuf buf = tensorBufs[i];
            LOG(INFO) << "desc.defaultStride == 0 ";
            if (desc.defaultStride == 0) {

                desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
                buf.stride = desc.defaultStride;
                LOG(INFO) << "desc.defaultStride == 0 " << buf.stride;
            }
            Postprocess(buf, desc, fileList[i]);
        }
        LOG(INFO) << "dump final data success " << fileList[0];
        return true;
    }
}