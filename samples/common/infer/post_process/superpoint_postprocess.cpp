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

#include "superpoint_postprocess.h"
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
#include "utils.h"
#include <unistd.h>
#include "log.h"
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <map>
#include "superpoint_utils.h"

using namespace std;
using json = nlohmann::json;

#define SIZE_OUTPUT0 (1 * 65 * 30 * 40)
#define SIZE_OUTPUT1 (1 * 256 * 30 * 40)

namespace Infer
{
    using namespace std;
    static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

    std::string GetCurrentWorkingDir()
    {
        char buffer[PATH_MAX];
        if (getcwd(buffer, sizeof(buffer)) != NULL) {
            return std::string(buffer);
        } else {
            // 失败时可以通过 perror 打印错误原因
            LOG(ERROR) << "获取当前目录失败";
            return "";
        }
    }

    int SaveOutBufs(std::vector<Infer::TensorBuf> &outBufs, const std::string &fileName)
    {
        std::string outputName = fileName;
        std::string resultPath = "../out/result";
        struct stat info;
        if (stat(resultPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(resultPath.c_str(), 0777);
            INFO_LOG("create file success");
        }
        std::string jpgPath = resultPath + "/jpg";
        if (stat(jpgPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(jpgPath.c_str(), 0777);
        }
        std::string binPath = resultPath + "/bin";
        if (stat(binPath.c_str(), &info) != 0) {
            // 文件夹不存在，尝试创建
            mkdir(binPath.c_str(), 0777);
        }
        std::string path = GetCurrentWorkingDir() + "/result/bin/";
        for (size_t i = 0; i < outBufs.size(); i++) {
            // 将拼接字符串,类似 path+ outputName + "_0.bin"
            std::string outputPath = path + outputName + "_" + std::to_string(i) + ".bin";
            std::ofstream file(outputPath, std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("无法打开文件进行写入");
            }
            float *data = static_cast<float *>(outBufs[i].GetRawPtr());
            // 直接写入原始数据（不进行格式转换）
            file.write(reinterpret_cast<const char *>(data), outBufs[i].size);
            file.close();
            LOG(INFO) << "save outBufs[" << i << "] success " << outputPath;
        }
        return SUCCESS;
    };

    bool SuperPointPostprocess(std::vector<std::string>& inputFileList, std::vector<TensorBuf>& outBufs, std::vector<TensorDesc>& outDescs)
    {
        std::vector<std::vector<cv::Point>> bbox;
        // 读取配置文件
        for (size_t i = 0; i < outBufs.size(); i++) {
            TensorDesc desc = outDescs[i];
            TensorBuf buf = outBufs[i];
            if (desc.defaultStride == 0) {
                desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
                buf.stride = desc.defaultStride;
            }
        }
        LOG(INFO) << "inputFileList jpg path ==========  " << inputFileList[0] << " success";
        /* post process, save and extract feature point */
        size_t sizeOut0 = SIZE_OUTPUT0 * sizeof(float);
        size_t sizeOut1 = SIZE_OUTPUT1 * sizeof(float);
        if (outBufs.size() != 2 || outBufs[0].size != sizeOut0 || outBufs[1].size != sizeOut1) {
            throw std::runtime_error("model output is invalid ");
        }

        float *data0 = static_cast<float *>(outBufs[0].GetRawPtr());
        std::vector<float> out0(data0, data0 + outBufs[0].size / sizeof(float));
        float *data1 = static_cast<float *>(outBufs[1].GetRawPtr());
        std::vector<float> out1(data1, data1 + outBufs[1].size / sizeof(float));
        std::string gtPath = inputFileList[0];
        size_t start = gtPath.find_last_of("/");
        
        size_t end = gtPath.find_last_of(".");
        std::string outputName = gtPath.substr(start,  end - start);
        std::string tmpName = gtPath.substr(0,  start);
        start = tmpName.find_last_of("/");
        std::string subName = tmpName.substr(start+1);
        outputName = subName + "_" + ExtractFilename(inputFileList[0]);
        LOG(INFO) << "execute outputName path ==========  " << outputName << " success";
        SaveOutBufs(outBufs, outputName);
        std::vector<KeyPoint> keypoints = ProcessKeypoints(out0, out1);

        // 6. 处理解析结果
        LOG(INFO) <<  "解析完成，共检测到 " << keypoints.size() << " 个关键点";

        // 示例：输出前5个关键点信息
        for (size_t i = 0; i < std::min(5ul, keypoints.size()); ++i)
        {
            const auto &kp = keypoints[i];
            LOG(INFO) <<  "关键点 " << i + 1 << ":";
            LOG(INFO) <<  "  位置: (" << kp.x << ", " << kp.y << ")";
            LOG(INFO) <<  "  置信度: " << kp.score;
            LOG(INFO) <<  "  特征向量前3个值: " << kp.descriptor[0] << "," << kp.descriptor[1] << "," << kp.descriptor[2] << ",特征向量长度：" << kp.descriptor.size();
        }

        std::string jpgPath = GetCurrentWorkingDir() + "/result/jpg/";
        LOG(INFO) << "execute jpg path ==========  " << jpgPath;
        std::string jpgOutputPath = jpgPath + outputName + "_draw_keypoints.jpg";
        VisualizeKeypoints(keypoints, inputFileList[0], jpgOutputPath);
        LOG(INFO) << "execute jpg path ==========  " << jpgOutputPath << " success";
        LOG(INFO) << "dump final data success " << jpgOutputPath[0];
        return true;
    }
}