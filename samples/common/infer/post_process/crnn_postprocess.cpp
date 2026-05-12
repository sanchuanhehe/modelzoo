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

#include "crnn_postprocess.h"
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
constexpr int BYTE_BIT_NUM = 8;

static Result GetAndSaveOutputWithBin(Infer::TensorBuf& outBuf, Infer::TensorDesc& outDesc, std::string outputBinFileName, std::vector<float>& noStrideBuf)
{
    int64_t lastDim = outDesc.dims[outDesc.dimCount - 1];
    size_t dataSize = outDesc.typeSize / BYTE_BIT_NUM;
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = outBuf.size / outBuf.stride;
    size_t strideElemNum = outBuf.stride / dataSize;
    float* outData = static_cast<float*>(outBuf.GetRawPtr());
    size_t outSize = outBuf.size / dataSize;
    if (outBuf.stride == lastDimSize || outBuf.stride == 0) {
        std::vector<float> tempBuf(outData, outData + outSize);
        noStrideBuf.assign(tempBuf.begin(), tempBuf.end());
    } else {
        std::vector<float> tempBuf(outData, outData + outSize);
        for (size_t i = 0; i < loopNum; ++i) {
            size_t offset = i * strideElemNum;
            for (size_t index = 0; index < static_cast<size_t>(lastDim); index++) {
                noStrideBuf.push_back(tempBuf[offset + index]);
            }
        }
    }
    if (noStrideBuf.empty()) {
        LOG(ERROR) << "noStrideBuf malloc fail";
        return FAILED;
    }
    std::ofstream fout(outputBinFileName, std::ios::out | std::ios::binary);
    if (fout.good() == false) {
        LOG(ERROR) << "create output file " << outputBinFileName.c_str() << " failed";
        return FAILED;
    }
    fout.write((char*)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
    fout.close();
    chmod(outputBinFileName.c_str(), DIR_PERMISSIONS);
    return SUCCESS;
}

static Result SaveResultWithTxt(const std::string& filePath, string cxt)
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

    file << cxt;
    file.close();
    return SUCCESS;
}

void CalResult(const std::vector<float>& output, const std::string& fileName)
{
    int timeStep = 41;
    int alphabetsLength = output.size() / timeStep;
    std::vector<int> charIndeces;
    for (int i = 0; i < timeStep; ++i) {
        std::vector<float> temp(output.begin() + i * alphabetsLength, output.begin() + (i + 1) * alphabetsLength);
        // 找到最大值的迭代器
        auto maxIter = std::max_element(temp.begin(), temp.end());
        // 迭代器差值 = 索引（size_t 是无符号整数，适配索引类型）
        int charIndex = std::distance(temp.begin(), maxIter); // 等价于 maxIter - vec.begin()
        charIndeces.emplace_back(charIndex);
    }
    std::vector<int> charList;
    string cxt;
    for (size_t i = 0; i < charIndeces.size(); ++i) {
        if (charIndeces[i] != 0 && !(i > 0 && charIndeces[i - 1] == charIndeces[i])) {
            cxt += to_string(charIndeces[i]) + " ";
            charList.emplace_back(charIndeces[i]);
        }
    }
    SaveResultWithTxt("../out/result/txt/" + fileName + "_result.txt", cxt);
}

static void ProcessAndSaveOutput(Infer::TensorBuf& outBuf, Infer::TensorDesc& outDesc,
    const std::string& resultPath, const std::string& fileName)
{
    auto txtPath = resultPath + "/txt";
    auto binPath = resultPath + "/bin";
    struct stat info;

    if (stat(txtPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(txtPath.c_str(), DIR_PERMISSIONS);
    }
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), DIR_PERMISSIONS);
    }
    std::vector<float> res;
    // 解析为OpenCV Mat
    GetAndSaveOutputWithBin(outBuf, outDesc, "../out/result/bin/" + fileName + "_output.bin", res);
    if (res.empty()) {
        LOG(ERROR) << "res is empty!";
        return;
    }
    CalResult(res, fileName);
}

std::vector<std::string> SplitStr(const std::string& s, char delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = s.find(delimiter);

    while (end != std::string::npos) {
        result.push_back(s.substr(start, end - start));
        start = end + 1;
        end = s.find(delimiter, start);
    }
    result.push_back(s.substr(start));
    return result;
}

static void PostProcess(std::vector<Infer::TensorBuf>& outBufs, std::vector<Infer::TensorDesc>& outDescs,
    const std::vector<std::string>& fileList)
{
    auto filePath = fileList[0];

    // 获取保存文件路径和文件名
    auto parts = SplitStr(filePath, '/');
    std::string fileName = parts.back().substr(0, parts.back().find_last_of("."));
    std::string resultPath = "../out/result";
    struct stat info;

    // 可以使得权限设置生效
    mode_t oldUmask = umask(0);

    if (stat(resultPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(resultPath.c_str(), DIR_PERMISSIONS);
        LOG(INFO) << "Create result directory successfully: " << resultPath;
    }

    std::string txtPath = resultPath + "/txt";
    if (stat(txtPath.c_str(), &info) != 0) {
        mkdir(txtPath.c_str(), DIR_PERMISSIONS);
    }

    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0) {
        mkdir(binPath.c_str(), DIR_PERMISSIONS);
    }
    if (outBufs.size() > 1) {
        ProcessAndSaveOutput(outBufs[4], outDescs[4], resultPath, fileName);
    } else {
        ProcessAndSaveOutput(outBufs[0], outDescs[0], resultPath, fileName);
    }

    umask(oldUmask);

    LOG(INFO) << filePath << " already finished!";
}

bool CRNNPostProcess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    PostProcess(tensorBufs, tensorDescs, fileList);
    return true;
}
}