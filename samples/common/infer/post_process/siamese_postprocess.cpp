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

#include "log.h"
#include "post_process.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <map>
#include <sys/stat.h>

namespace Infer {
using namespace std;
constexpr size_t BYTE_BIT_NUM = 8; // 1 byte = 8 bit
constexpr size_t TOP_NUM = 5;
constexpr float DISSIMILARITY_THRESHOLD = 0.9;

static Result GetAndSaveOutputWithBin(Infer::TensorBuf& outBuf, Infer::TensorDesc& outDesc,
    std::string outputBinFileName, std::vector<float>& noStrideBuf)
{
    size_t lastDim = outDesc.dims[outDesc.dimCount - 1];
    size_t dataSize = outDesc.typeSize / BYTE_BIT_NUM; // 一般为4
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
            for (size_t index = 0; index < lastDim; index++) {
                noStrideBuf.push_back(tempBuf[offset + index]);
            }
        }
    }
    if (noStrideBuf.empty()) {
        std::cout << "noStrideBuf malloc fail" << std::endl;
        return FAILED;
    }
    std::ofstream fout(outputBinFileName, std::ios::out | std::ios::binary);
    if (fout.good() == false) {
        std::cout << "create output file [%s] failed" << outputBinFileName.c_str() << std::endl;
        return FAILED;
    }
    fout.write((char*)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
    fout.close();
    chmod(outputBinFileName.c_str(), 0777);
    return SUCCESS;
}

static Result SaveResultWithTxt(const std::string& filePath, std::vector<float>& temp, string tempStr = "")
{
    std::vector<std::pair<unsigned int, float>> vec;
    std::string line;

    for (size_t i = 0; i < temp.size(); i++) {
        vec.push_back({ static_cast<unsigned int>(i), temp[i] });
    }

    std::sort(vec.begin(), vec.end(), [](const std::pair<unsigned int, float>& a,
    const std::pair<unsigned int, float>& b) {
        return a.second > b.second;
    });

    // 先获取原文件属性（可选）
    struct stat file_stat;
    stat(filePath.data(), &file_stat);

    // 打开文件并保留属性
    std::fstream file;
    file.open(filePath, std::ios::in | std::ios::out | std::ios::app);

    if (!file.is_open()) {
        // 如果文件不存在，创建它
        file.open(filePath, std::ios::out);
        if (file.is_open()) {
            file.close();
            // 确保文件已创建后再设置权限
            chmod(filePath.c_str(), 0777);
            // 重新以读写模式打开
            file.open(filePath, std::ios::in | std::ios::out | std::ios::app);
        }
    }
    string printContent;
    for (size_t i = 0; i < vec.size(); i++) {
        const auto& item = vec[i];
        unsigned int id = item.first;
        float value = item.second;
        printContent += to_string(id) + "," + to_string(value) + ";";
        file << id << "," << value << "\n";
    }
    if (temp.size() == 0 && tempStr.length() > 0) {
        file << tempStr << "\n";
    }
    file.close();
    if (temp.size() > 0) {
        LOG(INFO) << printContent << "result of " << filePath;
    }

    return SUCCESS;
}

template <typename T>
T CalculateEuclideanDistance(const std::vector<T>& vec1, const std::vector<T>& vec2)
{
    // 1. 检查向量维度是否一致
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("两个向量的维度必须相同才能计算欧氏距离。");
    }

    // 2. 计算差值的平方和
    T squaredDistanceSum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        T diff = vec1[i] - vec2[i];
        squaredDistanceSum += diff * diff;
    }

    // 3. 计算平方根并返回结果
    return std::sqrt(squaredDistanceSum);
}

std::vector<std::string> SplitString(const std::string& s, char delimiter)
{
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
    const std::vector<std::string>& fileList, int& perdictCorrectCount)
{
    auto filePath = fileList[0];
    auto filePath2 = fileList[1];
    auto class1 = SplitString(filePath, '/')[SplitString(filePath, '/').size() - 2];
    auto class2 = SplitString(filePath2, '/')[SplitString(filePath2, '/').size() - 2];
    int label = class1 == class2 ? 0 : 1;

    std::vector<float> res1;
    std::vector<float> res2;
    LOG(INFO) << filePath << " compared with " << filePath2;
    // 获取保存文件路径和文件名

    auto parts = SplitString(filePath, '/');
    std::string fileName1 = parts[parts.size() - 2] + "_" + parts.back().substr(0, parts.back().find_last_of("."));
    parts = SplitString(filePath2, '/');
    std::string fileName2 = parts[parts.size() - 2] + "_" + parts.back().substr(0, parts.back().find_last_of("."));
    std::string resultPath = "../out/result";
    struct stat info;
    mode_t oldUmask = umask(0);
    if (stat(resultPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(resultPath.c_str(), 0777);
        LOG(INFO) << "create file success";
    }
    std::string txtPath = resultPath + "/txt";
    if (stat(txtPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(txtPath.c_str(), 0777);
    }
    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), 0777);
    }
    // 保存bin文件
    std::string binFile1 = binPath + "/" + fileName1 + "_0.bin";
    std::string binFile2 = binPath + "/" + fileName2 + "_0.bin";
    GetAndSaveOutputWithBin(outBufs[0], outDescs[0], binFile1, res1);
    GetAndSaveOutputWithBin(outBufs[1], outDescs[1], binFile2, res2);
    std::string txtFile = txtPath + "/" + fileName1 + "_" + fileName2 + "_0.txt";
    std::vector<float> temp {};
    SaveResultWithTxt(txtFile, temp, "[Separate line]------------------" + filePath + "----------------------");
    SaveResultWithTxt(txtFile, res1, fileName1);
    SaveResultWithTxt(txtFile, temp, "[Separate line]------------------" + filePath2 + "----------------------");
    SaveResultWithTxt(txtFile, res2, fileName2);

    auto dis = CalculateEuclideanDistance(res1, res2);
    SaveResultWithTxt(txtFile, temp, "[Result line]------------------Dismillary: " + to_string(dis) + " , label: " +
        to_string(label) + "----------------------");
    LOG(INFO) << "dismillary: " << dis << " , label: " << label;
    perdictCorrectCount += ((label == 1 && dis >= DISSIMILARITY_THRESHOLD) ||
        (label == 0 && dis < DISSIMILARITY_THRESHOLD) ? 1 : 0);
    umask(oldUmask);
}

bool SiamesePostProcess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    int perdictCorrectCount;
    PostProcess(tensorBufs, tensorDescs, fileList, perdictCorrectCount);
    return true;
}
}