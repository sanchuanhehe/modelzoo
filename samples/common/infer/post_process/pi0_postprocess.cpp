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

#include "pi0_postprocess.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <array>
#include <functional>
#include "nlohmann/json.hpp"

namespace Infer {
static constexpr int MIN_FILELIST_INPUT_NUM = 3;
static constexpr int MAX_FILELIST_INPUT_NUM = 4;
static constexpr int BYTE_BIT_NUM = 8;
static constexpr int FP16_BIT_NUM = 16;
static constexpr int FP32_BIT_NUM = 32;
static constexpr int STATE_FILELIST_IDX = 1;

using json = nlohmann::json;
struct NormalizeParam
{
    std::vector<float> kMean;
    std::vector<float> kStd;
};

static std::vector<std::vector<float>> GetModelOutput(TensorBuf &outBufs, TensorDesc &outDescs, int actionLen)
{
    LOG(INFO) << "LoadModelOutput: outDescs.dimCount " << outDescs.dimCount;
    std::vector<std::vector<float>> data;
    void *outData = nullptr;
    const size_t dataSize = outDescs.typeSize / BYTE_BIT_NUM;
    int64_t width = 1;
    int64_t loopTimes = 1;
    if (outDescs.dimCount != 0) {
        width = outDescs.dims[outDescs.dimCount - 1]; // 最后一维是width
        loopTimes = outDescs.defaultSize / dataSize / width;
    }
    LOG(INFO) << "LoadModelOutput: outDescs.defaultSize " << outDescs.defaultSize;

    if (outDescs.typeSize == FP16_BIT_NUM) {
        outData = new uint16_t[outDescs.defaultSize];
        DevMemcpy(outData, outDescs.defaultSize, outBufs.GetRawPtr(), outDescs.defaultSize);
    } else if (outDescs.typeSize == FP32_BIT_NUM) {
        outData = new float[outDescs.defaultSize];
        DevMemcpy(outData, outDescs.defaultSize, outBufs.GetRawPtr(), outDescs.defaultSize);
    } else {
        LOG(ERROR) << "Unsupported outDescs.typeSize: " << outDescs.typeSize;
        return data;
    }

    actionLen = actionLen <= width ? actionLen : width;
    for (int64_t loop = 0; loop < loopTimes; loop++) {
        std::vector<float> row;
        row.reserve(actionLen);
        
        for (int64_t w = 0; w < actionLen; w++) {
            if (outDescs.typeSize == FP16_BIT_NUM) {
                row.push_back(HalfToFloat(static_cast<uint16_t*>(outData)[loop * width + w]));
            } else if (outDescs.typeSize == FP32_BIT_NUM) {
                row.push_back(static_cast<float*>(outData)[loop * width + w]);
            }
        }
        data.push_back(row);
    }
    if (outData != nullptr) {
        if (outDescs.typeSize == FP16_BIT_NUM) {
            delete[] static_cast<uint16_t*>(outData);
        } else if (outDescs.typeSize == FP32_BIT_NUM) {
            delete[] static_cast<float*>(outData);
        }
        outData = nullptr;
    }

    return data;
}

static bool ParseActionStatsJsonFile(const std::string& filePath, NormalizeParam& param)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        LOG(ERROR) << "Open action stats json file fail, path: " << filePath;
        return false;
    }
    try {
        json config;
        file >> config;
        auto& kMean = config["mean"];
        auto& kStd = config["std"];
        if (!kMean.is_array()) {
            file.close();
            throw std::runtime_error("'mean' must be array");

        }
        for (const auto& val : kMean) {
            param.kMean.emplace_back(float(val));
        }
        if (!kStd.is_array()) {
            file.close();
            throw std::runtime_error("'std' must be array");
        }
        for (const auto& val : kStd) {
            param.kStd.emplace_back(float(val));
        }
    } catch (const json::parse_error& e) {
        LOG(ERROR) << "JSON parse error: " << e.what();
        file.close();
        return false;
    } catch (const json::type_error& e) {
        LOG(ERROR) << "Data type error: " << e.what();
        file.close();
        return false;
    } catch (const std::exception& e) {
        LOG(ERROR) << "Running error: " << e.what();
        file.close();
        return false;
    }
    file.close();
    return true;
}

static Result Unnormalize(const std::string& actionStatsPath, std::vector<std::vector<float>> &modelOut, int actionLen)
{
    NormalizeParam param;
    ParseActionStatsJsonFile(actionStatsPath, param);
    if (param.kMean.size() != actionLen || param.kStd.size() != actionLen) {
        LOG(ERROR) << "The number of kMean: "<< param.kMean.size() << " or kStd: "<< param.kStd.size() <<
        " is wrong, please check";
        return FAILED;
    }

    for (int i = 0; i < modelOut.size() ; ++i) {
        for (int j = 0; j < modelOut[i].size() ; ++j) {
            modelOut[i][j] = modelOut[i][j] * param.kStd[j] + param.kMean[j];
        }
    }
    return SUCCESS;
}

static Result SaveAction(std::vector<std::string>& fileList, const std::string& saveTxtDir,
    const std::vector<std::vector<float>> &modelOut, int actionLen)
{
    if (fileList.size() < MIN_FILELIST_INPUT_NUM || fileList.size() > MAX_FILELIST_INPUT_NUM) {
        LOG(ERROR) << "The number of input is wrong, please check fileList_1.json " << fileList.size();
        return FAILED;
    }
    std::string statePath = fileList[STATE_FILELIST_IDX];
    size_t start = statePath.find_last_of("_");
    size_t end = statePath.find_last_of(".");
    std::string outputIdx = statePath.substr(start, end - start);
    std::string outputPath = saveTxtDir + "action" + outputIdx + ".txt";
    LOG(INFO) << "outputPath" << outputPath;
    std::ofstream file(outputPath);
    if (!file.is_open()) {
        LOG(ERROR) << "Cannot open file: " << outputPath;
        return FAILED;;
    }

    for (int i = 0; i < modelOut.size(); ++i) {
        for (int j = 0; j < actionLen; ++j) {
            file << modelOut[i][j];
            if (j < actionLen - 1) {
                file << " ";
            }
        }
        file << "\n";
    }
    file.close();
    return SUCCESS;
}

bool Pi0Postprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& outBufs,
    std::vector<TensorDesc>& outDescs)
{
    auto cfg = ReadCfgFile("../data/cfg.txt");
    int actionLen = std::stoi(cfg["action_len"]);
    std::string saveTxtDir = cfg["save_out_txt"];
    // 反归一化参数文件路径，用于模型推理结果反归一化处理，包含模型action输出结果的均值和方差
    // 该参数可从模型训练数据集中的meta/stats.json文件中获取
    // 本样例中使用的是aloha_sim_transfer_cube_scripted数据集中的统计参数
    std::string actionStatsPath = cfg["action_stats_path"];

    if (actionLen < 0) {
        LOG(ERROR) << "Invalid action len: " << actionLen;
        return false;
    }

    // 创建保存目录
    if (!saveTxtDir.empty() && !CreateDirectoryRecursive(saveTxtDir)) {
        LOG(ERROR) << "Create preprocess txt dir failed";
        return false;
    }

    std::vector<std::vector<float>> modelOut;
    modelOut = GetModelOutput(outBufs[0], outDescs[0], actionLen);
    // 反归一化
    if (Unnormalize(actionStatsPath, modelOut, actionLen) != SUCCESS) {
        LOG(ERROR) << "Failed to excute unnormalize";
        return false;
    } 

    // 保存action结果为txt文件
    if (SaveAction(fileList, saveTxtDir, modelOut, actionLen) != SUCCESS) {
        LOG(ERROR) << "Failed to save action result";
        return false;
    }
    return true;
}

}