/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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

#include "model.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "image_process.h"
#include "log.h"
#include "nlohmann/json.hpp"
#include "post_process.h"
#include "unet_postprocess.h"
#include "unet_preprocess.h"
#include "yolo10_postprocess.h"
#include "yolo11s_pose_postprocess.h"
#include "yolo11s_pose_preprocess.h"
#include "yolo11s_seg_postprocess.h"
#include "yolo11s_seg_preprocess.h"
#include "yolo3_postprocess.h"
#include "yolo3_preprocess.h"
#include "yolo6s_postprocess.h"
#include "yolo6s_preprocess.h"
#include "yolov7_postprocess.h"
#include "yolov7_preprocess.h"
#include "yolov8s_obb_postprocess.h"
#include "yolov8s_obb_preprocess.h"
#include "yolov8s_postprocess.h"
#include "yolov8s_preprocess.h"
#include "yolov8s_seg_postprocess.h"
#include "yolov8s_seg_preprocess.h"
#include "yolov9s_postprocess.h"
#include "yolov9s_preprocess.h"
#include "yolov8s_world_postprocess.h"
#include "yolov8s_world_preprocess.h"

namespace Infer {
struct ExecuteParam {
    size_t loop = 0;
    size_t processLoop = 0;
    std::vector<std::vector<std::string>> fileLists;
};
using json = nlohmann::json;
constexpr int US2S = 1000000;
constexpr int MS2S = 1000;
constexpr int S2MIN = 60;
constexpr float PERCENT100 = 100.00f;

#ifdef SVP_ACL_PLATFORM
const std::unordered_map<ModelType, std::pair<ProcessFunc, ProcessFunc>> Model::modelTypeToProcessMap_ = {
    { ModelType::Resnet50, { std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, ImageprocessOptions(256, 224, true)), PrintTop5 } },
    { ModelType::Yolov7, { std::bind(Yolov7Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), std::bind(GetYoloV7Box, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true) } },
    { ModelType::Yolov6s, { Yolo6s::Yolov6sPreprocess, Yolo6s::Yolov6sPostprocess } },
    { ModelType::Yolov8sSeg, { Yolov8sSegNS::Yolov8sSegPreprocess, Yolov8sSegNS::Yolov8sSegPostprocess } },
    { ModelType::Yolov11sSeg, { Yolo11sSegNS::Yolov11sSegPreprocess, Yolo11sSegNS::Yolov11sSegPostprocess } },
    { ModelType::Yolov8sObb, { Yolo8sObb::Yolov8sObbPreprocess, Yolo8sObb::Yolov8sObbPostprocess } },
    { ModelType::Yolov11sPose, { Yolo11sPoseNS::Yolov11sPosePreprocess, Yolo11sPoseNS::Yolov11sPosePostprocess } },
    { ModelType::Unet, { std::bind(UnetPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, ImageprocessOptions(572, -1, true)), UnetPostprocess } }, // unet模型需要中心剪裁572大小，不进行resize
    { ModelType::Yolov10s, { std::bind(Yolo3Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), GetYolo10Box } },
    { ModelType::Yolov3, { std::bind(Yolo3Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), GetYolo3Box } },
    { ModelType::Yolov8s, { std::bind(Yolov8sPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), GetYoloV8sBox } },
    { ModelType::Yolov5, { std::bind(Yolov7Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), std::bind(GetYoloV7Box, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true) } },
    { ModelType::Yolov9s, { std::bind(Yolov9sPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), GetYoloV9sBox } },
    { ModelType::Yolov8sWorld, { std::bind(Yolov8sWorldPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), GetYoloV8sWorldBox } }
};
#else
const std::unordered_map<ModelType, std::pair<ProcessFunc, ProcessFunc>> Model::modelTypeToProcessMap_ = {
    { ModelType::Resnet50, { std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, ImageprocessOptions(256, 256, false)), PrintTop5 } },
    { ModelType::VGG16, { std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, ImageprocessOptions(256, 224, false)), PrintTop5 } },
    { ModelType::Yolov7, { std::bind(Yolov7Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), std::bind(GetYoloV7Box, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false) } },
    { ModelType::Yolov6s, { Yolo6s::Yolov6sPreprocess, Yolo6s::Yolov6sPostprocess } },
    { ModelType::Yolov8sSeg, { Yolov8sSegNS::Yolov8sSegPreprocess, Yolov8sSegNS::Yolov8sSegPostprocess } },
    { ModelType::Yolov11sSeg, { Yolo11sSegNS::Yolov11sSegPreprocess, Yolo11sSegNS::Yolov11sSegPostprocess } },
    { ModelType::Yolov8sObb, { Yolo8sObb::Yolov8sObbPreprocess, Yolo8sObb::Yolov8sObbPostprocess } },
    { ModelType::Yolov11sPose, { Yolo11sPoseNS::Yolov11sPosePreprocess, Yolo11sPoseNS::Yolov11sPosePostprocess } },
    { ModelType::Unet, { std::bind(UnetPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, ImageprocessOptions(572, -1, true)), UnetPostprocess } },
    { ModelType::Yolov3, { std::bind(Yolo3Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), GetYolo3Box } },
    { ModelType::Yolov8s, { std::bind(Yolov8sPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), GetYoloV8sBox } },
    { ModelType::Yolov5, { std::bind(Yolov7Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), std::bind(GetYoloV7Box, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false) } },
    { ModelType::Yolov9s, { std::bind(Yolov9sPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), GetYoloV9sBox } },
    { ModelType::Yolov8sWorld, { std::bind(Yolov8sWorldPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), GetYoloV8sWorldBox } }
};
#endif

static bool ParseInputJsonFile(const std::string& filePath,
    ExecuteParam& param)
{
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }
    try {
        json config;
        file >> config;

        auto& fileList = config["fileList"];
        if (!fileList.is_array()) {
            throw std::runtime_error("'fileList' 必须是数组类型");
        }
        for (const auto& array : fileList) {
            if (!array.is_array() || array.empty()) {
                continue;
            }
            std::vector<std::string> row;
            for (const auto& item : array) {
                row.push_back(item.get<std::string>());
            }
            param.fileLists.emplace_back(row);
        }
        if (config.contains("loop")) {
            param.loop = config["loop"].get<size_t>();
        }
        if (config.contains("processLoop")) {
            param.processLoop = config["processLoop"].get<size_t>();
        }
    } catch (const json::parse_error& e) {
        LOG(ERROR) << "JSON 解析错误: " << e.what();
        return false;
    } catch (const json::type_error& e) {
        LOG(ERROR) << "数据类型错误: " << e.what();
        return false;
    } catch (const std::exception& e) {
        LOG(ERROR) << "运行时错误: " << e.what();
        return false;
    }
    return true;
}

int32_t EnvInit(const std::string& configPath) { return DevInit(configPath); }
int32_t EnvDeinit() { return DevDeInit(); }

int32_t Model::Load(const std::string& modelPath, ModelType modelType)
{
    if (modelTypeToProcessMap_.find(modelType) == modelTypeToProcessMap_.end()) {
        LOG(ERROR) << "unsupported model type";
        return -1;
    }
    preprocessFunc_ = modelTypeToProcessMap_.at(modelType).first;
    postprocessFunc_ = modelTypeToProcessMap_.at(modelType).second;
    if (mdl_->LoadModel(modelPath) != 0) {
        LOG(ERROR) << "failed to load model";
        return -1;
    }
    mdlInputDescs_.resize(mdl_->GetInTensorNum());
    mdlOutputDescs_.resize(mdl_->GetOutTensorNum());
    for (size_t i = 0; i < mdlInputDescs_.size(); ++i) {
        if (mdl_->GetInTensorDescByIdx(i, mdlInputDescs_[i]) != 0) {
            LOG(ERROR) << "failed to get input tensor desc";
            return -1;
        }
    }
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        if (mdl_->GetOutTensorDescByIdx(i, mdlOutputDescs_[i]) != 0) {
            LOG(ERROR) << "failed to get output tensor desc";
            return -1;
        }
    }
    return 0;
}

int32_t Model::Unload() { return mdl_->UnLoadModel(); }

void Model::SetPreProcessFunc(ProcessFunc func) { preprocessFunc_ = func; }

void Model::SetPostProcessFunc(ProcessFunc func) { postprocessFunc_ = func; }

std::vector<std::vector<Tensor>> Model::Infer(const std::string& filePath,
    FileType fileType)
{
    std::vector<std::vector<Tensor>> outputs;
    std::vector<TensorBuf> inBuf;
    for (size_t i = 0; i < mdlInputDescs_.size(); ++i) {
        inBuf.emplace_back(mdlInputDescs_[i].defaultSize,
            mdlInputDescs_[i].defaultStride);
    }
    std::vector<TensorBuf> outBuf;
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize,
            mdlOutputDescs_[i].defaultStride);
    }

    ExecuteParam param;
    if (fileType == FileType::JsonFile) {
        if (!ParseInputJsonFile(filePath, param)) {
            LOG(ERROR) << "failed to parse input json file";
            return {};
        }
    } else {
        // 输入为单个文件
        if (inBuf.size() != 1) {
            LOG(ERROR) << "model input num should be 1";
            return {};
        }
        param.fileLists = { { filePath } };
    }
    if (param.loop > 0) {
        inBuf.emplace_back(4, 0);
    }
    size_t loop = param.loop > 0 ? param.loop : 1;
    int processLoop = param.processLoop > 0 ? param.processLoop : 1;;
    std::chrono::microseconds dur(0);
    auto start = std::chrono::high_resolution_clock::now();
    auto preprocessStart = std::chrono::high_resolution_clock::now();
    auto postprocessStart = std::chrono::high_resolution_clock::now();
    float preprocessCostTime = 0;
    float postprocessCostTime = 0;
    for (size_t i = 0; i < param.fileLists.size(); ++i) {
        preprocessStart = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < processLoop; j++) {
            if (preprocessFunc_ == nullptr || !preprocessFunc_(param.fileLists[i], inBuf, mdlInputDescs_)) {
                LOG(ERROR) << "failed to preprocess model input";
                return {};
            }
        }
        preprocessCostTime += static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - preprocessStart)
                .count());
        for (size_t j = 0; j < loop; j++) {
            if (mdl_->Execute(inBuf, outBuf) != 0) {
                LOG(ERROR) << "failed to execute model";
                return {};
            }
        }
        postprocessStart = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < processLoop; j++) {
            if (postprocessFunc_ == nullptr || !postprocessFunc_(param.fileLists[i], outBuf, mdlOutputDescs_)) {
                LOG(ERROR) << "failed to postprocess model output";
                return {};
            }
        }
        outputs.push_back({});
        postprocessCostTime += static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - postprocessStart)
                .count());
        auto costTime = static_cast<float>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start)
                .count());
        costTime = costTime / US2S;
        auto finishedPercent = (i + 1) * PERCENT100 / param.fileLists.size();
        LOG(INFO) << std::fixed << std::setprecision(2) << "already infer " << i + 1
                  << " image, cost " << costTime << " s, " << PERCENT100 - finishedPercent
                  << "% remains, "
                  << costTime * (param.fileLists.size() - i - 1) * 1.00f / (i * S2MIN + S2MIN)
                  << "min remains.";
    }
    if (param.loop > 0 && param.fileLists.size() > 0) {
        float msDur = *(static_cast<float*>(inBuf[inBuf.size() - 1].GetRawPtr())) / (param.loop * param.fileLists.size() * MS2S);
        LOG(INFO) << std::fixed << std::setprecision(2) << "execution time: " << msDur << "ms, frame rate: " << (MS2S / msDur) << "fps";
    }
    if (param.processLoop > 0 && param.fileLists.size() > 0) {
        float preprocessMsDur = preprocessCostTime / MS2S;
        LOG(INFO) << std::fixed << std::setprecision(2) << "preprocess time: " << preprocessMsDur /  (processLoop * param.fileLists.size()) << " ms";
        float postprocessMsDur = postprocessCostTime / MS2S;
        LOG(INFO) << std::fixed << std::setprecision(2) << "postprocess time: " << postprocessMsDur /  (processLoop * param.fileLists.size()) << " ms";
    }
    return outputs;
}

std::vector<Tensor> Model::Infer(std::vector<Tensor>& tensors)
{
    std::vector<TensorBuf> inBuf, outBuf;
    std::vector<Tensor> outputs;
    if (tensors.size() != mdlInputDescs_.size()) {
        LOG(ERROR) << "invalid input tensor num";
        return {};
    }
    for (size_t i = 0; i < mdlInputDescs_.size(); ++i) {
        inBuf.emplace_back(tensors[i].buf.GetRawPtr(), tensors[i].buf.size,
            tensors[i].buf.stride);
    }
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize,
            mdlOutputDescs_[i].defaultStride);
    }
    if (mdl_->Execute(inBuf, outBuf) != 0) {
        LOG(ERROR) << "failed to execute model";
        return {};
    }
    for (size_t k = 0; k < mdlOutputDescs_.size(); ++k) {
        outputs.push_back(Tensor(outBuf[k].DeepCopy(), mdlOutputDescs_[k]));
    }
    return outputs;
}
} // namespace Infer