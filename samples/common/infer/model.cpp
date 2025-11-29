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

#include "model.h"
#include "log.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "image_process.h"
#include "post_process.h"
#include "yolo4_preprocess.h"
#include "yolo4_postprocess.h"
#include "codeformer_preprocess.h"
#include "codeformer_postprocess.h"
#include "nlohmann/json.hpp"
#include "PillowResize/PillowResize.hpp"
#include "efficient_preprocess.h"

namespace Infer {
using json = nlohmann::json;
struct ExecuteParam
{
    size_t loop = 1;
    std::vector<std::vector<std::string>> fileLists;
};

#ifdef SVP_ACL_PLATFORM
const std::unordered_map<ModelType, std::pair<ProcessFunc, ProcessFunc>> Model::modelTypeToProcessMap_ = {
    { ModelType::Resnet50, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::InceptionV3, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(342, 299, true)), PrintTop5AndDumpResult} },
    { ModelType::SEResnet50, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::SwinT, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(256, 224, true, PillowResize::INTERPOLATION_BICUBIC)), PrintTop5AndDumpResult} },
    { ModelType::VitB16, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(248, 224, true, PillowResize::INTERPOLATION_BICUBIC)), PrintTop5AndDumpResult} },
    { ModelType::VGG16, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::Yolov4, {Yolo4::Yolov4Preprocess, Yolo4::Yolov4Postprocess} },
    { ModelType::CodeFormer, {CodeFormerNS::CodeFormerPreprocess, CodeFormerNS::CodeFormerPostprocess} },
    { ModelType::EfficientNet, {EfficientNetPreprocess, PrintTop5AndDumpResult} }
};
#else
const std::unordered_map<ModelType, std::pair<ProcessFunc, ProcessFunc>> Model::modelTypeToProcessMap_ = {
    { ModelType::Resnet50, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(256, 256, false)), PrintTop5AndDumpResult} },
    { ModelType::InceptionV3, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(342, 304, false)), PrintTop5AndDumpResult} },
    { ModelType::SEResnet50, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::SwinT, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(256, 224, false, PillowResize::INTERPOLATION_BICUBIC)), PrintTop5AndDumpResult} },
    { ModelType::VitB16, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            ImageprocessOptions(248, 224, false, PillowResize::INTERPOLATION_BICUBIC)), PrintTop5AndDumpResult} },
    { ModelType::VGG16, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::Yolov4, {Yolo4::Yolov4Preprocess, Yolo4::Yolov4Postprocess} },
    { ModelType::CodeFormer, {CodeFormerNS::CodeFormerPreprocess, CodeFormerNS::CodeFormerPostprocess} }
};
#endif

static bool ParseInputJsonFile(const std::string& filePath, ExecuteParam& param)
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

int32_t EnvInit(const std::string& configPath)
{
    return DevInit(configPath);
}
int32_t EnvDeinit()
{
    return DevDeInit();
}

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

int32_t Model::Unload()
{
    return mdl_->UnLoadModel();
}

void Model::SetPreProcessFunc(ProcessFunc func)
{
    preprocessFunc_ = func;
}

void Model::SetPostProcessFunc(ProcessFunc func)
{
    postprocessFunc_ = func;
}

std::vector<std::vector<Tensor>> Model::Infer(const std::string& filePath, FileType fileType)
{
    std::vector<std::vector<Tensor>> outputs;
    std::vector<TensorBuf> inBuf;
    for (size_t i = 0; i < mdlInputDescs_.size(); ++i) {
        inBuf.emplace_back(mdlInputDescs_[i].defaultSize, mdlInputDescs_[i].defaultStride);
    }
    std::vector<TensorBuf> outBuf;
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize, mdlOutputDescs_[i].defaultStride);
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
        param.fileLists = {{filePath}};
    }

    std::chrono::microseconds dur(0);
    for (size_t i = 0; i < param.fileLists.size(); ++i) {
        if (preprocessFunc_ == nullptr || !preprocessFunc_(param.fileLists[i], inBuf, mdlInputDescs_)) {
            LOG(ERROR) << "failed to preprocess model input";
            return {};
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < param.loop; j++) {
            if (mdl_->Execute(inBuf, outBuf) != 0) {
                LOG(ERROR) << "failed to execute model";
                return {};
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (postprocessFunc_ == nullptr || !postprocessFunc_(param.fileLists[i], outBuf, mdlOutputDescs_)) {
            LOG(ERROR) << "failed to postprocess model output";
            return {};
        }
        std::vector<Tensor> output;
        for (size_t k = 0; k < mdlOutputDescs_.size(); ++k) {
            output.push_back(Tensor(outBuf[k].DeepCopy(), mdlOutputDescs_[k]));
        }
        outputs.push_back(output);
    }
    float msDur = static_cast<float>(dur.count()) / (param.loop * param.fileLists.size() * 1000.0); // 	1 millisecond = 1000 microseconds
    LOG(INFO) << std::fixed << std::setprecision(2) << "execution time: " << msDur << "ms, frame rate: " << (1000.0 / msDur) << "fps";
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
        inBuf.emplace_back(tensors[i].buf.GetRawPtr(), tensors[i].buf.size, tensors[i].buf.stride);
    }
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize, mdlOutputDescs_[i].defaultStride);
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
}