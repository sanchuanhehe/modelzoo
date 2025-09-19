#include "model.h"
#include "log.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "image_process.h"
#include "post_process.h"
#include "nlohmann/json.hpp"

namespace Infer {

using json = nlohmann::json;

const std::unordered_map<ModelType, std::pair<PreProcess, PostProcess>> Model::modelTypeToProcessMap_ = {
    { ModelType::Resnet50, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2,
                            ImageprocessOptions(256, 224)), PrintTop5} }
};

static std::vector<std::vector<std::string>> ParseFileList(const std::string& fileListPaths)
{
    std::vector<std::vector<std::string>> result;
    std::ifstream file(fileListPaths);
    if (!file.is_open()) {
        return result;
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
            result.emplace_back(row);
        }
    } catch (const json::parse_error& e) {
        LOG(ERROR) << "JSON 解析错误: " << e.what();
        return {};
    } catch (const json::type_error& e) {
        LOG(ERROR) << "数据类型错误: " << e.what();
        return {};
    } catch (const std::exception& e) {
        LOG(ERROR) << "运行时错误: " << e.what();
        return {};
    }
    return result;
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
    preprocess_ = modelTypeToProcessMap_.at(modelType).first;
    postprocess_ = modelTypeToProcessMap_.at(modelType).second;
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

std::vector<std::vector<Tensor>> Model::Infer(const std::string& fileListPath, FileType fileType)
{
    std::vector<std::vector<Tensor>> outputs;
    std::vector<TensorBuf> inBuf(mdlInputDescs_.size()); 
    std::vector<TensorBuf> outBuf;
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize, mdlOutputDescs_[i].defaultStride);
    }
    std::vector<std::vector<std::string>> fileLists;
    if (fileType == FileType::JsonFile) {
        fileLists = ParseFileList(fileListPath);
        if (fileLists.size() == 0) {
            LOG(ERROR) << "failed to parse file list";
            return {};
        }
    } else {
        // 输入为单个文件
        if (inBuf.size() != 1) {
            LOG(ERROR) << "model input num should be 1";
            return {};
        }
        fileLists = {{fileListPath}};
    }

    for (size_t i = 0; i < fileLists.size(); ++i) {
        if (preprocess_ == nullptr || !preprocess_(fileLists[i], inBuf)) {
            LOG(ERROR) << "failed to preprocess model input";
            return {};
        }
        if (mdl_->Execute(inBuf, outBuf) != 0) {
            LOG(ERROR) << "failed to execute model";
            return {};
        }
        std::vector<Tensor> output;
        for (size_t k = 0; k < mdlOutputDescs_.size(); ++k) {
            output.push_back(Tensor(outBuf[k].DeepCopy(), mdlOutputDescs_[k]));
        }
        if (postprocess_ == nullptr || !postprocess_(output, fileLists[i])) {
            LOG(ERROR) << "failed to postprocess model output";
            return {};
        }
        outputs.push_back(output);
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