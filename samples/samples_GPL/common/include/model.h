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
#pragma once

#include <vector>
#include <functional>
#include <utility>
#include <unordered_map>
#include "dev_interface_adapter.h"

namespace Infer {
struct Tensor {
    Tensor(): buf(), desc() {}
    Tensor(const TensorBuf& tensorBuf, const TensorDesc& tensorDesc)
        : desc(tensorDesc), buf(tensorBuf) {}
    struct TensorBuf buf;
    struct TensorDesc desc;
};

int32_t EnvInit(const std::string& configPath = "");
int32_t EnvDeinit();

enum FileType {
    SingelImageFile,
    JsonFile
};

enum ModelType {
    Unknown,
    Resnet50,
    VGG16,
    Yolov5,
    Yolov3,
    Unet,
    Yolov7,
    Yolov8sSeg,
    Yolov11sSeg,
    Yolov11sPose,
    Yolov6s,
    Yolov10s,
    Yolov8sObb,
    Yolov8s,
    Yolov9s,
    Yolov8sWorld
};

using ProcessFunc = std::function<bool(std::vector<std::string>&, std::vector<TensorBuf>&, std::vector<TensorDesc>&)>;

class Model {
public:

    virtual ~Model() = default;
    Model() : mdl_(MdlCreate()) {}
    int32_t Load(const std::string& modelPath, ModelType modelType = Unknown);
    int32_t Unload();

    std::vector<std::vector<Tensor>> Infer(const std::string& filePath, FileType fileType = JsonFile);
    std::vector<Tensor> Infer(std::vector<Tensor>& tensors);

    void SetPreProcessFunc(ProcessFunc func);
    void SetPostProcessFunc(ProcessFunc func);

private:
    std::vector<TensorDesc> mdlInputDescs_;
    std::vector<TensorDesc> mdlOutputDescs_;
    const std::shared_ptr<MdlBase> mdl_;
    ProcessFunc preprocessFunc_;
    ProcessFunc postprocessFunc_;
    static const std::unordered_map<ModelType, std::pair<ProcessFunc, ProcessFunc>> modelTypeToProcessMap_;
};
}