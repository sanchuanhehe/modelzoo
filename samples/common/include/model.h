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
        : buf(tensorBuf), desc(tensorDesc) {}
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
    Custom,
    Resnet50,
    VGG16,
    Yolov5,
    PFLD,
    CrowdCount,
    FaceNet,
    FastSpeech2,
	EfficientNet,
    InceptionV3,
    SEResnet50,
    SwinT,
    VitB16,
    CodeFormer,
    Yolov4,
    SiameseNetwork,
    VDSR,
    SuperPoint,
    LRStereo,
    LRStereoDis,
    PaddleOCR_Det,
    PaddleOCR_Rec,
    CRNN,
    HRNet,
    TinySam,
    GraspNet,
    MiniCPM,
    ClipImg,
    ClipTxt,
    Pi0,
    SqueezeNet1_1,
    DenseNet121,
    ShuffleNetV2,
    MobileNetV2,
    ResNet18,
    ResNet101,
    DepthAnythingV2
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
    std::vector<Tensor> Infer(std::vector<Tensor>& tensors, std::string filePath);

    std::vector<TensorBuf> Infer(std::vector<TensorBuf>& tensorBufs);
    int Infer(std::vector<TensorBuf>& inBufs, std::vector<TensorBuf>& outBufs);
    std::pair<std::vector<TensorDesc>, std::vector<TensorDesc>> GetModelInfo();

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