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
#include "vit_preprocess.h"
#include "post_process.h"
#include "yolo4_preprocess.h"
#include "yolo4_postprocess.h"
#include "facenet_preprocess.h"
#include "facenet_postprocess.h"
#include "codeformer_preprocess.h"
#include "codeformer_postprocess.h"
#include "siamese_preprocess.h"
#include "siamese_postprocess.h"
#include "xstereo_preprocess.h"
#include "xstereo_dis_preprocess.h"
#include "xstereo_postprocess.h"
#include "nlohmann/json.hpp"
#include "PillowResize/PillowResize.hpp"
#include "efficient_preprocess.h"
#include "superpoint_preprocess.h"
#include "superpoint_postprocess.h"
#include "ocr_det_preprocess.h"
#include "ocr_det_postprocess.h"
#include "ocr_rec_preprocess.h"
#include "ocr_rec_postprocess.h"
#include "pfld_preprocess.h"
#include "pfld_postprocess.h"
#include "crowdcount_preprocess.h"
#include "crowdcount_postprocess.h"
#include "crnn_preprocess.h"
#include "crnn_postprocess.h"
#include "vdsr_preprocess.h"
#include "vdsr_postprocess.h"
#include "graspnet_preprocess.h"
#include "graspnet_postprocess.h"
#include "hrnet_preprocess.h"
#include "hrnet_postprocess.h"
#include "fastspeech2_preprocess.h"
#include "fastspeech2_postprocess.h"
#include "pi0_preprocess.h"
#include "pi0_postprocess.h"
#include "clip_txt_preprocess.h"
#include "clip_preprocess.h"
#include "clip_postprocess.h"
#include "vit_preprocess.h"
#include "depthanythingv2_preprocess.h"
#include "depthanythingv2_postprocess.h"

namespace Infer {
constexpr float MS2S = 1000.0f;
constexpr int US2S = 1000000;
constexpr int S2MIN = 60;
constexpr float PERCENT100 = 100.00f;

using json = nlohmann::json;
struct ExecuteParam
{
    size_t loop = 0;
    size_t processLoop = 0;
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
    { ModelType::VitB16, {std::bind(VitPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            VitPrerocessOptions(248, 224, true, PillowResize::INTERPOLATION_BICUBIC)), PrintTop5AndDumpResult} },
    { ModelType::VGG16, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::FaceNet, {FaceNetNS::FaceNetPreprocess, FaceNetNS::FaceNetPostprocess} },
    { ModelType::Yolov4, {Yolo4::Yolov4Preprocess, Yolo4::Yolov4Postprocess} },
    { ModelType::FastSpeech2, {FastSpeech2NS::FastSpeech2Preprocess, FastSpeech2NS::FastSpeech2Postprocess} },
    { ModelType::CodeFormer, {CodeFormerNS::CodeFormerPreprocess, CodeFormerNS::CodeFormerPostprocess} },
    { ModelType::EfficientNet, {EfficientNetPreprocess, PrintTop5AndDumpResult} },
    { ModelType::SiameseNetwork, {SiameseNetworkPreprocess, SiamesePostProcess} },
    { ModelType::LRStereo, {XStereoPreprocess, XStereoPostprocess}},
    { ModelType::LRStereoDis, {XStereoDisPreprocess, XStereoPostprocess}},
    { ModelType::PaddleOCR_Det, {OcrDetPreprocess, OcrDetPostprocess} },
    { ModelType::PaddleOCR_Rec, { std::bind(OcrRecPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), 
                    std::bind(OcrRecPostprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true)} },
    { ModelType::PFLD, {PFLDNS::PFLDPreprocess, PFLDNS::PFLDPostprocess} },
    { ModelType::CrowdCount, {CrowdCountNS::CrowdCountPreprocess, CrowdCountNS::CrowdCountPostprocess} },
    { ModelType::CRNN, {CRNNPreprocess, CRNNPostProcess} },
    { ModelType::VDSR, {VDSRPreprocess, VDSRPostProcess} },
    { ModelType::HRNet, {HRNetPreprocess, HrnetPostprocess} },
    { ModelType::TinySam, {nullptr, nullptr} },
    { ModelType::MiniCPM, {nullptr, nullptr} },
    { ModelType::ClipImg, {ClipImgPreprocess, Clip::ClipImgPostprocess} },
    { ModelType::ClipTxt, {std::bind(ClipTxtPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), Clip::ClipTxtPostprocess} },
    { ModelType::SuperPoint, {SuperPointPreprocess, SuperPointPostprocess}},
    { ModelType::SqueezeNet1_1, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::DenseNet121, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::ShuffleNetV2, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::MobileNetV2, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::ResNet18, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::ResNet101, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, true)), PrintTop5AndDumpResult} },
    { ModelType::DepthAnythingV2, { std::bind(DepthAnythingV2Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, true), DepthAnythingV2Postprocess } },
    { ModelType::Custom, {nullptr, nullptr} }
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
    { ModelType::FaceNet, {FaceNetNS::FaceNetPreprocess, FaceNetNS::FaceNetPostprocess} },
    { ModelType::Yolov4, {Yolo4::Yolov4Preprocess, Yolo4::Yolov4Postprocess} },
    { ModelType::FastSpeech2, {FastSpeech2NS::FastSpeech2Preprocess, FastSpeech2NS::FastSpeech2Postprocess} },
    { ModelType::CodeFormer, {CodeFormerNS::CodeFormerPreprocess, CodeFormerNS::CodeFormerPostprocess} },
    { ModelType::SiameseNetwork, {SiameseNetworkPreprocess, SiamesePostProcess} },
    { ModelType::SuperPoint, {SuperPointPreprocess, SuperPointPostprocess}},
    { ModelType::PFLD, {PFLDNS::PFLDPreprocess, PFLDNS::PFLDPostprocess} },
    { ModelType::CrowdCount, {CrowdCountNS::CrowdCountPreprocess, CrowdCountNS::CrowdCountPostprocess} },
    { ModelType::CRNN, {CRNNPreprocess, CRNNPostProcess} },
    { ModelType::VDSR, {VDSRPreprocess, VDSRPostProcess} },
    { ModelType::CrowdCount, {CrowdCountNS::CrowdCountPreprocess, CrowdCountNS::CrowdCountPostprocess} },
    { ModelType::HRNet, {HRNetPreprocess, HrnetPostprocess} },
    { ModelType::PaddleOCR_Det, {OcrDetPreprocess, OcrDetPostprocess} },
    { ModelType::PaddleOCR_Rec, { std::bind(OcrRecPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), 
                    std::bind(OcrRecPostprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false)} },
    { ModelType::GraspNet, {Grasp::GraspNetPreprocess, Grasp::GraspNetPostprocess} },
    { ModelType::Pi0, {Pi0Preprocess, Pi0Postprocess} },
    { ModelType::SqueezeNet1_1, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                    ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::DenseNet121, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                    ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::ShuffleNetV2, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::MobileNetV2, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::ResNet18, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::ResNet101, {std::bind(ImageProcess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                        ImageprocessOptions(256, 224, false)), PrintTop5AndDumpResult} },
    { ModelType::DepthAnythingV2, { std::bind(DepthAnythingV2Preprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), DepthAnythingV2Postprocess } },
    { ModelType::ClipImg, {ClipImgPreprocess, Clip::ClipImgPostprocess} },
    { ModelType::ClipTxt, {std::bind(ClipTxtPreprocess, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, false), Clip::ClipTxtPostprocess} },
    { ModelType::Custom, {nullptr, nullptr} }
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
        param.fileLists = {{filePath}};
    }
    if (param.loop > 0) {
        inBuf.emplace_back(4, 0);
    }
    size_t loop = param.loop > 0 ? param.loop : 1;
    int processLoop = param.processLoop > 0 ? param.processLoop : 1;
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
    std::vector<Tensor> output;
    for (size_t k = 0; k < mdlOutputDescs_.size(); ++k) {
        output.push_back(Tensor(outBuf[k].DeepCopy(), mdlOutputDescs_[k]));
    }
    outputs.push_back(output);

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

std::vector<TensorBuf> Model::Infer(std::vector<TensorBuf>& tensorBufs)
{
    std::vector<TensorBuf> inBuf, outBuf;
    if (tensorBufs.size() != mdlInputDescs_.size()) {
        LOG(ERROR) << "invalid input tensor num";
        return {};
    }
    for (size_t i = 0; i < mdlInputDescs_.size(); ++i) {
        inBuf.emplace_back(tensorBufs[i].GetRawPtr(), tensorBufs[i].size, tensorBufs[i].stride);
    }
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize, mdlOutputDescs_[i].defaultStride);
    }
    if (mdl_->Execute(inBuf, outBuf) != 0) {
        LOG(ERROR) << "failed to execute model";
        return {};
    }
    return outBuf;
}

int Model::Infer(std::vector<TensorBuf>& inBufs, std::vector<TensorBuf>& outBufs)
{
    if (mdl_->Execute(inBufs, outBufs) != 0) {
        LOG(ERROR) << "failed to execute model";
        return 0;
    }
    return 0;
}

std::pair<std::vector<TensorDesc>, std::vector<TensorDesc>> Model::GetModelInfo()
{
    return std::make_pair(mdlInputDescs_, mdlOutputDescs_);
}

std::vector<Tensor> Model::Infer(std::vector<Tensor>& tensors, std::string filePath)
{
    std::vector<TensorBuf> inBuf, outBuf;
    std::vector<Tensor> outputs;
    for (size_t i = 0; i < mdlOutputDescs_.size(); ++i) {
        outBuf.emplace_back(mdlOutputDescs_[i].defaultSize, mdlOutputDescs_[i].defaultStride);
    }
    for (size_t i = 0; i < mdlInputDescs_.size(); ++i) {
        inBuf.emplace_back(mdlInputDescs_[i].defaultSize, mdlInputDescs_[i].defaultStride);
    }
    std::vector<Tensor> input;
    for (size_t k = 0; k < mdlInputDescs_.size(); ++k) {
        input.push_back(Tensor(inBuf[k], mdlInputDescs_[k]));
    }
    for (size_t k = 0; k < tensors.size(); ++k) {
        inBuf.push_back(tensors[k].buf);
    }
    LOG(INFO) << "input.size : " << input.size();
    std::vector<std::string> filePaths;
    filePaths.push_back(filePath);
    if (preprocessFunc_ == nullptr || !preprocessFunc_(filePaths, inBuf, mdlInputDescs_)) {
            LOG(ERROR) << "failed to preprocess model input";
            return {};
        }
    for (size_t k = 0; k < tensors.size(); ++k) {
        inBuf.pop_back();
    }
    if (mdl_->Execute(inBuf, outBuf) != 0) {
        LOG(ERROR) << "failed to execute model";
        return {};
    }
    for (size_t k = 0; k < mdlOutputDescs_.size(); ++k) {
        outputs.push_back(Tensor(outBuf[k].DeepCopy(), mdlOutputDescs_[k]));
    }
    if (postprocessFunc_ == nullptr || !postprocessFunc_(filePaths, outBuf, mdlOutputDescs_)) {
            LOG(ERROR) << "failed to postprocess model output";
            return {};
        }

    return outputs;
}
}
