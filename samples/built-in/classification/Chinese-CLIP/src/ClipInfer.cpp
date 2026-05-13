/*
 * Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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

#include "ClipInfer.h"
#include <algorithm>  // 必须添加，用于std::sort
#include <cmath>
#include <cstring>
#include <errno.h>  // 可选：用于错误码（如mkdir失败时打印原因）
#include <getopt.h>
#include <string>
#include <sys/stat.h>   // 用于struct stat、mkdir
#include <sys/types.h>  // 用于mode_t（mkdir的权限参数）
#include <unistd.h>     // 兼容部分Linux系统的mkdir声明
#include <vector>

#define MAX_LINE_LEN 1024  // 每行最大长度（可根据文件调整）
constexpr int DEFAULT_BATCH = 1;

// ====================
// 精度验证的时候，会根据lable_cn.txt填充一个模板。有11条模板，然后验证
// ====================

// 文本预处理函数：转小写 + 替换中文引号为英文引号
std::string PreprocessTextClip(const std::string &text)
{
    std::string processedText = text;
    for (auto& pattern : {"“", "”"}) {
        size_t pos = 0;
        while ((pos = processedText.find(pattern, pos)) != std::string::npos) {
            processedText.replace(pos, strlen(pattern), "\"");
            pos += 1;
        }
    }
    return processedText;
}

// 封装成可复用函数
std::vector<std::string> ClipInfer::GenerateCifar100TemplateTexts(const std::string &cn_name)
{
    std::vector<std::function<std::string(const std::string &)>> cifar100Templates = {
        [](const std::string &c) { return "一张" + c + "的照片"; },
        [](const std::string &c) { return "一张" + c + "的模糊照片"; }, [](const std::string &c) { return "一张" + c; },
        [](const std::string &c) { return "一张" + c + "的低对比度照片"; },
        [](const std::string &c) { return "一张" + c + "的高对比度照片"; },
        [](const std::string &c) { return "一张" + c + "的好照片"; },
        [](const std::string &c) { return "一张小" + c + "的照片"; },
        [](const std::string &c) { return "一张大" + c + "的照片"; },
        [](const std::string &c) { return "一张" + c + "的黑白照片"; },
        [](const std::string &c) { return "一张" + c + "的低对比度的照片"; },
        [](const std::string &c) { return "一张" + c + "的高对比度的照片"; }};

    std::vector<std::string> templateTexts;
    for (const auto &func : cifar100Templates) {
        templateTexts.push_back(PreprocessTextClip(func(cn_name)));
    }
    return templateTexts;
}

bool ClipInfer::ReadTxtToVector(
    const std::string &filename, std::vector<std::vector<std::string>> &outLines, bool skipEmpty)
{
    // 清空调用方传入的vector（避免原有数据干扰）
    outLines.clear();

    // 1. 打开文件（C++ ifstream，只读模式）
    std::ifstream file(filename, std::ios::in);
    if (!file.is_open()) {
        LOG(ERROR) << "file open error:" << filename;
        return false;
    }

    // 2. 逐行读取文件
    std::string line;

    while (std::getline(file, line)) {  // std::getline自动剔除换行符\n
        std::vector<std::string> tmp;
        // 处理Windows换行符\r（std::getline不会剔除\r）
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (skipEmpty && line.empty()) {
            continue;
        }
        if (doAccuary_) {
            tmp = GenerateCifar100TemplateTexts(line);
        } else {
            tmp.push_back(line);
        }
        outLines.push_back(tmp);
    }

    // 3. 检查读取过程是否出错
    if (file.bad()) {
        LOG(ERROR) << "IO error";
        file.close();
        outLines.clear();
        return false;
    }

    // 4. 关闭文件并返回结果
    file.close();
    return true;
}

ClipInfer::ClipInfer()
{
}

/**
 * @brief 保存二维float向量为bin文件
 * @param feats 输入的二维向量
 * @param binPath bin文件保存路径
 * @param shapePath
 * 可选：保存维度信息的txt路径（记录行数、列数，用于读取时还原形状）
 * @return 是否保存成功
 */
bool Save2DToBin(
    const std::vector<std::vector<float>> &feats, const std::string &binPath, const std::string &shapePath = "")
{
    if (feats.empty()) {
        LOG(ERROR) << "feats is empty";
        return false;
    }
    size_t rows = feats.size();     // 行数（如5）
    size_t cols = feats[0].size();  // 列数（如1024）
    std::ofstream binFile(binPath, std::ios::out | std::ios::binary);
    if (!binFile.is_open()) {
        std::cerr << "错误：无法打开bin文件 " << binPath << std::endl;
        return false;
    }
    for (const auto &row : feats) {
        binFile.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
    }
    binFile.close();
    return true;
}

std::vector<std::vector<float>> ClipInfer::InferTxt()
{
    std::vector<std::vector<float>> zeroshotWeights;
    std::vector<std::vector<float>> feats;
    std::vector<std::vector<Infer::Tensor>> ret;
    if (!txtModel_) {
        LOG(ERROR) << "infer txt model empty";
        return feats;
    }
    if (loop_) {
        LOG(ERROR) << "only test TXT profiling";
        return InfeTxtPerformance();
    }
    for (size_t i = 0; i < txtFileList_.size(); ++i) {
        feats.clear();
        for (size_t j = 0; j < txtFileList_[i].size(); ++j) {
            ret = txtModel_->Infer(txtFileList_[i][j], FileType::SingelImageFile);
            if (ret.size() == 0) {
                LOG(ERROR) << "fail to infer txt model";
                return feats;
            }
            std::vector<float> result;
            float *outData = static_cast<float *>(ret[0][0].buf.GetRawPtr());
            size_t floatElementCount = ret[0][0].buf.size / sizeof(float);
            result.assign(outData, outData + floatElementCount);
            feats.push_back(result);
        }

        Infer::Clip::L2Normalize(feats);
        std::vector<float> combinedTensor = Infer::Clip::MeanByCol(feats);  // 1x512
        Infer::Clip::NormalizeSingleVec(combinedTensor);
        zeroshotWeights.push_back(combinedTensor);  // classx512
    }
    std::vector<std::vector<float>> final = Infer::Clip::StackDim(zeroshotWeights);
    return final;
}

std::vector<float> ClipInfer::InferImagePerformance()
{
    std::vector<std::vector<float>> zeroshotWeights;
    std::vector<std::vector<float>> feats;
    std::vector<std::vector<Infer::Tensor>> ret;
    std::vector<float> result;
    if (!imgModel_) {
        LOG(ERROR) << "infer img model empty";
        return result;
    }
    ret = imgModel_->Infer(imgList_, FileType::JsonFile);
    if (ret.size() == 0) {
        LOG(ERROR) << "fail to infer img model";
        return result;
    }
    float *outData = static_cast<float *>(ret[0][0].buf.GetRawPtr());
    size_t floatElementCount = ret[0][0].buf.size / sizeof(float);
    result.assign(outData, outData + floatElementCount);
    feats.push_back(result);
    Infer::Clip::L2Normalize(feats);
    return result;
}

std::vector<std::vector<float>> ClipInfer::InfeTxtPerformance()
{
    std::vector<std::vector<float>> zeroshotWeights;
    std::vector<std::vector<float>> feats;
    std::vector<std::vector<Infer::Tensor>> ret;
    if (!txtModel_) {
        LOG(ERROR) << "infer txt single model empty";
        return feats;
    }
    ret = txtModel_->Infer(txtList_, FileType::JsonFile);
    if (ret.size() == 0) {
        LOG(ERROR) << "fail to infer txt model";
        return feats;
    }
    std::vector<float> result;
    float *outData = static_cast<float *>(ret[0][0].buf.GetRawPtr());
    size_t floatElementCount = ret[0][0].buf.size / sizeof(float);
    result.assign(outData, outData + floatElementCount);
    feats.push_back(result);
    Infer::Clip::L2Normalize(feats);
    return feats;
}

void ClipInfer::LoadModel()
{
    if (imgOmPath_.size() > 0) {
        imgModel_ = make_unique<Model>();
        imgModel_->Load(imgOmPath_, ClipImg);
    }
    if (txtOmPath_.size() > 0) {
        txtModel_ = make_unique<Model>();
        txtModel_->Load(txtOmPath_, ClipTxt);
    }
}

void ClipInfer::UnLoadModel()
{
    if (imgModel_) {
        imgModel_->Unload();
        imgModel_ = nullptr;
    }
    if (txtModel_) {
        txtModel_->Unload();
        txtModel_ = nullptr;
    }
}

std::vector<float> ClipInfer::InferImageSingle(std::string &imgFile)
{
    std::vector<std::vector<float>> feats;
    std::vector<std::vector<Infer::Tensor>> ret;
    std::vector<float> result;
    if (!imgModel_) {
        LOG(ERROR) << "infer single img model empty";
        return result;
    }
    if (loop_) {
        LOG(ERROR) << "only test img profiling";
        return InferImagePerformance();
    }
    LOG(INFO) << "inferImg: " << imgFile;
    ret = imgModel_->Infer(imgFile, FileType::SingelImageFile);
    if (ret.size() == 0) {
        LOG(ERROR) << "fail to infer model";
        imgModel_->Unload();
        imgModel_ = nullptr;
        EnvDeinit();
        return result;
    }
    float *outData = static_cast<float *>(ret[0][0].buf.GetRawPtr());
    size_t floatElementCount = ret[0][0].buf.size / sizeof(float);
    result.assign(outData, outData + floatElementCount);
    feats.push_back(result);
    Infer::Clip::L2Normalize(feats);
    return feats[0];
}

/**
 * @brief modelLogits = (100.0 * image_features @ classifier).softmax(dim=-1)
 * @param imageFeats 图像特征（[batch_size, featDim]）
 * @param classifier 分类器权重（[featDim, numClasses]）
 * @param batch_size 批次大小
 * @param featDim 特征维度
 * @param numClasses 类别数
 * @return 最终的概率分布（[batch_size, numClasses]）
 */
std::vector<float> ClipInfer::ComputeModelLogits(std::vector<float> &imageFeats,
    std::vector<std::vector<float>> &classifier, int featDim, int numClasses, std::string &fileName)
{
    if (imageFeats.size() == 0 || classifier.size() == 0) {
        return imageFeats;
    }
    std::vector<float> logits =
        Infer::Clip::MatmulWithScale(imageFeats, classifier, DEFAULT_BATCH, featDim, numClasses);
    std::vector<float> modelLogits = Infer::Clip::Softmax(logits, DEFAULT_BATCH, numClasses);

    static std::string txtPath = resultDir_ + "out";
    struct stat info;
    mode_t oldUmask = umask(0);
    if (stat(resultDir_.c_str(), &info) != 0) {
        mkdir(resultDir_.c_str(), 0777);
    }
    if (stat(txtPath.c_str(), &info) != 0) {
        mkdir(txtPath.c_str(), 0777);
    }
    umask(oldUmask);
    size_t start = fileName.find_last_of("/");
    size_t end = fileName.find_last_of(".");
    std::string name = fileName.substr(start, end - start);
    std::string binFile = txtPath + name + "_0.txt";
    std::vector<std::pair<float, size_t>> result(modelLogits.size());
    for (size_t j = 0; j < modelLogits.size(); ++j) {
        result[j].first = modelLogits[j];
        result[j].second = j;
    }
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    LOG(INFO) << fileName << " inference result top 5:";
    int cnt = 0;
    for (auto it = result.begin(); it != result.end(); ++it) {
        if (++cnt > 5) {
            break;
        }
        LOG(INFO) << "top " << cnt << ": index[" << it->second << "] value[" << it->first << "]";
    }

    std::ofstream fout(binFile, std::ios::out | std::ios::trunc);
    if (fout.good() == false) {
        std::cout << "create output file failed " << binFile << std::endl;
        return modelLogits;
    }
    for (auto &t : result) {
        fout << t.second << "," << t.first << "\n";
    }
    fout.close();
    return modelLogits;
}

bool ClipInfer::ExecuteParams(int argc, char *argv[])
{
    int opt;
    const char *optstring = "hj:t:a:i:l:";
    struct option long_options[] = {{"help", no_argument, NULL, 'h'}, {"imgmodel", required_argument, NULL, 'j'},
        {"txtmodel", required_argument, NULL, 't'}, {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'}, {"loop", required_argument, NULL, 'l'}, {0, 0, 0, 0}};

    std::string aclConfigPath;
    std::string inputPath;
    static int loop = 1;
    while ((opt = getopt_long(argc, argv, optstring, long_options, NULL)) != -1) {
        switch (opt) {
        case 'h':
            INFO_LOG("Usage: [--help] [--model OM_MODEL_PATH] [--acl "
                     "ACL_CONFIG_PATH] [--input IMAGE_DIR] [--loop LOOP_COUNT]");
            return 0;
        case 'j':
            if (optarg) {
                if (!PathToRealPath(optarg, imgOmPath_)) {
                    ERROR_LOG("parse model path error");
                    return 0;
                }
            } else {
                // 如果没有提供参数，设置为空字符串或默认值
                txtOmPath_ = "";  // 或者设置为默认模型路径
                INFO_LOG("No img model specified, using default behavior");
            }
            break;
        case 't':
            if (optarg) {
                if (!PathToRealPath(optarg, txtOmPath_)) {
                    ERROR_LOG("parse text model path error");
                    return 0;
                }
            } else {
                // 如果没有提供参数，设置为空字符串或默认值
                txtOmPath_ = "";  // 或者设置为默认模型路径
                INFO_LOG("No text model specified, using default behavior");
            }
            break;
        case 'a':
            if (!PathToRealPath(optarg, aclConfigPath)) {
                ERROR_LOG("parse acl config path error");
                return 0;
            }
            break;
        case 'i':
            if (!PathToRealPath(optarg, inputPath)) {
                ERROR_LOG("parse image dir error");
                return 0;
            }
            break;
        case 'l': {
            char *endptr = nullptr;
            loop_ = strtoull(optarg, &endptr, 0);
            if (*endptr != '\0') {
                ERROR_LOG("incorrect input after -l/--loop");
                return 0;
            }
            break;
        }
        case '?':
            ERROR_LOG("unknown option:");
            return 0;
        default:
            ERROR_LOG("unexpected error");
            return 0;
        }
    }

    // 初始化
    auto cfg = ReadCfgFile(inputPath);
    txtList_ = cfg["txt_list"];
    imgList_ = cfg["img_list"];
    resultDir_ = cfg["result_dir"];
    LOG(INFO) << "infer " << txtList_ << " img: " << imgList_;
    doAccuary_ = std::stoi(cfg["accuary"]) == 1;
    loop_ = std::stoi(cfg["performance"]) == 1;
    LOG(INFO) << "doAccuary_ " << doAccuary_ << " performance " << loop_;
    imgFileList_ = ParseFileList(imgList_);
    ReadTxtToVector(txtList_, txtFileList_);
    std::cout << " txtFileList_ " << txtFileList_.size() << std::endl;
    return 1;
}
