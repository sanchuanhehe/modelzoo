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

#include "pi0_preprocess.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "log.h"
#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

namespace Infer {
static constexpr int MIN_FILELIST_INPUT_NUM = 3;
static constexpr int MAX_FILELIST_INPUT_NUM = 4;
static constexpr int BYTE_BIT_NUM = 8;
static constexpr int FP16_BIT_NUM = 16;
static constexpr int FP32_BIT_NUM = 32;
static constexpr float PIXEL_NORMALIZE_FACTOR = 255.0f;
static constexpr int CHANNEL_NUM = 3;
static constexpr float NORMALIZE_SCALE = 1.0f;

static constexpr int LANG_TOKENS_INBUF_IDX = 0;
static constexpr int LANG_MASKS_INBUF_IDX = 1;
static constexpr int STATE_INBUF_IDX = 2;
static constexpr int IMAGE_INBUF_IDX = 3;

static constexpr int LANG_FILELIST_IDX = 0;
static constexpr int STATE_FILELIST_IDX = 1;
static constexpr int IMAGE_FILELIST_IDX = 2;

static std::vector<int> EncodeLanguage(const std::string& text, const std::string& modelPath)
{
    std::vector<int> tokenIds;
#ifdef USE_SENTENCEPIECE
    // 初始化 SentencePieceProcessor
    sentencepiece::SentencePieceProcessor tokenizer;
    
    // 加载 PaliGemma tokenizer 模型文件（替换为你的实际路径）
    auto status = tokenizer.Load(modelPath);
    
    // 检查模型加载是否成功
    if (!status.ok()) {
        LOG(ERROR) << "load tokenizer fail: " << status.ToString();
        return tokenIds;
    }
    LOG(INFO) << "Input text: " << text;

    // 获取 <bos> 的 ID（手动添加 <bos> 标记，匹配 Python 行为）
    int bosId = tokenizer.PieceToId("<bos>");
    if (bosId == tokenizer.unk_id()) {
        LOG(ERROR) << "Do not found <bos> label，please check tokenizer model";
        return tokenIds;
    }
    // 添加 <bos> 到 tokenIds 开头
    tokenIds.push_back(bosId);

    // 编码纯文本
    std::vector<int> textIds;  // 先存储纯文本的编码结果
    status = tokenizer.Encode(text, &textIds);
    if (!status.ok()) {
        LOG(ERROR) << "Encode text fail: " << status.ToString();
        return tokenIds;
    }
    // 将纯文本编码结果追加到 tokenIds 中（拼接 <bos> + 文本 ID）
    tokenIds.insert(tokenIds.end(), textIds.begin(), textIds.end());
#endif
    return tokenIds;
}

static Result BuildLangInputsToBuf(const std::vector<int>& tokenIds, size_t maxLen,
    int64_t padTokenId, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
{
    // 1. 截断或者补齐 token 序列
    std::vector<int64_t> tokens;
    // 取前maxLen个 token
    int tokenLen = std::min(static_cast<int>(tokenIds.size()), static_cast<int>(maxLen));
    tokens.assign(tokenIds.begin(), tokenIds.begin() + tokenLen);
    // 不足maxLen时用padTokenId填充
    if (tokens.size() < maxLen) {
        tokens.resize(maxLen, padTokenId);
    }

    // 2. 构造token张量
    std::vector<std::vector<int64_t>> langTokens;
    langTokens.push_back(tokens);  // 外层 vector 对应 batch 维度（这里固定为 1）

    // 3. 生成mask掩码(True表示有效token，false表示padding）
    std::vector<std::vector<uint8_t>> langMasks;
    std::vector<uint8_t> mask;
    for (int64_t token : tokens) {
        mask.push_back(token != padTokenId ? 1 : 0);
    }
    langMasks.push_back(mask);

    // 4. 拷贝langTokens和langMasks到inBuf
    if (inDescs[LANG_TOKENS_INBUF_IDX].defaultSize != langTokens[0].size() * sizeof(int64_t)) {
        LOG(ERROR) << "Input size not match, defaultSize :" << inDescs[LANG_TOKENS_INBUF_IDX].defaultSize <<
            " langTokens size: " << langTokens[0].size() * sizeof(int64_t);
        return FAILED;
    }
    DevMemcpy(inBufs[LANG_TOKENS_INBUF_IDX].GetRawPtr(), inDescs[LANG_TOKENS_INBUF_IDX].defaultSize,
        langTokens[0].data(), inDescs[LANG_TOKENS_INBUF_IDX].defaultSize);

    if (inDescs[LANG_MASKS_INBUF_IDX].defaultSize != langMasks[0].size() * sizeof(uint8_t)) {
        LOG(ERROR) << "Input size not match, defaultSize :" << inDescs[LANG_MASKS_INBUF_IDX].defaultSize <<
            " langMasks size: " << langMasks[0].size() * sizeof(uint8_t);
        return FAILED;
    }
    DevMemcpy(inBufs[LANG_MASKS_INBUF_IDX].GetRawPtr(), inDescs[LANG_MASKS_INBUF_IDX].defaultSize,
        langMasks[0].data(), inDescs[LANG_MASKS_INBUF_IDX].defaultSize);

    return SUCCESS;
}

static Result EncodeLanguageToBuf(const std::string& text, const std::string& modelPath,
    size_t tokenLen, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
{
    std::vector<int> tokenIds;
    tokenIds = EncodeLanguage(text, modelPath);
    if (tokenIds.size() == 0) {
        LOG(ERROR) << "Encode language fail";
        return FAILED;
    }

    // 打印 Input IDs
    LOG(INFO) << "Input token IDs: ";
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        LOG(INFO) << tokenIds[i];
    }
    
    if (BuildLangInputsToBuf(tokenIds, tokenLen, 0, inBufs, inDescs) != SUCCESS) {
        LOG(ERROR) << "Failed to fill tensor buffer for: " << text;
        return FAILED;
    }

    return SUCCESS;
}

static Result ReadStateToBuf(const std::string& path, std::vector<TensorBuf>& inBufs,
    std::vector<TensorDesc>& inDescs)
{
    std::vector<float> state;
    const size_t dataSize = inDescs[STATE_INBUF_IDX].typeSize / BYTE_BIT_NUM;
    state.resize(inDescs[STATE_INBUF_IDX].defaultSize / dataSize);
    LOG(INFO) << "state dims: " << state.size();

    std::ifstream infile(path); // 打开文件
    if (!infile.is_open()) {
        LOG(ERROR) << "Read state fail, file: " << path;
        return FAILED;
    }

    for (size_t i = 0; i < state.size(); i++) {
        infile >> state[i];
    }
    infile.close();

    if (inDescs[STATE_INBUF_IDX].defaultSize != state.size() * sizeof(float)) {
        LOG(ERROR) << "Input size not match, defaultSize :" << inDescs[STATE_INBUF_IDX].defaultSize <<
            " state size: " << state.size() * sizeof(float);
        return FAILED;
    }
    DevMemcpy(inBufs[STATE_INBUF_IDX].GetRawPtr(), inDescs[STATE_INBUF_IDX].defaultSize,
        state.data(), inDescs[STATE_INBUF_IDX].defaultSize);

    return SUCCESS;
}

static cv::Mat HwcToChw(const cv::Mat& hwcImg)
{
    CV_Assert(!hwcImg.empty() && hwcImg.channels() == CHANNEL_NUM);
    std::vector<cv::Mat> channels;
    cv::split(hwcImg, channels);

    cv::Mat chwMat(CHANNEL_NUM, hwcImg.rows * hwcImg.cols, CV_32F);
    chwMat = cv::Scalar(0);

    for (int channelIdx = 0; channelIdx < CHANNEL_NUM; ++channelIdx) {
        cv::Mat flatChannel = channels[channelIdx].reshape(1, 1).clone();
        flatChannel.convertTo(flatChannel, CV_32F);
        flatChannel.copyTo(chwMat.row(channelIdx));
    }

    return chwMat;
}

static Result ReadImgFileToBuf(cv::Mat &chwImg, TensorDesc &desc, TensorBuf &inBuf)
{
    LOG(INFO) << "ReadImgFileToBuf: desc.dimCount " << desc.dimCount;
    const float *srcData = chwImg.ptr<float>();

    if (desc.defaultStride == 0) {
        if (desc.typeSize == FP32_BIT_NUM) {
            memcpy(static_cast<float *>(inBuf.GetRawPtr()), srcData, desc.defaultSize);
            return SUCCESS;
        } else if (desc.typeSize == FP16_BIT_NUM){
            // 计算数据量
            size_t dataCount = chwImg.total();
            // 转换为FP16
            std::vector<uint16_t> fp16Data(dataCount);
            for (size_t i = 0; i < dataCount; ++i) {
                fp16Data[i] = FloatToHalf(srcData[i]);
            }
            memcpy(static_cast<void*>(inBuf.GetRawPtr()), fp16Data.data(), desc.defaultSize);
            return SUCCESS;
        } else {
            LOG(ERROR) << "Unsupported desc.typeSize: " << desc.typeSize;
            return FAILED;
        }
    }
    LOG(ERROR) << "Read img file To buf fail";
    return FAILED;
}

static Result ReadImageToBuf(std::vector<std::string>& fileList, int imgWidth, int imgHeight,
    std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
{
    for (size_t i = IMAGE_FILELIST_IDX, j = IMAGE_INBUF_IDX; i < fileList.size() && j < inDescs.size(); ++i, ++j) {
        std::string imgPath = fileList[i];
        LOG(INFO) << "Processing image: " << imgPath;
    
        cv::Mat im0 = cv::imread(imgPath);
        if (im0.empty()) {
            LOG(ERROR) << "Read image failed: " << imgPath;
            return FAILED;
        }

        // 调整尺寸
        cv::Mat resized;
        cv::resize(im0, resized, cv::Size(imgWidth, imgHeight), 0, 0, cv::INTER_LINEAR);

        // BGR转RGB
        cv::Mat rgbImg;
        cv::cvtColor(resized, rgbImg, cv::COLOR_BGR2RGB);

        // HWC转CHW
        cv::Mat chwImg = HwcToChw(rgbImg);

        if (!chwImg.isContinuous()) {
            chwImg = chwImg.clone();
        }
        // 归一化放到NPU侧处理
        // 填充tensor缓冲区
        if (ReadImgFileToBuf(chwImg, inDescs[j], inBufs[j]) != SUCCESS) {
            LOG(ERROR) << "Failed to fill tensor buffer for: " << imgPath;
            return FAILED;
        }
    }

    return SUCCESS;
}

bool Pi0Preprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs)
{
    auto cfg = ReadCfgFile("../data/cfg.txt");
    int imgWidth = std::stoi(cfg["img_width"]);
    int imgHeight = std::stoi(cfg["img_height"]);
    int tokenLen = std::stoi(cfg["token_len"]);
    std::string tokenModelPath = cfg["token_model_path"];

    if (imgWidth < 0 || imgHeight < 0 || tokenLen < 0) {
        LOG(ERROR) << "ImgWidth: " << imgWidth << " or imgHeight: " << imgHeight <<
            " or tokenLen: " << tokenLen << " is invalid";
        return false;
    }

    if (fileList.size() < MIN_FILELIST_INPUT_NUM || fileList.size() > MAX_FILELIST_INPUT_NUM) {
        LOG(ERROR) << "The number of input is wrong, please check fileList_1.json " << fileList.size();
        return false;
    }
    
    if (EncodeLanguageToBuf(fileList[LANG_FILELIST_IDX], tokenModelPath, tokenLen, inBufs, inDescs) != SUCCESS) {
        LOG(ERROR) << "Failed to fill tensor buffer for lang";
        return false;
    }

    if (ReadStateToBuf(fileList[STATE_FILELIST_IDX], inBufs, inDescs) != SUCCESS) {
        LOG(ERROR) << "Failed to fill tensor buffer for state";
        return false;
    }

    if (ReadImageToBuf(fileList, imgWidth, imgHeight, inBufs, inDescs) != SUCCESS) {
        LOG(ERROR) << "Failed to fill tensor buffer for image";
        return false;
    }

    return true;
}

}