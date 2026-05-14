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

#include "minicpm_preprocess.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cstring>
#include <limits.h>
#include <random>

using json = nlohmann::json;
using namespace std;
using namespace Infer;

Vocab vocabData;
Embedding embeddingData;

// 特殊Token ID常量定义
namespace SpecialTokens {
constexpr int32_t BOS_TOKEN = 1; // <s>
constexpr int32_t IM_START_TOKEN = 73441; // <|im_start|>
constexpr int32_t USER_TOKEN = 3060; // ▁user
constexpr int32_t NEWLINE_TOKEN = 5; // \n
constexpr int32_t IMAGE_ID_TOKEN = 113; // <image_id>
constexpr int32_t UNDERSCORE_0_TOKEN = 59320; // ▁0
constexpr int32_t IMAGE_ID_END_TOKEN = 59344; // </image_id>
constexpr int32_t SPACE_TOKEN = 114; // 空格
constexpr int32_t IMAGE_TOKEN = 101; // <image>
constexpr int32_t IMAGE_END_TOKEN = 102; // </image>
constexpr int32_t IM_END_TOKEN = 73440; // <|im_end|>
constexpr int32_t ASSISTANT_TOKEN = 16434; // ▁assistant
}

// 模板长度常量
namespace TemplateParams {
constexpr int32_t MAX_TEXT_LEN = 118; // 最大文本长度
constexpr int32_t TOTAL_PREFILL_LEN = 200; // 总prefill长度
constexpr int32_t PRE_TEMPLATE_LEN = 9; // 前置模板长度
constexpr int32_t MID_TEMPLATE_LEN = 3; // 中间模板长度
constexpr int32_t POST_TEMPLATE_LEN = 6; // 后置模板长度
}

static vector<float> ReshapeByPatch(const vector<float>& input, int32_t channelNum, int32_t height, int32_t width,
    int32_t patchSize)
{
    int32_t patchNumW = width / patchSize;
    int32_t patchNumH = height / patchSize;
    int32_t patchNum = patchNumW * patchNumH;

    int32_t outWidth = patchNum * patchSize;
    vector<float> output(channelNum * patchSize * outWidth, 0.0f);

    for (int32_t c = 0; c < channelNum; ++c) {
        for (int32_t ph = 0; ph < patchSize; ++ph) {
            for (int32_t pw = 0; pw < patchSize; ++pw) {
                for (int32_t i = 0; i < patchNumH; ++i) {
                    for (int32_t j = 0; j < patchNumW; ++j) {
                        int32_t inputIndex = c * height * width + (i * patchSize + ph) * width + (j * patchSize + pw);
                        int32_t outputIndex = (c * patchSize * outWidth) + ph * outWidth +
                            (i * patchNumW + j) * patchSize + pw;
                        output[outputIndex] = input[inputIndex];
                    }
                }
            }
        }
    }
    return output;
}

static void PreProcessImage(const string& imagePath, vector<float>& pixelValues)
{
    // Read image from path
    const char* fileNames = imagePath.c_str();
    char path[PATH_MAX + 1] = { 0 };
    if (realpath(fileNames, path) == NULL) {
        LOG(ERROR) << "Failed to get real path of " << fileNames;
        return;
    }
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        LOG(ERROR) << "Failed to read image from path: " << path;
        return;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    if (image.rows != SOURCE_TGT_SIZE || image.cols != SOURCE_TGT_SIZE) {
        cv::resize(image, image, cv::Size(SOURCE_TGT_SIZE, SOURCE_TGT_SIZE), 0, 0, cv::INTER_CUBIC);
    }

    // hwc->chw and do normalize
    int32_t mapSize = image.rows * image.cols;
    int32_t totalSize = mapSize * CHANNEL_NUM;
    vector<float> floatValues(totalSize);

    for (int32_t y = 0; y < image.rows; ++y) {
        for (int32_t x = 0; x < image.cols; ++x) {
            int32_t index = y * image.cols + x;
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            for (int32_t c = 0; c < CHANNEL_NUM; ++c) {
                floatValues[index] = (static_cast<float>(pixel[c]) / NORMALIZE_VALUE - HALF_VALUE) / HALF_VALUE;
                index += mapSize;
            }
        }
    }

    pixelValues = ReshapeByPatch(floatValues, CHANNEL_NUM, image.rows, image.cols, PATCH_SIZE);
}

void LoadVocab(const string& filePath)
{
    string realPath;
    char path[PATH_MAX + 1] = { 0 };
    if (realpath(filePath.c_str(), path) == nullptr) {
        LOG(ERROR) << "Failed to get real path of vocab file";
        return;
    }
    realPath = path;
    ifstream file(realPath);
    if (!file.is_open()) {
        LOG(ERROR) << "Failed to open vocab file: " << realPath.c_str();
        return;
    }
    json jsonFile;
    file >> jsonFile;
    auto model = jsonFile["model"];
    vocabData.vocab_ = model["vocab"].get<unordered_map<string, int>>();
    for (auto& item : vocabData.vocab_) {
        const string& token = item.first;
        int id = item.second;
        vocabData.idToToken_[id] = token;
    }
    if (model.contains("unk_token") && !model["unk_token"].is_null()) {
        vocabData.unkToken_ = model["unk_token"];
    }
    if (vocabData.vocab_.find(vocabData.unkToken_) != vocabData.vocab_.end()) {
        vocabData.unkTokenId_ = vocabData.vocab_[vocabData.unkToken_];
    }
    for (auto& item : vocabData.vocab_) {
        const string& token = item.first;
        if (token.length() > vocabData.maxTokenLen_) {
            vocabData.maxTokenLen_ = token.length();
        }
    }
    LOG(INFO) << "Vocabulary loaded successfully. Total tokens: " << vocabData.vocab_.size() << ", Max token length: "
        << vocabData.maxTokenLen_;
}

void LoadVocabEmbeddingTable(const std::string& binFileName, uint32_t dimSize, int32_t maxVocabIndex)
{
    embeddingData.vocabEmbeddingTable_ = make_shared<ifstream>(binFileName, ios::binary);
    if (!embeddingData.vocabEmbeddingTable_->is_open()) {
        LOG(ERROR) << "Failed to open embedding table file: " << binFileName.c_str();
    }

    embeddingData.vocabEmbeddingDimSize_ = dimSize;
    embeddingData.vocabEmbeddingMaxIndex_ = maxVocabIndex;
}

void LoadRotaryPostionEmbeddingTable(const std::string& binFileName0, const std::string& binFileName1,
    uint32_t dimSize, int32_t maxPositionIndex)
{
    embeddingData.rotaryPositionEmbeddingTable0_ = make_shared<ifstream>(binFileName0, ios::binary);
    if (!embeddingData.rotaryPositionEmbeddingTable0_->is_open()) {
        LOG(ERROR) << "Failed to open embedding table file: " << binFileName0.c_str();
    }
    embeddingData.rotaryPositionEmbeddingTable1_ = make_shared<ifstream>(binFileName1, ios::binary);
    if (!embeddingData.rotaryPositionEmbeddingTable1_->is_open()) {
        LOG(ERROR) << "Failed to open embedding table file: " << binFileName1.c_str();
    }

    embeddingData.rotaryPositionEmbeddingDimSize_ = dimSize;
    embeddingData.rotaryPositionEmbeddingMaxIndex_ = maxPositionIndex;
}

static void LookUpVocabTable(float* data, const int32_t* ids, uint32_t idsLen)
{
    for (uint32_t i = 0; i < idsLen; i++) {
        if (ids[i] < 0 || ids[i] >= embeddingData.vocabEmbeddingMaxIndex_) {
            LOG(ERROR) << "idx is out of range";
            return;
        }
        int32_t blockByteSize = embeddingData.vocabEmbeddingDimSize_ * sizeof(float);
        std::streampos offset = static_cast<std::streampos>(ids[i]) * blockByteSize;
        (*embeddingData.vocabEmbeddingTable_).seekg(offset, std::ios::beg);
        if (!(*embeddingData.vocabEmbeddingTable_)) {
            LOG(ERROR) << "Failed to seek in file at offset";
            return;
        }
        (*embeddingData.vocabEmbeddingTable_).read(reinterpret_cast<char*>(data), blockByteSize);
        data += embeddingData.vocabEmbeddingDimSize_;
    }
}

static void TokenizerEncode(const string& text, vector<int32_t>& tokenLists)
{
    if (vocabData.vocab_.empty()) {
        LOG(ERROR) << "Vocabulary is empty. Please load a valid vocab file first.";
        return;
    }

    vector<string> tokens;
    string curText = text;
    while (!curText.empty()) {
        bool found = false;
        int32_t maxLen = static_cast<int32_t>(min(vocabData.maxTokenLen_, curText.length()));
        for (int32_t len = maxLen; len > 0; --len) {
            string candidate = curText.substr(0, len);
            if (vocabData.vocab_.find(candidate) != vocabData.vocab_.end()) {
                tokens.push_back(candidate);
                curText = curText.substr(len);
                found = true;
                break;
            }
        }
        if (!found) {
            tokens.push_back(curText.substr(0, 1)); // Use unknown token
            curText = curText.substr(1);
        }
    }

    vector<int32_t> ids;

    for (const auto& token : tokens) {
        auto it = vocabData.vocab_.find(token);
        if (it != vocabData.vocab_.end()) {
            ids.push_back(vocabData.vocab_[token]);
        } else {
            ids.push_back(0); // Use unknown token ID
        }
    }

    for (auto word : ids) {
        tokenLists.push_back(word);
    }
}

static void SetInputEmbedding(float* inData, const TensorBuf& resampleBuf, const std::string& text,
    int& prefillLen)
{
    int m_prefillLen = 0;
    const int32_t preTemplate[] = {
        SpecialTokens::BOS_TOKEN,
        SpecialTokens::IM_START_TOKEN,
        SpecialTokens::USER_TOKEN,
        SpecialTokens::NEWLINE_TOKEN,
        SpecialTokens::IMAGE_ID_TOKEN,
        SpecialTokens::UNDERSCORE_0_TOKEN,
        SpecialTokens::IMAGE_ID_END_TOKEN,
        SpecialTokens::SPACE_TOKEN,
        SpecialTokens::IMAGE_TOKEN
    }; // <s><|im_start|>▁user\n<image_id>▁0</image_id><image>
    constexpr int32_t preTemplateLen = TemplateParams::PRE_TEMPLATE_LEN;
    LookUpVocabTable(inData, preTemplate, preTemplateLen);
    inData += (preTemplateLen * EMB_DATA_OFFSET);
    m_prefillLen += preTemplateLen;

    if (resampleBuf.size / sizeof(float) != VISION_TOKEN_LEN * EMB_DATA_OFFSET) {
        LOG(ERROR) << "Vision input size is not valid, resampleBuf size: " << resampleBuf.size
            << "VISION_TOKEN_LEN * EMB_DATA_OFFSET: " << VISION_TOKEN_LEN * EMB_DATA_OFFSET;
    }
    memcpy(inData, resampleBuf.GetRawPtr(), resampleBuf.size);
    inData += resampleBuf.size / sizeof(float);
    m_prefillLen += VISION_TOKEN_LEN;

    const int32_t midTemplate[] = {
        SpecialTokens::IMAGE_END_TOKEN,
        SpecialTokens::UNDERSCORE_0_TOKEN,
        SpecialTokens::NEWLINE_TOKEN
    }; // </image>▁\n
    constexpr int32_t midTemplateLen = TemplateParams::MID_TEMPLATE_LEN;
    LookUpVocabTable(inData, midTemplate, midTemplateLen);
    inData += (midTemplateLen * EMB_DATA_OFFSET);
    m_prefillLen += midTemplateLen;

    std::vector<int32_t> inputIds;
    TokenizerEncode(text, inputIds);
    if (inputIds.size() > TemplateParams::MAX_TEXT_LEN) {
        LOG(INFO) << "Text length[" << inputIds.size() << "] is resized to " << TemplateParams::MAX_TEXT_LEN << ".";
        inputIds.resize(TemplateParams::MAX_TEXT_LEN);
    }
    LookUpVocabTable(inData, inputIds.data(), inputIds.size());
    inData += (inputIds.size() * EMB_DATA_OFFSET);
    m_prefillLen += inputIds.size(); // Total 200 tokens, so text tokens num is (200 - 9 - 64 - 3 - 6) = 118

    const int32_t postTemplate[] = {
        SpecialTokens::IM_END_TOKEN,
        SpecialTokens::UNDERSCORE_0_TOKEN,
        SpecialTokens::NEWLINE_TOKEN,
        SpecialTokens::IM_START_TOKEN,
        SpecialTokens::ASSISTANT_TOKEN,
        SpecialTokens::NEWLINE_TOKEN
    }; // <|im_end|>▁\n<|im_start|>▁assistant\n
    constexpr int32_t postTemplateLen = TemplateParams::POST_TEMPLATE_LEN;
    LookUpVocabTable(inData, postTemplate, postTemplateLen);
    inData += (postTemplateLen * EMB_DATA_OFFSET);
    m_prefillLen += postTemplateLen;
    prefillLen = m_prefillLen;
}

static void SetAttentionMask(float* inAttentionMask, const TensorDesc& prefillDesc, int prefillLen)
{
    auto inDataSize = prefillDesc.defaultSize / sizeof(float);
    size_t fixLen = prefillDesc.dims[prefillDesc.dimCount - 1];

    if (inDataSize != fixLen * fixLen) {
        LOG(ERROR) << "inAttentionMask size is not valid, should be " << fixLen << "*" << fixLen
            << ", while it is " << inDataSize;
        return;
    }

    for (size_t i = 0; i < fixLen; i++) {
        for (size_t j = 0; j <= i; j++) {
            inAttentionMask[i * fixLen + j] = 0;
        }
        for (size_t j = i + 1; j < fixLen; j++) {
            inAttentionMask[i * fixLen + j] = MASK_MIN_VALUE;
        }
    }
    for (size_t i = prefillLen; i < fixLen; i++) {
        for (size_t j = prefillLen; j <= i; j++) {
            inAttentionMask[i * fixLen + j] = MASK_MIN_VALUE;
        }
    }
}

static void LookUpTable(float* data, std::ifstream& ifs, int32_t idx,
    uint32_t dimSize, int32_t maxIndex)
{
    if (idx < 0 || idx >= maxIndex) {
        LOG(ERROR) << "idx is out of range";
        return;
    }

    int32_t blockByteSize = dimSize * sizeof(float);
    std::streampos offset = static_cast<std::streampos>(idx) * blockByteSize;
    ifs.seekg(offset, std::ios::beg);
    if (!ifs) {
        LOG(ERROR) << "Failed to seek in file at offset";
        return;
    }
    ifs.read(reinterpret_cast<char*>(data), blockByteSize);
}

bool ParseInputJsonFile(const std::string& filePath, ExecuteParam& param)
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

void SaveLog(const string& imgPath, const string& text, const string& result, const string& txtDir)
{
    string sep = "-";
    for (int i = 0; i < SEP_NUM; ++i) {
        sep += sep;
    }
    mode_t old = umask(0);
    std::string content = format("%s\n [Question]: %s\n [Image path]: %s\n [MiniCPM infer result]:\n %s\n%s\n",
        sep.c_str(), text.c_str(), imgPath.c_str(), result.c_str(), sep.c_str());
    ofstream ofs(txtDir + "/result.txt", ios::app);
    if (!ofs.is_open())
        return;
    ofs << content << endl;
    ofs.close();
    umask(old);
}

float TimestampInner()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    float t = tv.tv_sec * TIME_CHANGE + tv.tv_usec / TIME_CHANGE; // ms
    return t;
}

int VisionPreprocess(vector<TensorBuf>& inBuf, const string& imgPath)
{
    vector<float> pixelValues;
    PreProcessImage(imgPath, pixelValues);
    memcpy(inBuf[0].GetRawPtr(), pixelValues.data(), pixelValues.size() * sizeof(float));
    return 0;
}

int ResamplePreprocess(vector<TensorBuf>& inBuf, const vector<TensorBuf>& fromBuf)
{
    inBuf[0] = fromBuf[0];
    return 0;
}

int PrefillPreprocess(vector<TensorBuf>& inBuf, const vector<TensorBuf>& fromBuf,
    const vector<TensorDesc>& inDesc, int& prefillLen, const string& text)
{
    float* inData = static_cast<float*>(inBuf[0].GetRawPtr());
    SetInputEmbedding(inData, fromBuf[0], text, prefillLen);
    float* inAttentionMask = static_cast<float*>(inBuf[1].GetRawPtr());
    SetAttentionMask(inAttentionMask, inDesc[1], prefillLen);
    return 0;
}

int DecodePreprocess(std::vector<Infer::TensorBuf>& decodeInputBufs, const std::vector<int32_t>& tokenIdVec,
    const std::pair<std::vector<Infer::TensorBuf>, std::vector<Infer::TensorBuf>>& fromBufs,
    const std::pair<std::vector<Infer::TensorDesc>, std::vector<Infer::TensorDesc>>& fromDescs,
    const std::pair<int, int>& fromParams)
{
    int loopId = fromParams.first;
    int outputNum = fromParams.second;
    const std::vector<Infer::TensorBuf>& lastDecodeOutBufs = fromBufs.first;
    const std::vector<Infer::TensorBuf>& prefillOutBufs = fromBufs.second;
    const std::vector<Infer::TensorDesc>& decodeInputDescs = fromDescs.first;
    const std::vector<Infer::TensorDesc>& prefillOutDescs = fromDescs.second;

    LookUpTable(static_cast<float*>(decodeInputBufs[EMBEDDING_INPUT_ID].GetRawPtr()),
        *embeddingData.vocabEmbeddingTable_, tokenIdVec.back(), embeddingData.vocabEmbeddingDimSize_,
        embeddingData.vocabEmbeddingMaxIndex_);
    auto* inData = static_cast<float*>(decodeInputBufs[LOOP_IDX_INPUT_ID].GetRawPtr());
    inData[0] = loopId;

    LookUpTable(static_cast<float*>(decodeInputBufs[ROTARY_POSITION_0_INPUT_ID].GetRawPtr()),
        *embeddingData.rotaryPositionEmbeddingTable0_, loopId, embeddingData.rotaryPositionEmbeddingDimSize_,
        embeddingData.rotaryPositionEmbeddingMaxIndex_);
    LookUpTable(static_cast<float*>(decodeInputBufs[ROTARY_POSITION_1_INPUT_ID].GetRawPtr()),
        *embeddingData.rotaryPositionEmbeddingTable1_, loopId, embeddingData.rotaryPositionEmbeddingDimSize_,
        embeddingData.rotaryPositionEmbeddingMaxIndex_);

    if (tokenIdVec.size() == 1) {
        auto outDims = prefillOutDescs[0];
        auto keyValueSize = decodeInputBufs[KV_START_INPUT_ID].size / sizeof(float);
        auto inDims = decodeInputDescs[KV_START_INPUT_ID];
        auto layers = outDims.dims[0];
        auto prefillDecodeFixSeq = outDims.dims[1];
        auto decodeFixSeq = inDims.dims[0];

        if (layers == 0 || prefillDecodeFixSeq == 0 || decodeFixSeq == 0)
            return 0;

        if (prefillOutBufs[0].size / sizeof(float) / layers / prefillDecodeFixSeq != keyValueSize / decodeFixSeq) {
            LOG(ERROR) << "PrefillOutBufs[0] size / sizeof(float) / layers / prefillDecodeFixSeq [%d] != \
                keyValueSize / decodeFixSeq [%d]"
                << prefillOutBufs[0].size / sizeof(float) / layers / prefillDecodeFixSeq << keyValueSize / decodeFixSeq;
        }

        if (layers != static_cast<size_t>(outputNum / 2)) {
            LOG(ERROR) << "layers [%d] != outputNum / 2 [%d]" << layers << outputNum / 2;
        }

        auto prefillKeyValueSize = keyValueSize / decodeFixSeq * loopId;
        auto fixPrefillKeyValueSize = keyValueSize / decodeFixSeq * prefillDecodeFixSeq;
        for (size_t i = 0; i < layers; i++) {
            memcpy(static_cast<float*>(decodeInputBufs[KV_START_INPUT_ID + i * 2].GetRawPtr()),
                static_cast<float*>(prefillOutBufs[0].GetRawPtr()) + i * fixPrefillKeyValueSize,
                prefillKeyValueSize * sizeof(float));
            memcpy(static_cast<float*>(decodeInputBufs[KV_START_INPUT_ID + i * 2 + 1].GetRawPtr()),
                static_cast<float*>(prefillOutBufs[1].GetRawPtr()) + i * fixPrefillKeyValueSize,
                prefillKeyValueSize * sizeof(float));
        }

        auto* inAttentionMask = static_cast<float*>(decodeInputBufs[ATTENTION_MASK_INPUT_ID].GetRawPtr());
        size_t loopIdSize = static_cast<size_t>(loopId);
        for (size_t i = 0; i <= loopIdSize; i++) {
            inAttentionMask[i] = 0;
        }
        for (size_t i = loopIdSize + 1; i < decodeInputBufs[ATTENTION_MASK_INPUT_ID].size / sizeof(float); i++) {
            inAttentionMask[i] = MASK_MIN_VALUE;
        }
    } else {
        auto* inAttentionMask = static_cast<float*>(decodeInputBufs[ATTENTION_MASK_INPUT_ID].GetRawPtr());
        inAttentionMask[loopId] = 0;
        // 非初次自回归KV矩阵更新
        for (size_t i = 0; i + 1 < lastDecodeOutBufs.size(); ++i) {
            decodeInputBufs[i + KV_START_INPUT_ID] = lastDecodeOutBufs[i];
        }
    }
    return 0;
}
