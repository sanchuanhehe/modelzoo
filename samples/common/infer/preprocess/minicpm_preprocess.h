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

#pragma once

#include "dev_interface_adapter.h"
#include "log.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

const int32_t BYTE_BIT_NUM = 8;
const double HALF_VALUE = 0.5;
const double NORMALIZE_VALUE = 255.0;
const int32_t SOURCE_TGT_SIZE = 512;
const int32_t PATCH_TGT_SIZE = 1024;
const int32_t CHANNEL_NUM = 3;
const int32_t PATCH_SIZE = 16;
constexpr int32_t VISION_TOKEN_LEN = 64;
constexpr int32_t EMB_DATA_OFFSET = 1024;
constexpr int32_t KEY_OUTPUT_ID = 0;
constexpr int32_t VALUE_OUTPUT_ID = 1;
constexpr int32_t LOGITS_OUTPUT_ID = 2;
constexpr uint32_t EMBEDDING_INPUT_ID = 0;
constexpr uint32_t ATTENTION_MASK_INPUT_ID = 1;
constexpr uint32_t LOOP_IDX_INPUT_ID = 2;
constexpr uint32_t ROTARY_POSITION_0_INPUT_ID = 3;
constexpr uint32_t ROTARY_POSITION_1_INPUT_ID = 4;
constexpr uint32_t KV_START_INPUT_ID = 5;
constexpr int32_t MAX_TOKEN_LEN = 1024;
constexpr int32_t MINICPM_END_ID = 73440;
constexpr int32_t TOTAL_TOKEN_NUM = 73448;
constexpr int32_t ROTARY_POSITION_EMB_DATA_OFFSET = 64;
constexpr int32_t MAX_POSITION_IDX = 32768;
const int32_t CONFIGS_NUM = 6;
constexpr int32_t MASK_MIN_VALUE = -9999999;
constexpr int SEP_NUM = 8;
constexpr float TIME_CHANGE = 1000.0f;

struct Vocab {
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> idToToken_;
    std::string unkToken_ = "<unk>";
    size_t maxTokenLen_ = 0;
    int32_t unkTokenId_ = 0;
};
extern Vocab vocabData; // 只声明不定义

struct Embedding {
    std::shared_ptr<std::ifstream> vocabEmbeddingTable_;
    std::shared_ptr<std::ifstream> rotaryPositionEmbeddingTable0_;
    std::shared_ptr<std::ifstream> rotaryPositionEmbeddingTable1_;
    uint32_t vocabEmbeddingDimSize_;
    int32_t vocabEmbeddingMaxIndex_ { -1 };
    uint32_t rotaryPositionEmbeddingDimSize_;
    int32_t rotaryPositionEmbeddingMaxIndex_ { -1 };
};
extern Embedding embeddingData; // 只声明不定义

struct ExecuteParam {
    size_t loop = 0;
    std::vector<std::vector<std::string>> fileLists;
};

template <typename... Args>
std::string format(const std::string& fmt, Args... args)
{
    int size = std::snprintf(nullptr, 0, fmt.c_str(), args...);
    if (size <= 0)
        return "";

    std::string buf(size, '\0');
    std::snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

void SaveLog(const std::string& imgPath, const std::string& text, const std::string& result, const std::string& txtDir);

int VisionPreprocess(std::vector<Infer::TensorBuf>& inBuf, const std::string& imgPath);

int ResamplePreprocess(std::vector<Infer::TensorBuf>& inBuf, const std::vector<Infer::TensorBuf>& fromBuf);

int PrefillPreprocess(std::vector<Infer::TensorBuf>& inBuf, const std::vector<Infer::TensorBuf>& fromBuf,
    const std::vector<Infer::TensorDesc>& inDesc, int& prefillLen, const std::string& text);

int DecodePreprocess(std::vector<Infer::TensorBuf>& decodeInputBufs, const std::vector<int32_t>& tokenIdVec,
    const std::pair<std::vector<Infer::TensorBuf>, std::vector<Infer::TensorBuf>>& fromBufs,
    const std::pair<std::vector<Infer::TensorDesc>, std::vector<Infer::TensorDesc>>& fromDescs,
    const std::pair<int, int>& fromParams);

void LoadVocab(const std::string& filePath);

void LoadVocabEmbeddingTable(const std::string& binFileName, uint32_t dimSize, int32_t maxVocabIndex);

void LoadRotaryPostionEmbeddingTable(const std::string& binFileName0, const std::string& binFileName1,
    uint32_t dimSize, int32_t maxPositionIndex);

bool ParseInputJsonFile(const std::string& filePath, ExecuteParam& param);

float TimestampInner();
