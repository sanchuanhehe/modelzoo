
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

#include "log.h"
#include "minicpm_postprocess.h"
#include "minicpm_preprocess.h"
#include "model.h"
#include "utils.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>

using namespace Infer;
using namespace std;

constexpr const char* CFG_PATH = "../data/cfg.txt";
constexpr int QUESTION_PAIR_ITEM_NUM = 2;
constexpr int ID_0 = 0;
constexpr int ID_1 = 1;
constexpr int ZERO = 0;

class MiniCpmInfer {
public:
    MiniCpmInfer();
    void UnLoad();
    string InferSingle(const string& imgPath, const string& text);
    int GetTTFT();
    float GetTPS();
private:
    unique_ptr<Model> visionModel_;
    unique_ptr<Model> resampleModel_;
    unique_ptr<Model> prefillModel_;
    unique_ptr<Model> decodeModel_;
    vector<TensorBuf> visionInputBufs_;
    vector<TensorBuf> resampleInputBufs_;
    vector<TensorBuf> prefillInputBufs_;
    vector<TensorBuf> prefillOutputBufs_;
    vector<TensorBuf> decodeInputBufs_;
    vector<TensorBuf> decodeOutputBufs_;
    pair<vector<TensorDesc>, vector<TensorDesc>> prefillModelInputInfo_;
    pair<vector<TensorDesc>, vector<TensorDesc>> decodeModelInputInfo_;
    string resultDir_;
    int questionNum_ = 0;
    float sumTTFT_ = 0.0f;
    long long decodeTokenNum_ = 0;
    float sumDecodeTokenTime_ = 0.0f;
};