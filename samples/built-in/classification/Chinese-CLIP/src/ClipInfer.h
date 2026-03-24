
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
#include "clip_postprocess.h"
#include "clip_preprocess.h"
#include "clip_txt_preprocess.h"
#include "model.h"
#include "utils.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>

using namespace Infer;
using namespace std;

constexpr const char *CFG_PATH = "../data/cfg.txt";
constexpr int QUESTION_PAIR_ITEM_NUM = 2;
constexpr int ID_0 = 0;
constexpr int ID_1 = 1;
constexpr int ZERO = 0;

class ClipInfer
{
public:
    ClipInfer();
    std::vector<std::vector<float>> InferTxt();
    std::vector<float> InferImageSingle(std::string &imgFile);
    std::vector<std::vector<float>> InfeTxtPerformance();
    std::vector<float> InferImagePerformance();
    bool ReadTxtToVector(const std::string &filename,
                         std::vector<std::vector<std::string>> &out_lines,
                         bool skip_empty = true);
    std::vector<std::string> GenerateCifar100TemplateTexts(const std::string &cn_name);
    bool ExecuteParams(int argc, char *argv[]);
    void LoadModel();
    void UnLoadModel();
    std::vector<float> ComputeModelLogits(
        std::vector<float> &image_feats,
        std::vector<std::vector<float>> &classifier,
        int feat_dim,
        int num_classes,
        std::string &fileName);

public:
    string imgOmPath_;
    string txtOmPath_;
    string acl_path_;
    string txtList_;
    string imgList_;
    unique_ptr<Model> imgModel_;
    unique_ptr<Model> txtModel_;
    std::vector<std::pair<float, size_t>> gt_;
    std::vector<std::vector<std::string>> imgFileList_;
    std::vector<std::vector<std::string>> txtFileList_;
    string resultDir_;
    bool doAccuary_ = false;
    int questionNum_ = 0;
    int loop_ = 0;
};