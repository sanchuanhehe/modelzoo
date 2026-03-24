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

#include "MiniCpmInfer.h"

MiniCpmInfer::MiniCpmInfer()
{
    auto cfg = ReadCfgFile(CFG_PATH);
    string vision_om_path = cfg["vision_om_path"];
    string resample_om_path = cfg["resample_om_path"];
    string prefill_decode_om_path = cfg["prefill_decode_om_path"];
    string decode_om_path = cfg["decode_om_path"];
    resultDir_ = cfg["result_dir"];
    // 如果结果保存路径最后一位是"/", 则去掉
    if (resultDir_.length() > 0 && resultDir_.back() == '/')
        resultDir_.pop_back();

    mode_t old = umask(0);
    CreateDirectoryRecursive(resultDir_);
    umask(old);

    LoadVocab(cfg["tokenizer_json_path"]);
    LoadVocabEmbeddingTable(cfg["token_emb_bin_path"], EMB_DATA_OFFSET, TOTAL_TOKEN_NUM);
    LoadRotaryPostionEmbeddingTable(cfg["rotary_position_emb0_bin_path"], cfg["rotary_position_emb1_bin_path"],
        ROTARY_POSITION_EMB_DATA_OFFSET, MAX_POSITION_IDX);

    // --- 初始化模型 ---
    visionModel_ = make_unique<Model>();
    visionModel_->Load(vision_om_path, MiniCPM);
    auto visionInfo = visionModel_->GetModelInfo();
    for (auto& info : visionInfo.first) {
        visionInputBufs_.emplace_back(info.defaultSize, info.defaultStride);
    }

    resampleModel_ = make_unique<Model>();
    resampleModel_->Load(resample_om_path, MiniCPM);
    auto resampleInfo = resampleModel_->GetModelInfo();
    for (auto& info : resampleInfo.first) {
        resampleInputBufs_.emplace_back(info.defaultSize, info.defaultStride);
    }

    prefillModel_ = make_unique<Model>();
    prefillModel_->Load(prefill_decode_om_path, MiniCPM);
    auto prefillInfo = prefillModel_->GetModelInfo();
    for (auto& info : prefillInfo.first) {
        prefillInputBufs_.emplace_back(info.defaultSize, info.defaultStride);
    }
    for (auto& info : prefillInfo.second) {
        prefillOutputBufs_.emplace_back(info.defaultSize, info.defaultStride);
    }
    prefillModelInputInfo_ = prefillInfo;

    decodeModel_ = make_unique<Model>();
    decodeModel_->Load(decode_om_path, MiniCPM);
    auto decodeInfo = decodeModel_->GetModelInfo();
    for (auto& info : decodeInfo.first) {
        decodeInputBufs_.emplace_back(info.defaultSize, info.defaultStride);
    }
    for (auto& info : decodeInfo.second) {
        decodeOutputBufs_.emplace_back(info.defaultSize, info.defaultStride);
    }
    decodeModelInputInfo_ = decodeInfo;
}

string MiniCpmInfer::InferSingle(const string& imgPath, const string& text)
{
    float startTime = TimestampInner();
    questionNum_++;
    // --- 1. Vision ---
    VisionPreprocess(visionInputBufs_, imgPath);
    auto visionOutputBufs = visionModel_->Infer(visionInputBufs_);
    // --- 2. Resample ---
    ResamplePreprocess(resampleInputBufs_, visionOutputBufs);
    auto resampleOutputBufs = resampleModel_->Infer(resampleInputBufs_);

    // --- 3. Prefill ---
    int prefillLen = 0;
    PrefillPreprocess(prefillInputBufs_, resampleOutputBufs,
        prefillModelInputInfo_.first, prefillLen, text);
    prefillModel_->Infer(prefillInputBufs_, prefillOutputBufs_);
    int prefillTokenId;
    PrefillPostprocess(prefillOutputBufs_, prefillModelInputInfo_.second, prefillLen, prefillTokenId);
    sumTTFT_ += (TimestampInner() - startTime);

    // --- 4. Decode ---
    vector<int32_t> tokenIdVec { prefillTokenId };
    int loopId = prefillLen;
    int tokenId;

    for (int32_t curLoop = prefillLen; curLoop < MAX_TOKEN_LEN; curLoop++) {
        float decodeStartTime = TimestampInner();
        DecodePreprocess(decodeInputBufs_, tokenIdVec,
            make_pair(decodeOutputBufs_, prefillOutputBufs_),
            make_pair(decodeModelInputInfo_.first, prefillModelInputInfo_.second),
            make_pair(loopId, decodeModelInputInfo_.second.size()));
        decodeModel_->Infer(decodeInputBufs_, decodeOutputBufs_);
        DecodePostprocess(decodeOutputBufs_, decodeModelInputInfo_.second, tokenId);
        sumDecodeTokenTime_ += (TimestampInner() - decodeStartTime); // 计算token耗时;
        tokenIdVec.push_back(tokenId);
        decodeTokenNum_++;
        loopId++;
        if (tokenId == MINICPM_END_ID)
            break;
    }

    string result;
    DecodeTokenIds(tokenIdVec, result, vocabData.idToToken_);
    SaveLog(imgPath, text, result, resultDir_);
    return result;
}

int MiniCpmInfer::GetTTFT()
{
    if (questionNum_ == ZERO) {
        LOG(ERROR) << "Question number is 0.";
        return 0;
    }
    return round(sumTTFT_ / questionNum_);
}

float MiniCpmInfer::GetTPS()
{
    if (sumDecodeTokenTime_ == ZERO) {
        LOG(ERROR) << "DecodeToken total time is 0.";
        return 0.0f;
    }
    return decodeTokenNum_ * TIME_CHANGE / sumDecodeTokenTime_;
}

void MiniCpmInfer::UnLoad()
{
    if (visionModel_->Unload() != 0) {
        LOG(ERROR) << "Fail to unload vision model";
    }
    if (resampleModel_->Unload() != 0) {
        LOG(ERROR) << "Fail to unload resample model";
    }
    if (prefillModel_->Unload() != 0) {
        LOG(ERROR) << "Fail to unload prefill model";
    }
    if (decodeModel_->Unload() != 0) {
        LOG(ERROR) << "Fail to unload decode model";
    }
}