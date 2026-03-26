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

#include "minicpm_postprocess.h"
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace Infer;

static constexpr int32_t LOGITS_OUTPUT_ID = 2;
static constexpr uint32_t ENGLISH_SPECIAL_STR_LEN = 3;

static int FindMaxIndex(float arr[], int32_t n)
{
    if (n <= 0) {
        return -1;
    }

    int32_t maxIndex = 0;
    for (int32_t i = 1; i < n; i++) {
        if (arr[i] > arr[maxIndex]) {
            maxIndex = i;
        }
    }

    return maxIndex;
}

static void AddStr(string& result, const string& str)
{
    if (str.substr(0, ENGLISH_SPECIAL_STR_LEN) == "▁") {
        result += " " + str.substr(ENGLISH_SPECIAL_STR_LEN);
    } else {
        result += str;
    }
}

int PrefillPostprocess(const std::vector<Infer::TensorBuf>& outBuf,
    const std::vector<Infer::TensorDesc>& outDesc,
    int prefillLen,
    int& prefillTokenId)
{
    if (prefillLen < 1) {
        LOG(ERROR) << "[PrefillPostprocess] PrefillLen is less 1!";
        return 0;
    }
    auto outDims = outDesc[LOGITS_OUTPUT_ID];
    auto lastDim = outDims.dimCount - 1;

    float* logits = static_cast<float*>(outBuf[LOGITS_OUTPUT_ID].GetRawPtr())
        + outDims.dims[lastDim] * (prefillLen - 1);

    LOG(INFO) << "[PrefillPostprocess] PrefillLen: " << prefillLen;

    std::vector<float> logitsVec(logits, logits + outDims.dims[lastDim]);
    prefillTokenId = FindMaxIndex(logitsVec.data(), outDims.dims[lastDim]);

    LOG(INFO) << "[PrefillPostprocess] Prefill output: first tokenid is " << prefillTokenId;
    return 0;
}

int DecodePostprocess(const std::vector<Infer::TensorBuf>& outBuf,
    const std::vector<Infer::TensorDesc>& outDesc,
    int& tokenId)
{
    uint32_t logitsOutIdx = outDesc.size() - 1;
    auto* outData = static_cast<float*>(outBuf[logitsOutIdx].GetRawPtr());
    auto outSize = outBuf[logitsOutIdx].size / sizeof(float);

    std::vector<float> logitsDecode(outData, outData + outSize);
    auto maxIndex = FindMaxIndex(logitsDecode.data(), outSize);
    tokenId = maxIndex;
    return 0;
}

int DecodeTokenIds(const std::vector<int>& tokenIdVec,
    std::string& result,
    std::unordered_map<int, std::string>& idToToken)
{
    for (int32_t id : tokenIdVec) {
        if (idToToken.find(id) != idToToken.end()) {
            AddStr(result, idToToken[id]);
        }
    }

    if (result.size() == 0) {
        LOG(ERROR) << "DecodeTokenIds failed, decodeText is empty!";
        return 0;
    }
    return 0;
}
