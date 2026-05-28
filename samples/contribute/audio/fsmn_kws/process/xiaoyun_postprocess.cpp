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

#include "xiaoyun_postprocess.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <vector>

namespace Infer {
namespace XiaoYunPostprocess {

constexpr int VOCAB_SIZE = 2599;
constexpr int HASH_MULTIPLIER = 31;
constexpr int TOP_K_TOKEN_SIZE = 4;
constexpr int SCORE_BEAM_SIZE = 3;
constexpr int PATH_BEAM_SIZE = 20;
constexpr float PROB_THRESHOLD = 0.03f;
constexpr float MIN_PROB_THRESHOLD = 1e-6f;
const std::vector<int> KEYWORD_TOKENS = {1462, 976, 1462, 976};
constexpr const char* KEYWORD_STR = "xiaoyunxiaoyun";

constexpr int FP16_SIGN_SHIFT = 15;
constexpr int FP16_EXP_SHIFT = 10;
constexpr uint32_t FP16_MANTISSA_MASK = 0x3FF;
constexpr uint32_t FP16_EXP_MASK = 0x1F;
constexpr uint32_t FP16_MANTISSA_MSB_BIT = 0x400;

static std::string ReadConfig(const std::string& key)
{
    std::ifstream cfgFile("../data/cfg.txt");
    if (!cfgFile.is_open()) {
        return "";
    }
    
    std::string line;
    while (std::getline(cfgFile, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string cfgKey = line.substr(0, pos);
            std::string cfgValue = line.substr(pos + 1);
            
            size_t start = cfgValue.find_first_not_of(" \t\"");
            size_t end = cfgValue.find_last_not_of(" \t\"");
            if (start != std::string::npos && end != std::string::npos) {
                cfgValue = cfgValue.substr(start, end - start + 1);
            }
            
            if (cfgKey == key) {
                return cfgValue;
            }
        }
    }
    return "";
}

static void CreateDirRecursive(const std::string& path)
{
    size_t pos = 0;
    while ((pos = path.find_first_of('/', pos + 1)) != std::string::npos) {
        std::string subPath = path.substr(0, pos);
        if (!subPath.empty()) {
            mkdir(subPath.c_str(), 0755);
        }
    }
    if (!path.empty()) {
        mkdir(path.c_str(), 0755);
    }
}

static float Fp16ToFp32(uint16_t fp16)
{
    uint32_t sign = (fp16 >> FP16_SIGN_SHIFT) & 1;
    uint32_t exponent = (fp16 >> FP16_EXP_SHIFT) & FP16_EXP_MASK;
    uint32_t mantissa = fp16 & FP16_MANTISSA_MASK;

    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        }
        exponent = 1;
        while (!(mantissa & FP16_MANTISSA_MSB_BIT)) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= FP16_MANTISSA_MASK;
        exponent += 1;
    } else if (exponent == FP16_EXP_MASK) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return sign ? -NAN : NAN;
    }

    uint32_t fp32Sign = sign << 31;
    uint32_t fp32Exponent = (exponent + 112) << 23;
    uint32_t fp32Mantissa = mantissa << 13;

    uint32_t fp32Bits = fp32Sign | fp32Exponent | fp32Mantissa;
    float result;
    std::memcpy(&result, &fp32Bits, sizeof(float));
    return result;
}

struct HypNode {
    int frame = 0;
    float prob = 0.0f;
};

struct Hyp {
    std::vector<int> prefix;
    float pb = 0.0f;   // CTC空白token结尾的概率
    float pnb = 0.0f;  // CTC非空白token结尾的概率
    std::vector<HypNode> nodes;
    float TotalScore() const { return pb + pnb; }
};

class CtcPrefixBeamSearch {
public:
    CtcPrefixBeamSearch(int vocabSize, const std::vector<int>& keywordTokens)
        : vocabSize(vocabSize), keywordTokens(keywordTokens)
    {
        keywordsIdxSet.insert(0);
        for (int id : keywordTokens) {
            keywordsIdxSet.insert(id);
        }
        
        Hyp initHyp;
        initHyp.pb = 1.0f;
        curHyps.push_back(std::move(initHyp));
        
        probs.resize(vocabSize);
        nextHyps.reserve(PATH_BEAM_SIZE * TOP_K_TOKEN_SIZE);
    }

    void ProcessFrame(const float* frameLogits, int frameIdx)
    {
        Softmax(frameLogits, probs.data(), vocabSize);
        
        int topKCount = CollectTopKTokens();
        if (topKCount == 0) {
            return;
        }

        nextHyps.clear();
        prefixToIndex.clear();

        for (int tk = 0; tk < topKCount; tk++) {
            int s = topKTokens[tk];
            float ps = probs[s];
            
            for (const auto& hyp : curHyps) {
                ProcessToken(hyp, s, ps, frameIdx);
            }
        }

        std::sort(nextHyps.begin(), nextHyps.end(), [](const Hyp& a, const Hyp& b) { return a.TotalScore() > b.TotalScore(); });
        if (static_cast<int>(nextHyps.size()) > PATH_BEAM_SIZE) {
            nextHyps.resize(PATH_BEAM_SIZE);
        }

        curHyps.swap(nextHyps);
    }

    const std::vector<Hyp>& GetHyps() const { return curHyps; }

private:
    int CollectTopKTokens()
    {
        int count = 0;
        for (int i = 0; i < vocabSize; i++) {
            if (probs[i] > PROB_THRESHOLD && keywordsIdxSet.count(i) > 0) {
                topKTokens[count] = i;
                count++;
                if (count >= SCORE_BEAM_SIZE) {
                    break;
                }
            }
        }
        return count;
    }

    void ProcessToken(const Hyp& hyp, int token, float prob, int frameIdx)
    {
        int last = hyp.prefix.empty() ? -1 : hyp.prefix.back();
        
        if (token == 0) {
            ProcessBlankToken(hyp, prob);
        } else if (token == last) {
            ProcessRepeatToken(hyp, token, prob, frameIdx);
        } else {
            ProcessNewToken(hyp, token, prob, frameIdx);
        }
    }

    void ProcessBlankToken(const Hyp& hyp, float prob)
    {
        size_t prefixHash = HashPrefix(hyp.prefix);
        float pb = (hyp.pb + hyp.pnb) * prob;
        
        auto it = prefixToIndex.find(prefixHash);
        if (it != prefixToIndex.end()) {
            nextHyps[it->second].pb += pb;
        } else {
            Hyp newHyp;
            newHyp.prefix = hyp.prefix;
            newHyp.nodes = hyp.nodes;
            newHyp.pb = pb;
            newHyp.pnb = 0.0f;
            prefixToIndex[prefixHash] = nextHyps.size();
            nextHyps.push_back(std::move(newHyp));
        }
    }

    void ProcessRepeatToken(const Hyp& hyp, int token, float prob, int frameIdx)
    {
        if (hyp.pnb > MIN_PROB_THRESHOLD) {
            size_t prefixHash = HashPrefix(hyp.prefix);
            auto it = prefixToIndex.find(prefixHash);
            if (it != prefixToIndex.end()) {
                nextHyps[it->second].pnb += hyp.pnb * prob;
            } else {
                Hyp newHyp;
                newHyp.prefix = hyp.prefix;
                newHyp.nodes = hyp.nodes;
                newHyp.pb = 0.0f;
                newHyp.pnb = hyp.pnb * prob;
                prefixToIndex[prefixHash] = nextHyps.size();
                nextHyps.push_back(std::move(newHyp));
            }
            UpdateNodeProb(prefixHash, prob, frameIdx);
        }
        
        if (hyp.pb > MIN_PROB_THRESHOLD) {
            std::vector<int> newPrefix = hyp.prefix;
            newPrefix.push_back(token);
            size_t prefixHash = HashPrefix(newPrefix);
            
            auto it = prefixToIndex.find(prefixHash);
            if (it != prefixToIndex.end()) {
                nextHyps[it->second].pnb += hyp.pb * prob;
            } else {
                Hyp newHyp;
                newHyp.prefix = newPrefix;
                newHyp.nodes = hyp.nodes;
                newHyp.nodes.push_back({frameIdx, prob});
                newHyp.pb = 0.0f;
                newHyp.pnb = hyp.pb * prob;
                prefixToIndex[prefixHash] = nextHyps.size();
                nextHyps.push_back(std::move(newHyp));
            }
        }
    }

    void ProcessNewToken(const Hyp& hyp, int token, float prob, int frameIdx)
    {
        std::vector<int> newPrefix = hyp.prefix;
        newPrefix.push_back(token);
        size_t prefixHash = HashPrefix(newPrefix);
        float pnb = (hyp.pb + hyp.pnb) * prob;
        
        auto it = prefixToIndex.find(prefixHash);
        if (it != prefixToIndex.end()) {
            nextHyps[it->second].pnb += pnb;
        } else {
            Hyp newHyp;
            newHyp.prefix = newPrefix;
            newHyp.nodes = hyp.nodes;
            newHyp.nodes.push_back({frameIdx, prob});
            newHyp.pb = 0.0f;
            newHyp.pnb = pnb;
            prefixToIndex[prefixHash] = nextHyps.size();
            nextHyps.push_back(std::move(newHyp));
        }
        UpdateNodeProb(prefixHash, prob, frameIdx);
    }

    void UpdateNodeProb(size_t prefixHash, float prob, int frameIdx)
    {
        auto it = prefixToIndex.find(prefixHash);
        if (it != prefixToIndex.end() && !nextHyps[it->second].nodes.empty()) {
            if (prob > nextHyps[it->second].nodes.back().prob) {
                nextHyps[it->second].nodes.back().prob = prob;
                nextHyps[it->second].nodes.back().frame = frameIdx;
            }
        }
    }
    size_t HashPrefix(const std::vector<int>& prefix) {
        size_t h = 0;
        for (int v : prefix) {
            h = h * HASH_MULTIPLIER + v;
        }
        return h;
    }

    void Softmax(const float* input, float* output, int size)
    {
        float maxVal = *std::max_element(input, input + size);
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            output[i] = std::exp(input[i] - maxVal);
            sum += output[i];
        }
        for (int i = 0; i < size; i++) {
            output[i] /= sum;
        }
    }

    int vocabSize;
    std::vector<int> keywordTokens;
    std::unordered_set<int> keywordsIdxSet;
    std::vector<Hyp> curHyps;
    std::vector<Hyp> nextHyps;
    std::vector<float> probs;
    std::unordered_map<size_t, size_t> prefixToIndex;
    int topKTokens[TOP_K_TOKEN_SIZE] = {0};
};

static int IsSublist(const std::vector<int>& mainList, const std::vector<int>& checkList)
{
    if (mainList.size() < checkList.size()) {
        return -1;
    }
    if (checkList.empty()) {
        return 0;
    }
    
    if (mainList.size() == checkList.size()) {
        return (mainList == checkList) ? 0 : -1;
    }

    for (size_t i = 0; i <= mainList.size() - checkList.size(); i++) {
        bool match = true;
        for (size_t j = 0; j < checkList.size(); j++) {
            if (mainList[i + j] != checkList[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static bool DetectKeyword(const std::vector<Hyp>& hyps, float& score)
{ 
    for (const auto& hyp : hyps) {
        int offset = IsSublist(hyp.prefix, KEYWORD_TOKENS);
        if (offset != -1) {
            score = 1.0f;
            for (int idx = offset; idx < offset + static_cast<int>(KEYWORD_TOKENS.size()); idx++) {
                if (idx < static_cast<int>(hyp.nodes.size())) {
                    score *= hyp.nodes[idx].prob;
                }
            }
            score = std::sqrt(score);
            return true;
        }
    }
    return false;
}

bool XiaoYunPostprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& outBufs, 
                        std::vector<TensorDesc>& outDescs)
{
    if (outBufs.empty() || outDescs.empty()) {
        LOG(ERROR) << "XiaoYunPostprocess: invalid output";
        return false;
    }

    const TensorDesc& desc = outDescs[0];
    int outFrames = static_cast<int>(desc.dims[1]);
    int outVocab = static_cast<int>(desc.dims[2]);
    size_t stride = outBufs[0].stride;
    if (stride == 0) {
        stride = outVocab * sizeof(uint16_t);
    }
    uint16_t* outputFp16 = static_cast<uint16_t*>(outBufs[0].GetRawPtr());
    if (outputFp16 == nullptr) {
        LOG(ERROR) << "XiaoYunPostprocess: null output pointer";
        return false;
    }
    size_t strideElements = stride / sizeof(uint16_t);
    std::vector<float> frameFp32(outVocab);
    CtcPrefixBeamSearch decoder(outVocab, KEYWORD_TOKENS);
    for (int t = 0; t < outFrames; t++) {
        const uint16_t* framePtr = outputFp16 + t * strideElements;
        for (int v = 0; v < outVocab; v++) {
            frameFp32[v] = Fp16ToFp32(framePtr[v]);
        }
        decoder.ProcessFrame(frameFp32.data(), t);
    }
    
    const auto& hyps = decoder.GetHyps();
    float score = 0.0f;
    bool detected = DetectKeyword(hyps, score);
    LOG(INFO) << fileList[0];
    std::ostringstream logStream;
    logStream << fileList[0] << "\n";
    if (detected) {
        logStream << "XiaoYunPostprocess: DETECTED keyword=" << KEYWORD_STR << " score=" << score << "\n";
        LOG(INFO) << "XiaoYunPostprocess: DETECTED keyword=" << KEYWORD_STR << " score=" << score;
    } else {
        logStream << "XiaoYunPostprocess: REJECTED (no keyword)\n";
        LOG(INFO) << "XiaoYunPostprocess: REJECTED (no keyword)";
    }
    
    std::string save_result_txt = ReadConfig("save_result_txt");
    if (!save_result_txt.empty()) {
        std::string wavPath = fileList[0];
        size_t lastSlash = wavPath.find_last_of('/');
        std::string wavFileName = (lastSlash != std::string::npos) ? wavPath.substr(lastSlash + 1) : wavPath;
        size_t lastDot = wavFileName.find_last_of('.');
        std::string baseName = (lastDot != std::string::npos) ? wavFileName.substr(0, lastDot) : wavFileName;
        std::string resultFilePath = save_result_txt + "/" + baseName + ".txt";
        CreateDirRecursive(save_result_txt);
        
        std::ofstream outFile(resultFilePath, std::ios::app);
        if (outFile.is_open()) {
            outFile << logStream.str();
            outFile.close();
        } else {
            LOG(ERROR) << "Failed to open result file: " << resultFilePath;
        }
    }
    
    return true;
}

}
}