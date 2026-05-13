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

#include "ocr_rec_postprocess.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <climits>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include "log.h"
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <map>

using namespace std;
using json = nlohmann::json;

namespace Infer
{
    using namespace std;
    constexpr int BYTE_BIT_NUM = 8;

    static Result GetAndSaveOutputWithBin(TensorDesc &outDesc, TensorBuf &outBuf, std::string outputBinFileName,
        std::vector<float> &noStrideBuf)
    {
        int64_t lastDim = outDesc.dims[outDesc.dimCount - 1];
        size_t dataSize = outDesc.typeSize / BYTE_BIT_NUM; // 一般为4
        size_t lastDimSize = dataSize * lastDim;
        size_t loopNum = outBuf.size / outBuf.stride;
        size_t strideElemNum = outBuf.stride / dataSize;
        float *outData = static_cast<float *>(outBuf.GetRawPtr());
        size_t outSize = outBuf.size / dataSize;
        if (outBuf.stride == lastDimSize || outBuf.stride == 0) {
            std::vector<float> tempBuf(outData, outData + outSize);
            noStrideBuf.assign(tempBuf.begin(), tempBuf.end());
        } else {
            std::vector<float> tempBuf(outData, outData + outSize);
            for (size_t i = 0; i < loopNum; ++i) {
                size_t offset = i * strideElemNum;
                for (size_t index = 0; index < lastDim; index++) {
                    noStrideBuf.push_back(tempBuf[offset + index]);
                }
            }
        }
        if (noStrideBuf.empty()) {
            std::cout << "noStrideBuf malloc fail" << std::endl;
            return FAILED;
        }
        std::ofstream fout(outputBinFileName, std::ios::out | std::ios::binary);
        if (fout.good() == false) {
            std::cout << "create output file [%s] failed" << outputBinFileName.c_str() << std::endl;
            return FAILED;
        }
        fout.write((char *)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
        fout.close();
        return SUCCESS;
    }

    static void SaveResult(TensorDesc &desc, TensorBuf &outBuf, const string &filePath)
    {
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");
        string outputName = filePath.substr(start, end - start);
        string resultPath = "../out/result";
        string binPath = resultPath + "/bin";
        string txtPath = resultPath + "/txt";
        struct stat info; 
        if (stat(resultPath.c_str(), &info) != 0) {
            mkdir(resultPath.c_str(), 0777);
            INFO_LOG("create file success");
        }

        if (stat(txtPath.c_str(), &info) != 0) {
            mkdir(txtPath.c_str(), 0777);
        }

        if (stat(binPath.c_str(), &info) != 0) {
            mkdir(binPath.c_str(), 0777);
        }
        string outputFileName = binPath + outputName + "_" + to_string(0) + ".bin";
        LOG(INFO) << " outputFileName: " << outputFileName;
        std::vector<float> res;
        GetAndSaveOutputWithBin(desc, outBuf, outputFileName, res);
    }

    static Result SaveTxt(string &filePath, string &predition)
    {
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");
        string outputName = filePath.substr(start, end - start);
        string resultPath = "../out/result";
        string txtPath = resultPath + "/txt";
        string outputFileName = txtPath + outputName + "_" + to_string(0) + ".txt";
        LOG(INFO) << " saveTXT:  " << outputFileName;
        struct stat file_stat;
        stat(outputFileName.data(), &file_stat);

        // 打开文件并保留属性
        std::fstream file(outputFileName, std::ios::in | std::ios::out | std::ios::trunc);
        if (!file) {
            LOG(ERROR) << "Error opening file";
            return FAILED;
        }
        file << predition << "\n";
        file.close();
        return SUCCESS;
    }

    static void LoadKeyFile(const std::string &file_path, std::vector<std::string> &characters)
    {
        std::ifstream file(file_path);
        std::string line;
        characters.push_back("blank");
        while (std::getline(file, line)) {
            // 清理行尾
            if (!line.empty() && line.back() == '\n')
                line.pop_back();
            if (!line.empty() && line.back() == '\r')
                line.pop_back();

            if (!line.empty()) {
                characters.push_back(line);
            }
        }

        characters.push_back(" ");
    }

    static std::string DecodeRec( const std::vector<int> &textIndex, const std::vector<float> &textProb,
        std::vector<std::string> &characters, bool ignoreSpace = true)
    {
        int ignoredTokens = 0;
        std::string text;
        for (int i = 0; i < textIndex.size(); i++) {
            int token = textIndex[i];
            // 检查是否是忽略的token
            if (token == ignoredTokens) {
                continue;
            }
            // 2. 跳过重复字符（CTC解码去重）
            if (i > 0 && token == textIndex[i - 1]) {
                continue;
            }
            std::string charStr = characters[token];
            if (ignoreSpace && charStr == " ") {
                continue;
            }
            if (token >= 0 && token < characters.size()) {
                text += characters[token];
            }
        }

        return text;
    }

    static void Predictions(TensorDesc &desc, TensorBuf &outBuf, string inputFile)
    {
        float *outData = static_cast<float *>(outBuf.GetRawPtr());
        size_t wStrideOffset = outBuf.stride / (desc.typeSize / BYTE_BIT_NUM);
        uint32_t outHeight = desc.dims[desc.dimCount - 2];
        uint32_t chnStep = wStrideOffset;
        uint32_t outWidth = desc.dims[desc.dimCount - 1];
        std::vector<std::vector<float>> noStrideBuf;
        float maxProp = 0;
        uint32_t maxIndex = 0;
        std::vector<int> textIndex;
        std::vector<float> textProb;
        for (uint32_t i = 0; i < outHeight; i++){
            chnStep = wStrideOffset * i;
            for (uint32_t j = 0; j < outWidth; j++){
                if (outData[chnStep + j] > maxProp){
                    maxProp = outData[chnStep + j];
                    maxIndex = j;
                }
            }
            
            textIndex.push_back(maxIndex);
            textProb.push_back(maxProp);
            maxIndex = 0;
            maxProp = 0;

            std::vector<float> tempBuf(outData + chnStep, outData + chnStep + outWidth);
            noStrideBuf.push_back(tempBuf);
        }

        SaveResult(desc, outBuf, inputFile);
        std::vector<std::string> characters;
        LoadKeyFile("../data/ppocr_keys_v1.txt", characters);

        std::string result = DecodeRec(textIndex, textProb, characters);
        LOG(INFO) << " result： " << result;
        SaveTxt(inputFile, result);
    }

    bool OcrRecPostprocess(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs,
        std::vector<TensorDesc> &tensorDescs, bool isDpico)
    {
        std::vector<std::vector<cv::Point>> bbox;

        for (size_t i = 0; i < tensorBufs.size(); i++) {
            TensorDesc desc = tensorDescs[i];
            TensorBuf buf = tensorBufs[i];
            if (desc.defaultStride == 0)
            {
                desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
                buf.stride = desc.defaultStride;
            }
            Predictions(desc, buf, fileList[0]);
            LOG(INFO) << "dump final data success " << fileList[0];
        }
        return true;
    }
}