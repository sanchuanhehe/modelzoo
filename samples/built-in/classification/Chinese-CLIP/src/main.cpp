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

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <sys/stat.h>
#include <chrono>
#include <sstream>
#include <sys/stat.h>
#include "log.h"
#include "dev_interface_adapter.h"
#include <getopt.h>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

#pragma once
#include <iostream>
#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

typedef enum ModelType {
    IMAGE = 0,
    TXT = 1
} ModelType;

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit
static int loop = 1;

Result ReadInputListFile(const std::string &fileName, std::vector<std::string> &inputLists)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file ");
        return FAILED;
    }

    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG(" is not a file, please enter a file");
        return FAILED;
    }
    std::ifstream imglistFile(fileName, std::ios::in);
    if (imglistFile.is_open() == false) {
        ERROR_LOG("open file  failed");
        return FAILED;
    }
    char absPath[PATH_MAX];
    auto ret = realpath(fileName.c_str(), absPath);
    if (ret == nullptr) {
        ERROR_LOG("get realpath failed");
        return FAILED;
    }
    std::string path(dirname(absPath));
    std::string img;
    while (std::getline(imglistFile, img)) {
        if (!imglistFile.eof() && !imglistFile.good()) {
            return FAILED;
        }
        img.erase(std::remove(img.begin(), img.end(), '\n'), img.end());
        img.erase(std::remove(img.begin(), img.end(), '\r'), img.end());
        std::string imgFullPath = path + "/" + img;
        inputLists.push_back(imgFullPath);
    }
    return SUCCESS;
}

Result ReadImgFileToBuf(const std::string &fileName, Infer::TensorDesc desc, Infer::TensorBuf inBuf)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file failed");
        binFile.close();
        return FAILED;
    }
    binFile.seekg(0, binFile.beg);
    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;

    for (int64_t loop = 0; loop < loopTimes; loop++) {
        binFile.read((static_cast<char *>(inBuf.GetRawPtr()) + loop * inBuf.stride), lineSize);
    }

    binFile.close();
    return SUCCESS;
}

Result GetOutputWithBin(Infer::TensorBuf &outBuf, Infer::TensorDesc &outDesc, std::string outputBinFileName, std::vector<float> &noStrideBuf)
{
    INFO_LOG(" GetOutputWithBin ");
    int64_t lastDim = outDesc.dims[outDesc.dimCount - 1];
    size_t dataSize = outDesc.typeSize / BYTE_BIT_NUM;
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = outBuf.size / outBuf.stride;
    size_t strideElemNum = outBuf.stride / dataSize;
    size_t invalidSize = outBuf.stride - lastDimSize;
    float *outData = static_cast<float *>(outBuf.GetRawPtr());
    size_t outSize = outBuf.size / dataSize;
    if (invalidSize == 0) {
        // not stride invalid space, return directly.
        INFO_LOG("not stride invalid space, return directly.");
        // malloc temp buffer to clear stride useless temp buffer to help output.bin compare
        std::vector<float> tempBuf(outData, outData + outSize);
        noStrideBuf.assign(tempBuf.begin(), tempBuf.end());
    } else {
        std::vector<float> tempBuf(outData, outData + outSize);
        for (size_t i = 0; i < loopNum; ++i) {
            size_t offset = i * strideElemNum;
            for (int64_t index = 0; index < lastDim; index++) {
                noStrideBuf.push_back(tempBuf[offset + index]);
            }
        }
    }
    if (noStrideBuf.empty()) {
        INFO_LOG(" noStrideBuf malloc fail");
        return FAILED;
    }
    std::ofstream fout(outputBinFileName, std::ios::out | std::ios::binary);
    if (fout.good() == false) {
        INFO_LOG(" create output file failed ");
        return FAILED;
    }
    fout.write((char *)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
    fout.close();
    return SUCCESS;
}

Result PostProcess(std::vector<Infer::TensorBuf> &outBufs, std::vector<Infer::TensorDesc> &outDescs, const std::string &filePath, ModelType modelType)
{
    // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");

    std::string fileName = filePath.substr(start, end - start);
    std::string resultPath = filePath.substr(0, start) + "/../../out/result";
    struct stat info;
    if (stat(resultPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(resultPath.c_str(), 0777);
        INFO_LOG("create file success");
    }
    std::string jpgPath = resultPath + "/jpg";
    if (stat(jpgPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(jpgPath.c_str(), 0777);
    }
    std::string binPath = resultPath + "/txt";
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), 0777);
    }

    // 保存bin文件
    std::string binFile = binPath + fileName + "_0.bin";
    if (modelType) {
        binFile = binPath + fileName + "_0.bin";
    } else {
        binFile = jpgPath + fileName + "_0.bin";
    }
    std::vector<float> temp;
    GetOutputWithBin(outBufs[0], outDescs[0], binFile, temp);
    return SUCCESS;
}

bool PathToRealPath(const std::string &path, std::string &realPath)
{
    if (path.empty()) {
        ERROR_LOG("path is empty");
        return false;
    }
    if (path.length() > PATH_MAX) {
        ERROR_LOG("illegal path len , the len is");
        return false;
    }
    char tmpPath[PATH_MAX] = {};
    if (realpath(path.c_str(), tmpPath) == nullptr) {
        ERROR_LOG("path to realpath error");
        return false;
    }
    realPath = tmpPath;
    return true;
}

int InferModel(std::string modelPath, ModelType modelType, std::string inputPath)
{
    int ret;
    /* load model */
    std::shared_ptr<Infer::MdlBase> model = Infer::MdlCreate();
    ret = model->LoadModel(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("load model failed");
        Infer::DevDeInit();
        return FAILED;
    }

    /* read img list file, write abs img file path to vector inputLists */
    std::vector<std::string> inputLists;
    std::string realPath;
    if (modelType) {
        realPath = inputPath + "/txt_list.txt";
    } else {
        realPath = inputPath + "/img_list.txt";
    }
    ret = ReadInputListFile(realPath, inputLists);
    if (ret != SUCCESS) {
        ERROR_LOG("ReadInputListFile failed");
        model->UnLoadModel();
        Infer::DevDeInit();
        return FAILED;
    }

    /* set in and out bufs, only support single input*/
    std::vector<Infer::TensorBuf> inBufs, outBufs;
    std::vector<Infer::TensorDesc> inDescs, outDescs;
    Infer::TensorDesc desc;
    size_t inputNum = model->GetInTensorNum();
    size_t outputNum = model->GetOutTensorNum();
    if (inputNum != 1) {
        ERROR_LOG("only support single-input model");
        model->UnLoadModel();
        Infer::DevDeInit();
        return FAILED;
    }

    for (size_t i = 0; i < inputNum; i++) {
        model->GetInTensorDescByIdx(i, desc);
        inDescs.push_back(desc);
        inBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    for (size_t i = 0; i < outputNum; i++) {
        model->GetOutTensorDescByIdx(i, desc);
        outDescs.push_back(desc);
        outBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    model->GetInTensorDescByIdx(0, desc);

    /* model execute */
    std::chrono::microseconds dur(0);
    std::chrono::microseconds dur1(0);
    for (size_t i = 0; i < inputLists.size(); ++i) {
        /* read img to inBuf */
        ret = ReadImgFileToBuf(inputLists[i], desc, inBufs[0]);
        if (ret != SUCCESS) {
            ERROR_LOG("read img file to buf failed");
            model->UnLoadModel();
            Infer::DevDeInit();
            return FAILED;
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < loop; j++) {
            ret = model->Execute(inBufs, outBufs);
            if (ret != SUCCESS) {
                ERROR_LOG("execute inference failed");
                model->UnLoadModel();
                Infer::DevDeInit();
                return FAILED;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        (void)PostProcess(outBufs, outDescs, inputLists[i], modelType);
    }
    INFO_LOG("time: %d, fps: %f", dur.count(), 1000.0 * 1000.0 * (loop * inputLists.size()) / (float)dur.count());
    model->UnLoadModel();
}

int main(int argc, char *argv[])
{
    int opt;
    const char *optstring = "hj:t:a:i:l:";
    struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"imgmodel", required_argument, NULL, 'j'},
        {"txtmodel", required_argument, NULL, 't'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {"loop", required_argument, NULL, 'l'},
        {0, 0, 0, 0}};

    std::string aclConfigPath;
    std::string imgModelPath;
    std::string txtModelPath;
    std::string inputPath;
    while ((opt = getopt_long(argc, argv, optstring, long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'h':
            INFO_LOG("Usage: [--help] [--model OM_MODEL_PATH] [--acl ACL_CONFIG_PATH] [--input IMAGE_DIR] [--loop LOOP_COUNT]");
            return 0;
        case 'j':
            if (optarg) {
                if (!PathToRealPath(optarg, imgModelPath)) {
                    ERROR_LOG("parse model path error");
                    return 0;
                }
            } else {
                // 如果没有提供参数，设置为空字符串或默认值
                imgModelPath = ""; // 或者设置为默认模型路径
                INFO_LOG("No img model specified, using default behavior");
            }
            break;
        case 't':
            if (optarg) {
                if (!PathToRealPath(optarg, txtModelPath)) {
                    ERROR_LOG("parse text model path error");
                    return 0;
                }
            } else {
                // 如果没有提供参数，设置为空字符串或默认值
                txtModelPath = ""; // 或者设置为默认模型路径
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
        case 'l':
        {
            char *endptr = nullptr;
            loop = strtoull(optarg, &endptr, 0);
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

    int ret;

    /* dev init */
    ret = Infer::DevInit(aclConfigPath);
    if (ret != SUCCESS) {
        ERROR_LOG("dev init failed");
        return FAILED;
    }
    if (imgModelPath != "") {
        INFO_LOG(" start image model");
        InferModel(imgModelPath, IMAGE, inputPath);
    }
    if (txtModelPath != "") {
        INFO_LOG(" start txt model");
        InferModel(txtModelPath, TXT, inputPath);
    }
    Infer::DevDeInit();
    return SUCCESS;
}