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

using namespace Infer;

constexpr int num_classes = 80;
constexpr int bbox_size = 116;
constexpr int bbox_num = 8400;
constexpr int protos_num = 32;
constexpr int protos_size = 160;
static int loop = 1;
static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

// 定义检测框结构体
struct BBox {
    float x1, y1, x2, y2; // 左上和右下坐标
    float score;          // 置信度
    int class_id;         // 类别ID
};

Result ReadImglistFile(const std::string& fileName, std::vector<std::string>& imglists)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", fileName.c_str());
        return FAILED;
    }

    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return FAILED;
    }
    std::ifstream imglistFile(fileName, std::ios::in);
    if (imglistFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
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
        imglists.push_back(imgFullPath);
    }
    return SUCCESS;
}

Result ReadImgFileToBuf(const std::string& fileName, TensorDesc desc,
    TensorBuf inBuf)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
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

Result PostProcess(std::vector<TensorBuf> &outBufs, const std::string &filePath)
{
    if (outBufs.size() != 2 || outBufs[1].size != bbox_size * bbox_num * sizeof(float) ||
            outBufs[0].size != protos_num * protos_size * protos_size * sizeof(float)) {
        ERROR_LOG("model output is invalid, outBufs[0].size:%u, outBufs[1].size:%u", outBufs[0].size, outBufs[1].size);
        return FAILED;
    }
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find(".");

    std::string binPath = filePath.substr(0, start) + "/../result/bin/";
    std::string txtPath = filePath.substr(0, start) + "/../result/txt/";
    std::string fileName = filePath.substr(start + 1, end-start-1);
    // 保存bin文件
    for (size_t i = 0; i < outBufs.size(); i++) {
        std::stringstream ss;
        ss << binPath << fileName << "_result" << i << ".bin";
        std::string binFilePath;
        ss >> binFilePath;
        std::ofstream file(binFilePath, std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char*>(outBufs[i].GetRawPtr()), outBufs[i].size);
            file.close();
        } else {
            ERROR_LOG("open %s result bin failed\n", filePath.c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    if ((argv[1] == nullptr) || (argv[2] == nullptr) || (argv[3] == nullptr)) {
        ERROR_LOG("Please input: ./main <acl_config_path> <model_path> <image_list_path> <loop_count>");
        return FAILED;
    }
    int ret;
    const std::string aclConfigPath = std::string(argv[1]);
    const std::string modelPath = std::string(argv[2]);
    const std::string imgListPath = std::string(argv[3]);
    if (argv[4] != nullptr) {
        loop = std::stoi(argv[4]);
    }

    /* dev init */
    ret = DevInit(aclConfigPath);
    if (ret != SUCCESS) {
        ERROR_LOG("dev init failed");
        return FAILED;
    }

    /* load model */
    std::shared_ptr<MdlBase> model = MdlCreate();
    ret = model->LoadModel(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("load model [%s] failed", modelPath.c_str());
        DevDeInit();
        return -1;
    }

    /* read img list file, write abs img file path to vector imglists */
    std::vector<std::string> imglists;
    ret = ReadImglistFile(imgListPath, imglists);
    if (ret != SUCCESS) {
        ERROR_LOG("ReadImglistFile failed");
        model->UnLoadModel();
        DevDeInit();
        return FAILED;
    }

    /* set in and out bufs, only support single input*/
    std::vector<TensorBuf> inBufs, outBufs;
    TensorDesc desc;
    size_t inputNum = model->GetInTensorNum();
    size_t  outputNum = model->GetOutTensorNum();
    if (inputNum != 1) {
        ERROR_LOG("only support single-input model, [%s] has %zu input", modelPath.c_str(), inputNum);
        model->UnLoadModel();
        DevDeInit();
        return FAILED;
    }

    for (size_t i = 0; i < inputNum; i++) {
        model->GetInTensorDescByIdx(i, desc);
        inBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    for (size_t i = 0; i < outputNum; i++) {
        model->GetOutTensorDescByIdx(i, desc);
        outBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }

    model->GetInTensorDescByIdx(0, desc);


    /* model execute */
    std::chrono::microseconds dur(0);
    for (size_t i = 0; i < imglists.size(); ++i) {
        /* read img to inBuf */
        ret = ReadImgFileToBuf(imglists[i], desc, inBufs[0]);
        if (ret != SUCCESS) {
            ERROR_LOG("read img file to buf failed");
            model->UnLoadModel();
            DevDeInit();
            return FAILED;
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < loop; j++) {
            ret = model->Execute(inBufs, outBufs);
            if (ret != SUCCESS) {
                ERROR_LOG("execute inference failed");
                model->UnLoadModel();
                DevDeInit();
                return FAILED;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        INFO_LOG("execute %s success", imglists[i].c_str());

        /* post process: save bin file*/
        (void)PostProcess(outBufs, imglists[i]);
    }
    INFO_LOG("time: %d, fps: %f", dur.count(), 1000.0 * 1000.0 * (loop * imglists.size()) / (float)dur.count());
    model->UnLoadModel();
    DevDeInit();
    return SUCCESS;
}