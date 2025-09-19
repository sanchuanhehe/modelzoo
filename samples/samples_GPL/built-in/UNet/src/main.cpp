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
#include <opencv2/opencv.hpp>
#include <getopt.h>

using namespace Infer;

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit
static int loop = 1;

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

Result ReadImgFileToBuf(const std::string& fileName, Infer::TensorDesc desc, Infer::TensorBuf inBuf)
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

void SaveResultBin(std::vector<Infer::TensorBuf> &outBufs, const std::string& filePath)
{
    // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");

    std::string fileName = filePath.substr(start , end-start);
    std::string resultPath = filePath.substr(0, start) + "/../../out/result";
    struct stat info; 
    if (stat(resultPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(resultPath.c_str(), 0777);
        INFO_LOG("create file success");
    }
    std::string jpgPath = resultPath + "/jpg";
    std::string binPath = resultPath + "/bin";
    if (stat(jpgPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(jpgPath.c_str(), 0777);
        INFO_LOG("create file success");
    }
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), 0777);
    }
    // 保存bin文件
    {
        std::string binFile = binPath + fileName + "_0.bin";
        std::ofstream file(binFile, std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char*>(outBufs[0].GetRawPtr()), outBufs[0].size);
            file.close();
        } else {
            ERROR_LOG("open %s result bin failed\n", filePath.c_str());
        }
    }
};

inline static float Sigmod(float a) {
    return 1.0f/ (1.0f + exp(-a));
}

Result PostProcess(std::vector<Infer::TensorBuf> &outBufs, std::vector<Infer::TensorDesc> &outDescs, const std::string &filePath)
{
    if (outBufs.size() != 1) {
        ERROR_LOG("model output is invalid");
        return FAILED;
    }
    SaveResultBin(outBufs, filePath);

    float *data = static_cast<float *>(outBufs[0].GetRawPtr());
  
    // 创建OpenCV矩阵并应用sigmoid函数
    size_t size = outDescs[0].dims[outDescs[0].dimCount - 1];
    cv::Mat mask(size, size, CV_32FC1, data);
    cv::Mat mask_sigmoid;

    // 创建一个变换矩阵（单位矩阵，不做实际变换）
    cv::Mat transformMat = cv::Mat::eye(1, 1, CV_32F);

    // 使用cv::transform进行矩阵变换
    cv::transform(mask, mask_sigmoid, transformMat);

    // 然后手动应用sigmoid函数（仍然需要）
    mask_sigmoid.forEach<float>([](float &pixel, const int* position) -> void {
        pixel = 1.0f / (1.0f + exp(-pixel));
    });
    // 二值化处理
    cv::Mat mask_binary;
    cv::threshold(mask_sigmoid, mask_binary, 0.5, 1.0, cv::THRESH_BINARY);
    mask_binary.convertTo(mask_binary, CV_8UC1, 255.0); // 转换为8位无符号整数
   
    // 文件名类似：~/img/00001.bin
    // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");

    std::string binPath = filePath.substr(0, start) + "/../../out/result/jpg";
    std::string fileName = filePath.substr(start, end-start);

    // 保存jpg文件
    std::string imgName = binPath + fileName + ".jpg";
    cv::imwrite(imgName, mask_binary);
    return SUCCESS;
}

bool PathToRealPath(const std::string &path, std::string &realPath)
{
    if (path.empty()) {
        ERROR_LOG("path is empty");
        return false;
    }
    if (path.length() > PATH_MAX) {
        ERROR_LOG("illegal path len , the len is %zu", path.length());
        return false;
    }
    char tmpPath[PATH_MAX] = {};
    if (realpath(path.c_str(), tmpPath) == nullptr) {
        ERROR_LOG("path[%s] to realpath error", path.c_str());
        return false;
    }
    realPath = tmpPath;
    return true;
}

int main(int argc, char *argv[])
{
    int opt;
    const char *optstring = "hm:a:i:l:";
    struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {"loop", required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };

    std::string aclConfigPath;
    std::string modelPath;
    std::string imgListPath;
    while ((opt = getopt_long(argc, argv, optstring, long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                INFO_LOG("Usage: %s [--help] [--model OM_MODEL_PATH] [--acl ACL_CONFIG_PATH] [--input IMAGE_DIR] [--loop LOOP_COUNT]", argv[0]);
                return 0;
            case 'm':
                if (!PathToRealPath(optarg, modelPath)) {
                    ERROR_LOG("parse model path error");
                    return 0;
                }
                break;
            case 'a':
                if (!PathToRealPath(optarg, aclConfigPath)) {
                    ERROR_LOG("parse acl config path error");
                    return 0;
                }
                break;
            case 'i':
                if (!PathToRealPath(optarg, imgListPath)) {
                    ERROR_LOG("parse image dir error");
                    return 0;
                }
                break;
            case 'l': {
                char *endptr = nullptr;
                loop = strtoull(optarg, &endptr, 0);
                if (*endptr != '\0') {
                    ERROR_LOG("incorrect input after -l/--loop, %s", endptr);
                    return 0;
                }
                break;
            }
            case '?':
                ERROR_LOG("unknown option: -%c", optopt);
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

    /* load model */
    std::shared_ptr<Infer::MdlBase> model = Infer::MdlCreate();
    ret = model->LoadModel(modelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("load model [%s] failed", modelPath.c_str());
        Infer::DevDeInit();
        return -1;
    }

    /* read img list file, write abs img file path to vector imglists */
    std::vector<std::string> imglists;
    ret = ReadImglistFile(imgListPath, imglists);
    if (ret != SUCCESS) {
        ERROR_LOG("ReadImglistFile failed");
        model->UnLoadModel();
        Infer::DevDeInit();
        return FAILED;
    }

    /* set in and out bufs, only support single input*/
    std::vector<Infer::TensorBuf> inBufs, outBufs;
    std::vector<Infer::TensorDesc> inDescs, outDescs;
    Infer::TensorDesc desc;
    size_t inputNum = model->GetInTensorNum();
    size_t  outputNum = model->GetOutTensorNum();
    if (inputNum != 1) {
        ERROR_LOG("only support single-input model, [%s] has %zu input", modelPath.c_str(), inputNum);
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
    for (size_t i = 0; i < imglists.size(); ++i) {
        /* read img to inBuf */
        ret = ReadImgFileToBuf(imglists[i], desc, inBufs[0]);
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
        INFO_LOG("execute %s success", imglists[i].c_str());

        /* post process: nms*/
        (void)PostProcess(outBufs, outDescs, imglists[i]);
    }
    INFO_LOG("time: %d, fps: %f", dur.count(), 1000.0 * 1000.0 * (loop * imglists.size()) / (float)dur.count());
    model->UnLoadModel();
    Infer::DevDeInit();
    return SUCCESS;
}
