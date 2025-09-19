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

using namespace Infer;
using namespace std;

constexpr size_t BYTE_BIT_NUM = 8; // 1 byte = 8 bit
constexpr size_t TOP_NUM = 5;
struct InferParam {
    string omModelPath;
    string aclConfigPath;
    string imglistPath;
    size_t loop {1};
};

static Result ReadImglistFile(const std::string& fileName, std::vector<std::string>& imglists)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        LOG(ERROR) << "failed to get file, "<< fileName.c_str();
        return FAILED;
    }

    if (S_ISREG(sBuf.st_mode) == 0) {
        LOG(ERROR) << fileName << " is not a file, please enter a file";
        return FAILED;
    }
    std::ifstream imglistFile(fileName, std::ios::in);
    if (imglistFile.is_open() == false) {
        LOG(ERROR) << "open file failed, " << fileName;
        return FAILED;
    }
    char absPath[PATH_MAX];
    auto ret = realpath(fileName.c_str(), absPath);
    if (ret == nullptr) {
        LOG(ERROR) << "get realpath failed";
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

static Result ReadImgFileToBuf(const std::string& fileName, Infer::TensorDesc desc, Infer::TensorBuf inBuf)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        LOG(ERROR) << "open file failed, " << fileName;
        binFile.close();
        return FAILED;
    }
    binFile.seekg(0, binFile.beg);
    if (inBuf.stride == 0) {
        binFile.read(static_cast<char*>(inBuf.GetRawPtr()), desc.defaultSize);
        return SUCCESS;
    }
    size_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    size_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width */
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;

    for (size_t loop = 0; loop < loopTimes; loop++) {
        binFile.read((static_cast<char *>(inBuf.GetRawPtr()) + loop * inBuf.stride), lineSize);
    }

    binFile.close();
    return SUCCESS;
}

static Result GetAndSaveOutputWithBin(Infer::TensorBuf &outBuf, Infer::TensorDesc &outDesc, std::string outputBinFileName, std::vector<float> &noStrideBuf)
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
        std::cout <<"noStrideBuf malloc fail"<< std::endl;
        return FAILED;
    }
    std::ofstream fout(outputBinFileName, std::ios::out|std::ios::binary);
    if (fout.good() == false) {
        std::cout << "create output file [%s] failed"<< outputBinFileName.c_str()<< std::endl;
        return FAILED;
    }
    fout.write((char*)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
    fout.close();
    return SUCCESS;
}

static Result SaveResultWithTxt(const std::string& filePath, std::vector<float>& temp)
{
    std::vector<std::pair<unsigned int, float>> vec;
    std::string line;
    unsigned int topkIndex = 0;
    for(int i = 0; i< temp.size();i++) {
        vec.push_back({i, temp[i]});
    }

    std::sort(vec.begin(), vec.end(), [](const std::pair<unsigned int, float>& a, const std::pair<unsigned int, float>& b ){
        return a.second > b.second;
    });

    // 先获取原文件属性（可选）
    struct stat file_stat;
    stat(filePath.data(), &file_stat);

    // 打开文件并保留属性
    std::fstream file(filePath, std::ios::in | std::ios::out | std::ios::trunc);
    if (!file) {
        LOG(ERROR) << "Error opening file";
        return FAILED;
    }

    for(size_t i = 0; i < vec.size(); i++) {
        const auto& item = vec[i];
        unsigned int id = item.first;
        float value = item.second;
        if (i < TOP_NUM) {
            LOG(INFO) << id << "," << value;
        }
        file << id << "," << value << "\n";
    }
    file.close();
    return SUCCESS;
}

static void PostProcess(std::vector<Infer::TensorBuf> &outBufs, std::vector<Infer::TensorDesc> &outDescs, const std::string &filePath)
{
    std::vector<float> res;
    // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");

    std::string fileName = filePath.substr(start , end-start);
    std::string resultPath = filePath.substr(0, start) + "/../../out/result";
    struct stat info; 
    if (stat(resultPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(resultPath.c_str(), 0777);
        LOG(INFO) << "create file success";
    }
    std::string txtPath = resultPath + "/txt";
    if (stat(txtPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(txtPath.c_str(), 0777);
    }
    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), 0777);
    }
    
    // 保存bin文件
    std::string binFile = binPath + fileName + "_0.bin";
    GetAndSaveOutputWithBin(outBufs[0], outDescs[0], binFile, res);
    std::string txtFile = txtPath + fileName + "_0.txt";
    SaveResultWithTxt(txtFile, res);
}

static bool PathToRealPath(const std::string &path, std::string &realPath)
{
    if (path.empty()) {
        return false;
    }
    if (path.length() > PATH_MAX) {
        return false;
    }
    char tmpPath[PATH_MAX] = {};
    if (realpath(path.c_str(), tmpPath) == nullptr) {
        return false;
    }
    realPath = tmpPath;
    return true;
}

static bool ParseCmd(int argc, char *argv[], InferParam &inferParam)
{
    int opt;
    const char *optstring = "hm:a:i:l:";
    struct option longOptions[] = {
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {"loop", required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, optstring, longOptions, NULL)) != -1) {
        switch (opt) {
            case 'm':
                if (!PathToRealPath(optarg, inferParam.omModelPath)) {
                    LOG(ERROR) << "parse model path error";
                    return false;
                }
                break;
            case 'a':
                if (!PathToRealPath(optarg, inferParam.aclConfigPath)) {
                    LOG(ERROR) << "parse acl config path error";
                    return false;
                }
                break;
            case 'i':
                if (!PathToRealPath(optarg, inferParam.imglistPath)) {
                    LOG(ERROR) << "parse image dir error";
                    return false;
                }
                break;
            case 'l':{
                char *endptr = nullptr;
                inferParam.loop = strtoull(optarg, &endptr, 0);
                if (*endptr != '\0') {
                    LOG(ERROR) << "incorrect input after -l/--loop, " << endptr;
                    return false;
                }
                break;
            }
            case '?':
                LOG(ERROR) << "incorrect config";
                return false;
            default:
                return false;
        }
    }
    return true;
}

static int ModelInfer(InferParam &inferParam)
{
    std::shared_ptr<Infer::MdlBase> model = Infer::MdlCreate();
    int ret = model->LoadModel(inferParam.omModelPath);
    if (ret != 0) {
        LOG(ERROR) << "load model failed, path : " << inferParam.omModelPath;
        return ret;
    }
    /* read img list file, write abs img file path to vector imglists */
    std::vector<std::string> imglists;
    ret = ReadImglistFile(inferParam.imglistPath, imglists);
    if (ret != SUCCESS) {
        LOG(ERROR) << "read img list file failed";
        model->UnLoadModel();
        return ret;
    }
    /* set in and out bufs, only support single input*/
    std::vector<Infer::TensorBuf> inBufs, outBufs;
    std::vector<Infer::TensorDesc> inDescs, outDescs;
    Infer::TensorDesc desc;
    size_t inputNum = model->GetInTensorNum();
    size_t outputNum = model->GetOutTensorNum();
    if (inputNum != 1) {
        LOG(ERROR) << "only support single-input model. model has " << inputNum << " inputs";
        model->UnLoadModel();
        return -1;
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
    std::chrono::microseconds dur(0);
    for (size_t i = 0; i < imglists.size(); ++i) {
        /* read img to inBuf */
        ret = ReadImgFileToBuf(imglists[i], desc, inBufs[0]);
        if (ret != SUCCESS) {
            LOG(ERROR) << "read img file to buf failed";
            model->UnLoadModel();
            return -1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < inferParam.loop; j++) {
            ret = model->Execute(inBufs, outBufs);
            if (ret != SUCCESS) {
                LOG(ERROR) << "execute inference failed";
                model->UnLoadModel();
                return ret;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        PostProcess(outBufs, outDescs, imglists[i]);
        LOG(INFO) << imglists[i] << " infer success";
    }
    LOG(INFO) << "time: " << dur.count() << ", fps: " << 1000.0 * 1000.0 * (inferParam.loop * imglists.size()) / (float)dur.count();
    model->UnLoadModel();
    return 0;
}

int main(int argc, char *argv[])
{
    InferParam inferParam;
    if (!ParseCmd(argc, argv, inferParam)) {
        LOG(ERROR) << "fail to parse cmd";
        return -1;
    }
    if (inferParam.imglistPath.empty() || inferParam.omModelPath.empty()) {
        LOG(ERROR) << "imglistPath: " << inferParam.imglistPath << ", omModelPath:" << inferParam.omModelPath;
        return -1;
    }
    int ret = DevInit(inferParam.aclConfigPath);
    if (ret != 0) {
        LOG(ERROR) << "dev init failed";
        return -1;
    }
    ret = ModelInfer(inferParam);
    if (ret != 0) {
        LOG(ERROR) << "model infer failed";
        return -1;
    }
    Infer::DevDeInit();
    return 0;
}