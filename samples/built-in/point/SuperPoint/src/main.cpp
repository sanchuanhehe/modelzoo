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
#include <unistd.h>
#include <climits>
#include <fstream>
#include <libgen.h>
#include <chrono>
#include <sys/stat.h>
#include <getopt.h>
#include "log.h"
#include "dev_interface_adapter.h"
#include "superpoint_utils.h"

#define SUCCESS 0
#define FAILED -1

#define SIZE_OUTPUT0 (1 * 65 * 30 * 40)
#define SIZE_OUTPUT1 (1 * 256 * 30 * 40)

using namespace Infer;

std::string g_aclConfigPath, g_modelPath, g_imgListPath, g_picturePath;
static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit
static int g_loop = 1;

int ReadImglistFile(const std::string& fileName, std::vector<std::string>& imglists)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        LOG(ERROR) << "failed to get file "<< fileName;
        return FAILED;
    }

    if (S_ISREG(sBuf.st_mode) == 0) {
        LOG(ERROR) << fileName << "is not a file, please enter a file";
        return FAILED;
    }
    std::ifstream imglistFile(fileName, std::ios::in);
    if (imglistFile.is_open() == false) {
        LOG(ERROR) << "open file "<< fileName <<" failed!";
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


int ReadImgFileToBuf(const std::string& fileName, const Infer::TensorDesc &desc, Infer::TensorBuf inBuf)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        LOG(ERROR) << "open file " << fileName << " failed";
        binFile.close();
        return FAILED;
    }
    binFile.seekg(0, binFile.beg);
    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    int64_t width = desc.dims[desc.dimCount - 1]; /* dims last dim is width 40*/
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;

    for (int64_t loop = 0; loop < loopTimes; loop++) {
        binFile.read((static_cast<char *>(inBuf.GetRawPtr()) + loop * inBuf.stride), lineSize);
    }

    binFile.close();
    return SUCCESS;
}


int ConvertImgDataToBuf(const std::vector<float> &imgData, const Infer::TensorDesc &desc, Infer::TensorBuf &inBuf)
{
    long long int loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    long long int width = desc.dims[desc.dimCount - 1]; /* dims last dim is width 40*/
    size_t lineSize = width * desc.typeSize / BYTE_BIT_NUM;
    
    long long int imgDataSize = imgData.size() * sizeof(float);
    if (imgDataSize  != loopTimes * (long long int)lineSize || imgDataSize != (long long int)desc.defaultSize ) {
        LOG(ERROR) <<"imgDataSize(" << imgDataSize<< ") != loopTimes("<< loopTimes << ")*lineSize("<< lineSize << ")";
        return FAILED;
    }
    float *dataPtr = static_cast<float *>(inBuf.GetRawPtr());
    const std::size_t bytesNeeded = imgData.size() * sizeof(float);
    const std::size_t bytesToCopy = std::min(inBuf.size, bytesNeeded);
    const unsigned char* first = reinterpret_cast<const unsigned char*>(imgData.data());
    std::copy(first, first + bytesToCopy, reinterpret_cast<unsigned char*>(dataPtr));
    return SUCCESS;
}

std::string GetCurrentWorkingDir() {
    char buffer[PATH_MAX];
    if (getcwd(buffer, sizeof(buffer)) != NULL) {
        return std::string(buffer);
    } else {
        // 失败时可以通过 perror 打印错误原因
        LOG(ERROR) << "获取当前目录失败";
        return "";
    }
}

int SaveOutBufs(std::vector<Infer::TensorBuf> &outBufs, const std::string& fileName)
{
    std::string resultPath = "../out/result";
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
    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0)
    {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), 0777);
    }
    std::string path = GetCurrentWorkingDir() + "/result/bin/";
    std::string outputName = ExtractFilename(fileName);
    for (size_t i =0 ;i < outBufs.size(); i++){
        // 将拼接字符串,类似 path+ outputName + "_0.bin"
        std::string outputPath = path + outputName + "_"+ std::to_string(i) + ".bin";
        std::ofstream file(outputPath, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("无法打开文件进行写入");
        }
        float *data = static_cast<float *>(outBufs[i].GetRawPtr());    
        // 直接写入原始数据（不进行格式转换）
        file.write(reinterpret_cast<const char*>(data), outBufs[i].size);
        file.close();
        LOG(INFO) << "save outBufs[" << i <<"] success "<< outputPath;
    }
    return SUCCESS;
};


std::vector<KeyPoint> PostProcess(std::vector<Infer::TensorBuf> &outBufs, const std::string& fileName)
{
    size_t sizeOut0 = SIZE_OUTPUT0 * sizeof(float);
    size_t sizeOut1 =  SIZE_OUTPUT1 * sizeof(float);
    if (outBufs.size() != 2 || outBufs[0].size != sizeOut0 ||  outBufs[1].size != sizeOut1 ) {
        throw std::runtime_error("model output is invalid ");
    }

    float *data0 = static_cast<float *>(outBufs[0].GetRawPtr());
    std::vector<float> out0(data0, data0 + outBufs[0].size / sizeof(float));
    float *data1 = static_cast<float *>(outBufs[1].GetRawPtr());
    std::vector<float> out1(data1, data1 + outBufs[1].size / sizeof(float));
    SaveOutBufs(outBufs, fileName);
    std::vector<KeyPoint> keypoints = ProcessKeypoints(out0, out1);

    // 6. 处理解析结果
    std::cout << "解析完成，共检测到 " << keypoints.size() << " 个关键点" << std::endl;
    
    // 示例：输出前5个关键点信息
    for (size_t i = 0; i < std::min(5ul, keypoints.size()); ++i) {
        const auto& kp = keypoints[i];
        std::cout << "关键点 " << i + 1 << ":" << std::endl;
        std::cout << "  位置: (" << kp.x << ", " << kp.y << ")" << std::endl;
        std::cout << "  置信度: " << kp.score << std::endl;
        std::cout << "  特征向量前3个值: " << kp.descriptor[0] << ","<< kp.descriptor[1] <<
            "," << kp.descriptor[2] <<  ",特征向量长度："<< kp.descriptor.size() << std::endl;
    }
    return keypoints;
}

bool PathToRealPath(const std::string &path, std::string &realPath)
{
    if (path.empty()) {
        LOG(ERROR) << "path is empty";
        return false;
    }
    if (path.length() > PATH_MAX) {
        LOG(ERROR) << "illegal path len , the len is "<< path.length();
        return false;
    }
    char tmpPath[PATH_MAX] = {};
    if (realpath(path.c_str(), tmpPath) == nullptr) {
        LOG(ERROR) << "path["<< path <<"] to realpath error";
        return false;
    }
    realPath = tmpPath;
    return true;
}
int ParseArgs (int argc, char *argv[]){
    int opt;
    const char *optstring = "hj:t:a:i:p:l:";
    struct option longOptions[] = {
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {"picture", required_argument, NULL, 'p'},
        {"loop", required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, optstring, longOptions, NULL)) != -1) {
        switch (opt) {
            case 'h':
                std::cout << "Usage: "<< argv[0]<<" [--help] [--model OM_MODEL_PATH] [--acl ACL_CONFIG_PATH] [--input IMAGE_DIR] [--loop LOOP_COUNT]" << std::endl;
                std::cout << "Options:\n";
                std::cout << "  -h, --help            Show this help message\n";
                std::cout << "  -m, --model <path>    Specify the model path\n";
                std::cout << "  -a, --acl <path>      Specify the ACL configuration path\n";
                std::cout << "  -i, --input <file>    Specify the input image list file\n";
                std::cout << "  -l, --loop <count>    Specify the loop count\n";
                std::cout << "  -p, --picture <file>  Specify the picture file\n";
                return FAILED;
            case 'm':
                if (!PathToRealPath(optarg, g_modelPath)) {
                    LOG(ERROR) << "parse model path error";
                    return FAILED;
                }
                break;
            case 'a':
                if (!PathToRealPath(optarg, g_aclConfigPath)) {
                    LOG(ERROR) << "parse acl config path error";
                    return FAILED;
                }
                break;
            case 'i':
                if (!PathToRealPath(optarg, g_imgListPath)) {
                    LOG(ERROR) << "parse image dir error";
                    return FAILED;
                }
                break;
            case 'p':
                if (!PathToRealPath(optarg, g_picturePath)) {
                    LOG(ERROR) << "parse picture file error";
                    return FAILED;
                }
                break;
            case 'l': {
                char *endptr = nullptr;
                g_loop = strtoull(optarg, &endptr, 0);
                if (*endptr != '\0') {
                    LOG(ERROR) << "incorrect input after -l/--loop" << endptr;
                    return FAILED;
                }
                break;
            }
            case '?':
                LOG(ERROR) << "incorrect config";
                return FAILED;
            default:
                LOG(ERROR) << "unexpected error";
                return FAILED;
        }
    }
    return SUCCESS;
}

int GetModelInfo(std::shared_ptr<Infer::MdlBase> &model, std::vector<Infer::TensorBuf> &inBufs, std::vector<Infer::TensorBuf> &outBufs,
    Infer::TensorDesc &desc) {
    
    size_t inputNum = model->GetInTensorNum();
    size_t outputNum = model->GetOutTensorNum();
    if (inputNum != 1) {
        LOG(ERROR) << "only support single-input model, current model has "<< inputNum <<" input";
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
    return SUCCESS;
}

int ModelInferImgList(std::shared_ptr<Infer::MdlBase> &model)
{
    /* read img list file, write abs img file path to vector imglists */
    std::vector<std::string> imglists;
    int ret;
    ret = ReadImglistFile(g_imgListPath, imglists);
    if (ret != SUCCESS) {
        LOG(ERROR) << "ReadImglistFile failed";
        return FAILED;
    }
    /* set in and out bufs, only support single input*/
    std::vector<Infer::TensorBuf> inBufs, outBufs;
    Infer::TensorDesc desc;
    GetModelInfo(model, inBufs, outBufs, desc);

    /* model execute */
    std::chrono::microseconds dur(0);
    for (size_t i = 0; i < imglists.size(); ++i) {
        /* read img to inBuf */
        ret = ReadImgFileToBuf(imglists[i], desc, inBufs[0]);
        if (ret != SUCCESS) {
            LOG(ERROR) << "read img file to buf failed";
            return FAILED;
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < g_loop; j++) {
            ret = model->Execute(inBufs, outBufs);   // 执行
            if (ret != SUCCESS) {
                LOG(ERROR) << "execute inference failed";
                return FAILED;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        LOG(INFO) << "execute "<< imglists[i] <<" success";
        /* post process: nms*/
        (void)PostProcess(outBufs, imglists[i]);
    }
    LOG(INFO) << "time:"<< dur.count() << ", fps: "
        << 1000.0 * 1000.0 * (g_loop * imglists.size()) / (float)dur.count();
    return SUCCESS;
}

int ModelInferPicture(std::shared_ptr<Infer::MdlBase> &model)
{
    std::vector<float> preprocessedData = SuperpointPreprocess(g_picturePath);
    /* set in and out bufs, only support single input*/
    std::vector<Infer::TensorBuf> inBufs, outBufs;
    Infer::TensorDesc desc;
    GetModelInfo(model, inBufs, outBufs, desc);

    /* model execute */
    std::chrono::microseconds dur(0);
    /* read img to inBuf */
    int ret;
    if(inBufs[0].size !=  preprocessedData.size() * sizeof(float)) {
        LOG(ERROR) << "inBufs[0].size("<<inBufs[0].size <<") != preprocessedData.size("<<
            preprocessedData.size() <<") *sizeof(float)";
        return FAILED;
    }
    ConvertImgDataToBuf(preprocessedData, desc, inBufs[0]);

    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < g_loop; j++) {
        ret = model->Execute(inBufs, outBufs);   // 执行
        if (ret != SUCCESS) {
            LOG(ERROR) << "execute inference failed";
            return FAILED;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG(INFO) << "execute "<< g_picturePath <<" success";
    LOG(INFO) << "time: "<< dur.count() <<", fps: "<< 1000.0 * 1000.0 * g_loop / (float)dur.count();
    
    /* post process, save and extract feature point */
    auto keypoints = PostProcess(outBufs, g_picturePath);
    std::string outputPath = ExtractFilename(g_picturePath) +"_draw_keypoints.jpg";
    VisualizeKeypoints(keypoints, g_picturePath, outputPath);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    int32_t ret;
    ret = ParseArgs(argc, argv);
    if(ret!= SUCCESS){
        return FAILED;
    }

    /* dev init */
    ret = Infer::DevInit(g_aclConfigPath);
    if (ret != SUCCESS) {
        LOG(ERROR) <<"dev init failed";
        return FAILED;
    }

    /* load model */
    std::shared_ptr<Infer::MdlBase> model = Infer::MdlCreate();
    ret = model->LoadModel(g_modelPath);
    if (ret != 0) {
        LOG(ERROR) <<"load model ["<< g_modelPath << "] failed";
        Infer::DevDeInit();
        return FAILED;
    }

    /* excute model */
    if (g_imgListPath != "") { // 不存在 g_imgListPath 跳过
        ret = ModelInferImgList(model);
        if(ret !=SUCCESS){
            LOG(ERROR) << "ModelInferImgList["<< g_modelPath <<"] failed";
        }
    }
    if (g_picturePath != "") {  // 不存在 g_picturePath 跳过
        ret = ModelInferPicture(model);
        if(ret !=SUCCESS){
            LOG(ERROR) << "ModelInferPicture[" << g_picturePath <<"] failed";
        }
    }
    model->UnLoadModel();
    Infer::DevDeInit();
    return SUCCESS;
}