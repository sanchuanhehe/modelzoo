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
#include "nlohmann/json.hpp"
using json = nlohmann::json;

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

Result GetOutputWithBin(Infer::TensorBuf &outBuf, Infer::TensorDesc &outDesc, std::string outputBinFileName, std::vector<float> &noStrideBuf)
{
    INFO_LOG("GetOutputWithBin");
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
        /** */
        std::vector<float> tempBuf(outData, outData + outSize);
        for (size_t i = 0; i < loopNum; ++i) {
            size_t offset = i * strideElemNum;
            for (int64_t index = 0; index < lastDim; index++) {
                // int8_t* outOffset = outData + ;
                noStrideBuf.push_back(tempBuf[offset + index]);
            }
        }
    }
    if (noStrideBuf.empty()) {
        ERROR_LOG("noStrideBuf malloc fail");
        return FAILED;
    }
    std::ofstream fout(outputBinFileName, std::ios::out|std::ios::binary);
    if (fout.good() == false) {
        ERROR_LOG("create output file [%s] failed", outputBinFileName.c_str());
        return FAILED;
    }
    fout.write((char*)&noStrideBuf[0], noStrideBuf.size() * sizeof(float));
    fout.close();
    return SUCCESS;
}

Result GetPad(std::map<std::string, std::tuple<int, int, int, int, int, int, int, int>>& shapes_data) {
// 读取 JSON 文件
    std::ifstream f("../data/shapes.json");
    if (!f.is_open()) {
        ERROR_LOG("open fail: ../data/shapes.json");
        return FAILED;
    }
    
    // 解析 JSON 数据
    json loaded_data;
    try {
        f >> loaded_data;
    } catch (const json::parse_error& e) {
        ERROR_LOG("JSON parse fail: ../data/shapes.json");
        return FAILED;
    }
    f.close();
    
    // 创建映射表，键是去掉扩展名的文件名，值是包含8个整数的元组
    for (auto& item : loaded_data.items()) {
        std::string key = item.key();
        json value = item.value();
        
        // 去掉文件扩展名
        size_t dot_pos = key.find_last_of(".");
        std::string base_name = (dot_pos != std::string::npos) ? key.substr(0, dot_pos) : key;
        
        // 确保值是一个包含8个整数的数组
        if (value.is_array() && value.size() == 8) {
            shapes_data[base_name] = std::make_tuple(
                value[0].get<int>(), value[1].get<int>(), value[2].get<int>(), value[3].get<int>(),
                value[4].get<int>(), value[5].get<int>(), value[6].get<int>(), value[7].get<int>()
            );
        }
    }
    return SUCCESS;
}

void SaveResultBin(std::vector<Infer::TensorBuf> &outBufs, std::vector<Infer::TensorDesc> &outDescs, const std::string& filePath, std::vector<float>& temp)
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
    if (stat(jpgPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(jpgPath.c_str(), 0777);
    }
    std::string binPath = resultPath + "/bin";
    if (stat(binPath.c_str(), &info) != 0) {
        // 文件夹不存在，尝试创建
        mkdir(binPath.c_str(), 0777);
    }
    
    // 保存bin文件
    std::string binFile = binPath + fileName + "_0.bin";
    GetOutputWithBin(outBufs[0], outDescs[0], binFile, temp);
};

inline static float Sigmod(float a) {
    return 1.0f/ (1.0f + exp(-a));
}

cv::Mat VisualizeDepthMap(const cv::Mat& depth_map, 
                           const std::string& output_path = "", 
                           int colormap = cv::COLORMAP_JET) {
    // 检查输入是否为单通道浮点型
    CV_Assert(depth_map.type() == CV_32FC1 || depth_map.type() == CV_64FC1);
    
    // 创建深度图的副本
    cv::Mat processed_depth = depth_map.clone();
    
    // 处理无限大和NaN值
    if (depth_map.type() == CV_32FC1) {
        for (int i = 0; i < processed_depth.rows; ++i) {
            for (int j = 0; j < processed_depth.cols; ++j) {
                float value = processed_depth.at<float>(i, j);
                if (std::isnan(value) || std::isinf(value)) {
                    processed_depth.at<float>(i, j) = 0.0f;
                }
            }
        }
    } else if (depth_map.type() == CV_64FC1) {
        for (int i = 0; i < processed_depth.rows; ++i) {
            for (int j = 0; j < processed_depth.cols; ++j) {
                double value = processed_depth.at<double>(i, j);
                if (std::isnan(value) || std::isinf(value)) {
                    processed_depth.at<double>(i, j) = 0.0;
                }
            }
        }
    }
    
    // 找到非零最小值和最大值
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    if (processed_depth.type() == CV_32FC1) {
        for (int i = 0; i < processed_depth.rows; ++i) {
            for (int j = 0; j < processed_depth.cols; ++j) {
                float value = processed_depth.at<float>(i, j);
                if (value > 0) {
                    min_val = std::min(min_val, static_cast<double>(value));
                    max_val = std::max(max_val, static_cast<double>(value));
                }
            }
        }
    } else if (processed_depth.type() == CV_64FC1) {
        for (int i = 0; i < processed_depth.rows; ++i) {
            for (int j = 0; j < processed_depth.cols; ++j) {
                double value = processed_depth.at<double>(i, j);
                if (value > 0) {
                    min_val = std::min(min_val, value);
                    max_val = std::max(max_val, value);
                }
            }
        }
    }
    
    // 如果没有有效值，设置默认值
    if (min_val > max_val) {
        min_val = 0;
        max_val = 1;
    }
    
    // 归一化深度值到0-255范围
    cv::Mat depth_normalized;
    if (max_val > min_val) {
        processed_depth.convertTo(depth_normalized, CV_32F, 255.0 / (max_val - min_val), 
                                 -min_val * 255.0 / (max_val - min_val));
    } else {
        depth_normalized = cv::Mat::zeros(processed_depth.size(), CV_32F);
    }
    
    // 转换为8位无符号整数
    cv::Mat depth_8bit;
    depth_normalized.convertTo(depth_8bit, CV_8U);
    
    // 应用颜色映射
    cv::Mat depth_colored;
    cv::applyColorMap(depth_8bit, depth_colored, colormap);
    
    // 保存图像（如果指定了输出路径）
    if (!output_path.empty()) {
        cv::imwrite(output_path, depth_colored);
        INFO_LOG("save jpg sucess ");
    }
    
    return depth_colored;
}

Result PostProcess(std::vector<Infer::TensorBuf> &outBufs, std::vector<Infer::TensorDesc> &outDescs, const std::string &filePath)
{
    INFO_LOG("start post");
    std::vector<float> temp;
    SaveResultBin(outBufs, outDescs, filePath, temp);

    std::map<std::string, std::tuple<int, int, int, int, int, int, int, int>> shapes_data;

    GetPad(shapes_data);
     // 从 shapes_data 中获取对应的元组
     // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");
    std::string fileName = filePath.substr(start+1 , end-start-1);
    auto it = shapes_data.find(fileName);
    if (it == shapes_data.end()) {
        ERROR_LOG("key error");
        return FAILED;
    }
    // 解包元组到各个变量
    int h, w, h1, w1, top_pad, bottom_pad, left_pad, right_pad;
    std::tie(h, w, h1, w1, top_pad, bottom_pad, left_pad, right_pad) = it->second;
   
    
    // 创建 inference_result 的示例（实际应用中应该从其他地方获取）
    cv::Mat inference_result(518, 518, CV_32FC1, temp.data()); // 示例：518x518，1通道浮点数
    
    // 计算裁剪区域
    int crop_height = 518 - bottom_pad - top_pad;
    int crop_width = 518 - right_pad - left_pad;
    
    // 定义裁剪区域 (ROI)
    cv::Rect roi(left_pad, top_pad, crop_width, crop_height);
    cv::Mat cropped_array = inference_result(roi);
    // 还原图像大小 - 使用双线性插值
    cv::Mat resized_depth;
    cv::resize(cropped_array, resized_depth, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    // 获取保存文件路径和文件名
    std::string jpgName = filePath.substr(0, start) + "/../../out/result/jpg/" + fileName + ".jpg";
    
    VisualizeDepthMap(resized_depth,  jpgName  );
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
        (void)PostProcess(outBufs, outDescs, imglists[i]);
    }
    INFO_LOG("time: %d, fps: %f", dur.count(), 1000.0 * 1000.0 * (loop * imglists.size()) / (float)dur.count());
    model->UnLoadModel();
    Infer::DevDeInit();
    return SUCCESS;
}