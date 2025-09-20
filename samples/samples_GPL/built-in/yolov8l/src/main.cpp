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
constexpr int bbox_size = 84;
constexpr int bbox_num = 8400;
constexpr float conf_thres = 0.25f;
constexpr float iou_thres = 0.45f;
constexpr int yolov8_input_size = 640;
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

Result ReadImgFileToBuf(const std::string& fileName, Infer::TensorDesc desc,
    Infer::TensorBuf inBuf)
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

void SaveBboxResult(std::vector<Infer::TensorBuf> &outBufs, std::vector<BBox> bboxs,
    const std::string& filePath)
{
    // 文件名类似：~/img/00001.bin
    // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find(".");

    std::string binPath = filePath.substr(0, start) + "/../result/bin/";
    std::string txtPath = filePath.substr(0, start) + "/../result/txt/";
    std::string fileName = filePath.substr(start + 1, end-start-1);

    // 保存bin文件
    {
        std::string binFile = binPath + fileName + "_result.bin";
        std::ofstream file(binFile, std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char*>(outBufs[0].GetRawPtr()), outBufs[0].size);
            file.close();
        } else {
            ERROR_LOG("open %s result bin failed\n", filePath.c_str());
        }
    }

    // 保存bbox结果
    {
        std::string txtFile = txtPath + fileName + "_result.txt";
        std::ofstream file(txtFile, std::ios::out);
        if (file.is_open()) {
            for (auto &box : bboxs) {
                file << "Class " << box.class_id << " | Score: " << box.score 
                    << " | Box: [" << box.x1 << ", " << box.y1 << ", "
                    << box.x2 << ", " << box.y2 << "]\n";
            }
            file.close();
        } else {
            ERROR_LOG("open %s result txt failed\n", filePath.c_str());
        }
    }
};

// 计算两个框的IoU（交并比）
float calculateIoU(const BBox& box1, const BBox& box2) {
    // 计算交集区域
    float inter_x1 = std::max(box1.x1, box2.x1);
    float inter_y1 = std::max(box1.y1, box2.y1);
    float inter_x2 = std::min(box1.x2, box2.x2);
    float inter_y2 = std::min(box1.y2, box2.y2);

    // 计算交集面积
    float inter_area = std::max(0.0f, inter_x2 - inter_x1 + 1) * 
                      std::max(0.0f, inter_y2 - inter_y1 + 1);

    // 计算各自面积
    float area1 = (box1.x2 - box1.x1 + 1) * (box1.y2 - box1.y1 + 1);
    float area2 = (box2.x2 - box2.x1 + 1) * (box2.y2 - box2.y1 + 1);

    // 计算并集面积
    float union_area = area1 + area2 - inter_area;

    return inter_area / union_area;
}

// NMS主函数
std::vector<BBox> nms(std::vector<BBox>& boxes) {
    std::vector<BBox> result;

    // 1. 按置信度降序排序
    std::sort(boxes.begin(), boxes.end(), 
              [](const BBox& a, const BBox& b) { return a.score > b.score; });

    // 2. 初始化是否保留的标记
    std::vector<bool> keep(boxes.size(), true);

    // 3. 遍历所有框
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!keep[i]) continue; // 已标记移除则跳过

        // 加入结果集
        result.push_back(boxes[i]);

        // 与后续框计算IoU
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!keep[j]) continue;

            // 4. 计算IoU并判断是否移除
            if (calculateIoU(boxes[i], boxes[j]) > iou_thres) {
                keep[j] = false; // 标记为移除
            }
        }
    }

    return result;
}

// 按类别分组NMS
std::vector<BBox> multiclass_nms(std::vector<BBox>& boxes)
{
    std::vector<BBox> result;
    
    // 1. 按类别分组
    std::sort(boxes.begin(), boxes.end(), 
              [](const BBox& a, const BBox& b) { return a.class_id < b.class_id; });
    
    // 2. 对每个类别单独执行NMS
    int current_class = -1;
    std::vector<BBox> class_boxes;
    
    for (const auto& box : boxes) {
        if (box.class_id != current_class) {
            // 处理上一个类别
            if (!class_boxes.empty()) {
                auto nms_result = nms(class_boxes);
                result.insert(result.end(), nms_result.begin(), nms_result.end());
                class_boxes.clear();
            }
            current_class = box.class_id;
        }
        class_boxes.push_back(box);
    }
    
    // 处理最后一个类别
    if (!class_boxes.empty()) {
        auto nms_result = nms(class_boxes);
        result.insert(result.end(), nms_result.begin(), nms_result.end());
    }
    
    return result;
}

void transpose(float* data, int rows, int cols) {
    float* temp = new float[rows * cols];
    
    // 执行转置操作
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            temp[j * rows + i] = data[i * cols + j];
        }
    }
    memcpy(data, temp, rows * cols * sizeof(float));

    delete[] temp;
}

std::vector<BBox> GetBBox(std::vector<Infer::TensorBuf> &outBufs)
{
    std::vector<BBox> bboxs;
    float *data = static_cast<float *>(outBufs[0].GetRawPtr());

    for (int i = 0; i < bbox_num; i++) {
        // 提取类别分数
        const float* scores = data + 4;
        int class_id = std::max_element(scores, scores + num_classes) - scores;
        float conf = scores[class_id];

        // 获取x y x y
        float x1 = data[0] - data[2] / 2;
        float y1 = data[1] - data[3] / 2;
        float x2 = data[0] + data[2] / 2;
        float y2 = data[1] + data[3] / 2;

        // 置信度过滤
        if (conf >= conf_thres) {
            bboxs.push_back({x1, y1, x2, y2, conf, class_id});
        }
        data += bbox_size;
    }
    return bboxs;
}

std::vector<BBox> GetNmsBboxs(std::vector<Infer::TensorBuf> &outBufs)
{
    // yolo网输出的格式为xywh，代表边框中心点的坐标(x, y)和边框宽高(w, h)
    std::vector<BBox> bboxs = std::move(GetBBox(outBufs));

    // 执行NMS
    std::vector<BBox> result =  multiclass_nms(bboxs);

    return result;
}

Result PostProcess(std::vector<Infer::TensorBuf> &outBufs, const std::string &filePath)
{
    if (outBufs.size() != 1 || outBufs[0].size != bbox_size * bbox_num * sizeof(float)) {
        ERROR_LOG("model output is invalid");
        return FAILED;
    }

    // 对矩阵进行转置，从84*8400变为8400*84，方便逐行处理
    float *data = static_cast<float *>(outBufs[0].GetRawPtr());
    transpose(data, bbox_size, bbox_num); 

    // 得到经过nms后的bbox框,选取conf_thres=0.25 iou_thres=0.45
    std::vector<BBox> bboxs = std::move(GetNmsBboxs(outBufs));

    // 保存结果文件，保存output到bin，保存框结果到txt中
    SaveBboxResult(outBufs, bboxs, filePath);
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
        (void)PostProcess(outBufs, imglists[i]);
    }
    INFO_LOG("time: %d, fps: %f", dur.count(), 1000.0 * 1000.0 * (loop * imglists.size()) / (float)dur.count());
    model->UnLoadModel();
    Infer::DevDeInit();
    return SUCCESS;
}