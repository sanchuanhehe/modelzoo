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

#include "yolov8s_world_postprocess.h"

#include <getopt.h>
#include <libgen.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "log.h"

namespace Infer {

constexpr int NUM_CLASSES = 80;
constexpr int BBOX_SIZE = 84;
constexpr int BBOX_NUM = 8400;
constexpr float CONF_THRES = 0.001f;
constexpr float IOU_THRES = 0.7f;
constexpr int YOLO_INPUT_SIZE = 640;
constexpr int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

static int gOriginWidth = 640;
static int gOriginHeight = 640;

// 定义检测框结构体
struct BBox {
    float x1, y1, x2, y2; // 左上和右下坐标
    float score; // 置信度
    int classId; // 类别ID
};

static BBox ScaleBboxToOriginal(const BBox& box)
{
    BBox boxNew;
    float scale = std::min((float)YOLO_INPUT_SIZE / gOriginHeight,
        (float)YOLO_INPUT_SIZE / gOriginWidth);
    float newW = gOriginWidth * scale;
    float newH = gOriginHeight * scale;
    float dx = (YOLO_INPUT_SIZE - newW) / 2;
    float dy = (YOLO_INPUT_SIZE - newH) / 2;

    boxNew.x1 = std::max(0.0f, std::min((box.x1 - dx) / scale, (float)gOriginWidth));
    boxNew.y1 = std::max(0.0f, std::min((box.y1 - dy) / scale, (float)gOriginHeight));
    boxNew.x2 = std::max(0.0f, std::min((box.x2 - dx) / scale, (float)gOriginWidth));
    boxNew.y2 = std::max(0.0f, std::min((box.y2 - dy) / scale, (float)gOriginHeight));
    boxNew.score = box.score;
    boxNew.classId = box.classId;

    return boxNew;
}

static void SaveBboxResult(std::vector<TensorBuf>& outBufs, std::vector<BBox> bboxs,
    const std::string& filePath)
{
    mode_t oldUmask = umask(0);
    // 文件名类似：~/img/00001.bin
    // 获取保存文件路径和文件名
    size_t start = filePath.find_last_of("/");
    size_t end = filePath.find_last_of(".");

    std::string fileName = filePath.substr(start + 1, end - start - 1);
    std::string resultPath = "../out/result";
    std::string txtPath = resultPath + "/txt/";
    for (auto& path : { resultPath, txtPath }) {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            mkdir(path.c_str(), 0777);
        }
    }

    // 保存bbox结果
    std::string txtFile = txtPath + fileName + "_result.txt";
    std::ofstream file1(txtFile, std::ios::out);
    if (file1.is_open()) {
        for (auto& box1 : bboxs) {
            auto box = ScaleBboxToOriginal(box1);
            file1 << "Class " << box.classId << " | Score: " << box.score
                  << " | Box: [" << box.x1 << ", " << box.y1 << ", "
                  << box.x2 << ", " << box.y2 << "]\n";
        }
        file1.close();
    } else {
        LOG(ERROR) << "Save result failed, " << filePath.c_str();
    }

    umask(0);
};


static float CalculateIoU(const BBox& box1, const BBox& box2)
{
    float interX1 = std::max(box1.x1, box2.x1);
    float interY1 = std::max(box1.y1, box2.y1);
    float interX2 = std::min(box1.x2, box2.x2);
    float interY2 = std::min(box1.y2, box2.y2);

    float interW = std::max(0.0f, interX2 - interX1);
    float interH = std::max(0.0f, interY2 - interY1);
    float interArea = interW * interH;

    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float unionArea = area1 + area2 - interArea;

    return interArea / unionArea;
}


// NMS主函数
static std::vector<BBox> NMS(std::vector<BBox>& boxes)
{
    std::vector<BBox> result;
    // 1. 按置信度降序排序
    std::sort(boxes.begin(), boxes.end(),
        [](const BBox& a, const BBox& b) { return a.score > b.score; });
    // 2. 初始化是否保留的标记
    std::vector<bool> keep(boxes.size(), true);
    // 3. 遍历所有框
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!keep[i])
            continue; // 已标记移除则跳过
        // 加入结果集
        result.push_back(boxes[i]);
        // 与后续框计算IoU
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (!keep[j])
                continue;
            // 4. 计算IoU并判断是否移除
            if (CalculateIoU(boxes[i], boxes[j]) > IOU_THRES) {
                keep[j] = false; // 标记为移除
            }
        }
    }
    return result;
}

// 按类别分组NMS
static std::vector<BBox> MulticlassNms(std::vector<BBox>& boxes)
{
    std::vector<BBox> result;

    // 1. 按类别分组
    std::sort(boxes.begin(), boxes.end(),
        [](const BBox& a, const BBox& b) { return a.classId < b.classId; });
    // 2. 对每个类别单独执行NMS
    int currentClass = -1;
    std::vector<BBox> classBoxes;
    for (const auto& box : boxes) {
        if (box.classId != currentClass) {
            // 处理上一个类别
            if (!classBoxes.empty()) {
                auto nmsResult = NMS(classBoxes);
                result.insert(result.end(), nmsResult.begin(), nmsResult.end());
                classBoxes.clear();
            }
            currentClass = box.classId;
        }
        classBoxes.push_back(box);
    }
    // 处理最后一个类别
    if (!classBoxes.empty()) {
        auto nmsResult = NMS(classBoxes);
        result.insert(result.end(), nmsResult.begin(), nmsResult.end());
    }
    return result;
}

static void Transpose(float* data, int rows, int cols)
{
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

static std::vector<BBox> GetBBox(std::vector<TensorBuf>& outBufs)
{
    std::vector<BBox> bboxs;
    float* data = static_cast<float*>(outBufs[0].GetRawPtr());

    for (int i = 0; i < BBOX_NUM; i++) {
        // 提取类别分数
        const float* scores = data + 4; // 类别得分长度偏移
        int classId = std::max_element(scores, scores + NUM_CLASSES) - scores;
        float conf = scores[classId];

        // 获取x y x y
        float x1 = data[0] - data[2] / 2;
        float y1 = data[1] - data[3] / 2;
        float x2 = data[0] + data[2] / 2;
        float y2 = data[1] + data[3] / 2;

        // 置信度过滤
        if (conf >= CONF_THRES) {
            bboxs.push_back({ x1, y1, x2, y2, conf, classId });
        }
        data += BBOX_SIZE;
    }
    return bboxs;
}

static std::vector<BBox> GetNmsBboxs(std::vector<TensorBuf>& outBufs)
{
    // yolo网输出的格式为xywh，代表边框中心点的坐标(x, y)和边框宽高(w, h)
    std::vector<BBox> bboxs = std::move(GetBBox(outBufs));
    // 执行NMS
    std::vector<BBox> result = MulticlassNms(bboxs);
    return result;
}

static Result PostProcess(std::vector<TensorBuf>& outBufs, const std::string& filePath)
{
    float* data = static_cast<float*>(outBufs[0].GetRawPtr());
    // 对矩阵进行转置，从84*8400变为8400*84，方便逐行处理
    Transpose(data, BBOX_SIZE, BBOX_NUM);
    std::vector<BBox> bboxs = std::move(GetNmsBboxs(outBufs));
    // 保存结果文件，保存output到bin，保存框结果到txt中
    SaveBboxResult(outBufs, bboxs, filePath);
    return SUCCESS;
} 

bool GetYoloV8sWorldBox(std::vector<std::string>& fileList,
    std::vector<TensorBuf>& tensorBufs,
    std::vector<TensorDesc>& tensorDescs)
{
    cv::Mat img = cv::imread(fileList[0]);
    cv::Size imgSize = img.size();
    gOriginWidth = imgSize.width;
    gOriginHeight = imgSize.height;
    (void)PostProcess(tensorBufs, fileList[0]);
    LOG(INFO) << "Finished data postprocess successfully, " << fileList[0];
    return true;
}
} // namespace Infer