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

#include "yolo6s_postprocess.h"
#include "log.h"
#include "utils.h"
#include <map>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>

namespace Infer {
namespace Yolo6s {
using namespace std;

// 常量定义
const float MIN_VALUE = 0.0f;
const int YOLO80_CLASS_COUNT = 80;
const int COCO90_CLASS_COUNT = 90;
const int NUM_BOXES = 8400;
const int BOX_DIM_85 = 85;
const int BOX_DIM_88 = 88;
const int BOX_CX_IDX = 0;
const int BOX_CY_IDX = 1;
const int BOX_W_IDX = 2;
const int BOX_H_IDX = 3;
const int BOX_OBJ_CONF_IDX = 4;
const int BOX_CLS_START_IDX = 5;
const float HALF_VALUE = 2.0f;
constexpr int BYTE_BIT_NUM = 8;
constexpr int FP16_BIT_NUM = 16;
constexpr int FP32_BIT_NUM = 32;

struct BBox {
    float x1, y1, x2, y2, score;
    int classId;
    int cocoClassId;
};

// YOLO80到COCO90的官方ID映射表
const vector<int> yolo80ToCoco90 = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90
};

// 提取文件名（不含路径和扩展名）
string GetFileName(const string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == string::npos || end <= start) ? path.substr(start) : path.substr(start, end - start);
}

// 按类别分组执行NMS（与PyTorch batched_nms完全对齐）
vector<BBox> BatchedNMS(vector<BBox>& boxes, float iouThres) {
    // 按类别分组
    map<int, vector<BBox>> classBoxes;
    for (const auto& box : boxes) {
        classBoxes[box.classId].push_back(box);
    }

    vector<BBox> nmsResult;
    for (map<int, vector<BBox>>::iterator it = classBoxes.begin(); it != classBoxes.end(); ++it) {
        vector<BBox>& clsBoxes = it->second;  // 只获取需要的value，不定义key变量

        // 按置信度降序排序
        sort(clsBoxes.begin(), clsBoxes.end(), [](const BBox& a, const BBox& b) {
            return a.score > b.score;
        });

        vector<bool> keep(clsBoxes.size(), true);
        for (size_t i = 0; i < clsBoxes.size(); ++i) {
            if (!keep[i]) continue;
            nmsResult.push_back(clsBoxes[i]);

            // 计算当前框与后续框的IOU
            const BBox& boxI = clsBoxes[i];
            for (size_t j = i + 1; j < clsBoxes.size(); ++j) {
                if (!keep[j]) continue;
                const BBox& boxJ = clsBoxes[j];

                float interLeft = max(boxI.x1, boxJ.x1);
                float interTop = max(boxI.y1, boxJ.y1);
                float interRight = min(boxI.x2, boxJ.x2);
                float interBottom = min(boxI.y2, boxJ.y2);
                float interArea = max(MIN_VALUE, interRight - interLeft) * max(MIN_VALUE, interBottom - interTop);

                float areaI = (boxI.x2 - boxI.x1) * (boxI.y2 - boxI.y1);
                float areaJ = (boxJ.x2 - boxJ.x1) * (boxJ.y2 - boxJ.y1);
                float unionArea = areaI + areaJ - interArea;
                float iou = unionArea > MIN_VALUE ? interArea / unionArea : MIN_VALUE;

                if (iou > iouThres) {
                    keep[j] = false;
                }
            }
        }
    }
    return nmsResult;
}

// 计算缩放比例和填充量
void CalculateScaleAndPad(int originalHeight, int originalWidth, int targetWidth, int targetHeight,
                          float& scale, int& padWidth, int& padHeight) {
    scale = min(static_cast<float>(targetWidth) / originalWidth,
                static_cast<float>(targetHeight) / originalHeight);

    float newWidth = originalWidth * scale;
    float newHeight = originalHeight * scale;

    int deltaWidth = targetWidth - static_cast<int>(newWidth);
    int deltaHeight = targetHeight - static_cast<int>(newHeight);
    padWidth = deltaWidth / 2;
    padHeight = deltaHeight / 2;
}

// 缩放边界框到原始图像尺寸
void ScaleBoxes(vector<BBox>& boxes, int originalWidth, int originalHeight,
                float scale, int padWidth, int padHeight) {
    for (auto& box : boxes) {
        box.x1 = (box.x1 - padWidth) / scale;
        box.y1 = (box.y1 - padHeight) / scale;
        box.x2 = (box.x2 - padWidth) / scale;
        box.y2 = (box.y2 - padHeight) / scale;

        box.x1 = max(MIN_VALUE, min(box.x1, static_cast<float>(originalWidth)));
        box.y1 = max(MIN_VALUE, min(box.y1, static_cast<float>(originalHeight)));
        box.x2 = max(MIN_VALUE, min(box.x2, static_cast<float>(originalWidth)));
        box.y2 = max(MIN_VALUE, min(box.y2, static_cast<float>(originalHeight)));
    }
}

// 按COCO类别和置信度排序
void SortResults(vector<BBox>& results) {
    sort(results.begin(), results.end(), [](const BBox& a, const BBox& b) {
        if (a.cocoClassId != b.cocoClassId) {
            return a.cocoClassId < b.cocoClassId;
        } else {
            if (a.score == b.score) {
                return a.x1 < b.x1;
            }
            return a.score > b.score;
        }
    });
}

// 保存TXT结果
void SaveTxtResults(const vector<BBox>& sortedResult, const string& saveTxt, const string& fileName) {
    string txtPath = saveTxt + "/" + fileName + ".txt";
    ofstream txtFile(txtPath);
    if (!txtFile.is_open()) { // 增加文件打开失败的判断
        LOG(ERROR) << "无法打开TXT文件: " << txtPath;
        return;
    }
    txtFile.precision(6);
    txtFile << fixed;
    for (const auto& box : sortedResult) {
        if (box.cocoClassId == -1) continue;
        float width = box.x2 - box.x1;
        float height = box.y2 - box.y1;
        txtFile << box.cocoClassId << " "
                << box.score << " "
                << box.x1 << " "
                << box.y1 << " "
                << width << " "
                << height << "\n";
    }
    // 显式关闭文件，释放文件句柄
    txtFile.close();
}

const float* GetOutDataAsFP32(const TensorBuf& buf, const TensorDesc& desc, std::vector<float>& preds) {
    // 清空输出容器（确保数据干净）
    preds.clear();
    
    // 获取Tensor基础信息
    size_t typeSize = desc.typeSize;
    size_t floatCount = buf.size / (typeSize / BYTE_BIT_NUM);

    // 根据数据类型处理
    if (typeSize == FP16_BIT_NUM) {
        // FP16转FP32
        const uint16_t* fp16Preds = reinterpret_cast<const uint16_t*>(buf.data.get());
        preds.resize(floatCount);
        for (size_t i = 0; i < floatCount; ++i) {
            preds[i] = HalfToFloat(fp16Preds[i]);
        }
        return preds.data();
    } else if (typeSize == FP32_BIT_NUM) {
        // 直接返回FP32数据
        return reinterpret_cast<const float*>(buf.data.get());
    } else {
        LOG(ERROR) << "output unsupported desc.typeSize: " << typeSize;
        return nullptr;
    }
}

// 核心后处理函数
bool Yolov6sPostprocess(vector<string>& fileList,
                       vector<TensorBuf>& outBufs,
                       vector<TensorDesc>& outDescs) {
    LOG(INFO) << "====== 开始后处理 ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    float confThres = stof(cfg["conf_threshold"]);
    float nmsThres = stof(cfg["nms_threshold"]);
    string saveBin = cfg["save_result_bin"];
    string saveTxt = cfg["save_result_txt"];
    int targetWidth = stoi(cfg["img_width"]);
    int targetHeight = stoi(cfg["img_height"]);

    if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
        LOG(ERROR) << "创建bin目录失败: " << saveBin;
        return false;
    }
    if (!saveTxt.empty() && !CreateDirectoryRecursive(saveTxt)) {
        LOG(ERROR) << "创建txt目录失败: " << saveTxt;
        return false;
    }

    if (fileList.size() != outBufs.size()) {
        LOG(ERROR) << "文件列表与输出缓冲区数量不匹配";
        return false;
    }

    size_t imgIdx = 0;
    const string& imgPath = fileList[imgIdx];
    string fileName = GetFileName(imgPath);
    if (!saveBin.empty()) {
        string binPath = saveBin + "/" + fileName + "_output_0.bin";
        ofstream binFile(binPath, ios::binary);
        if (binFile.is_open()) {
            binFile.write(reinterpret_cast<char*>(outBufs[imgIdx].data.get()), outBufs[imgIdx].size);
            binFile.close(); // 显式关闭，释放文件句柄
        } else {
            LOG(WARNING) << "无法打开bin文件: " << binPath;
        }
    }
    // 解析模型输出
    std::vector<float> detectionsPreds;  
    const float* data = GetOutDataAsFP32(outBufs[imgIdx], outDescs[imgIdx], detectionsPreds);
    if (data == nullptr) {
        LOG(WARNING) << "模型输出缓冲区为空: " << fileName;
        return false;
    }
    int64_t bufSize = outBufs[imgIdx].size;
    const int floatSize = sizeof(float);
    int totalElements = bufSize / floatSize;
    int actualBoxDim = totalElements / NUM_BOXES;
    if (actualBoxDim != BOX_DIM_85 && actualBoxDim != BOX_DIM_88) {
        LOG(ERROR) << "步长异常: " << actualBoxDim;
        return false;
    }
    vector<BBox> boxes;
    for (int j = 0; j < NUM_BOXES; ++j) {
        int baseIdx = j * actualBoxDim;
        float cx = data[baseIdx + BOX_CX_IDX];
        float cy = data[baseIdx + BOX_CY_IDX];
        float w = data[baseIdx + BOX_W_IDX];
        float h = data[baseIdx + BOX_H_IDX];
        float objConf = data[baseIdx + BOX_OBJ_CONF_IDX];
        if (objConf < confThres) continue;
        int classId = 0;
        float maxClsScore = MIN_VALUE;
        for (int c = 0; c < YOLO80_CLASS_COUNT; ++c) {
            int clsIdx = baseIdx + BOX_CLS_START_IDX + c;
            if (clsIdx >= (j + 1) * actualBoxDim) break;
            float clsScore = data[clsIdx];
            if (clsScore > maxClsScore) {
                maxClsScore = clsScore;
                classId = c;
            }
        }
        float finalScore = objConf * maxClsScore;
        if (finalScore < confThres) continue;
        BBox box;
        box.x1 = cx - w / HALF_VALUE;
        box.y1 = cy - h / HALF_VALUE;
        box.x2 = cx + w / HALF_VALUE;
        box.y2 = cy + h / HALF_VALUE;
        box.score = finalScore;
        box.classId = classId;
        box.cocoClassId = -1;
        boxes.push_back(box);
    }
    vector<BBox> nmsResult = BatchedNMS(boxes, nmsThres);
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        LOG(WARNING) << "无法读取图像: " << imgPath;
        return false;
    }
    int originalHeight = img.rows;
    int originalWidth = img.cols;
    float scale;
    int padWidth, padHeight;
    CalculateScaleAndPad(originalHeight, originalWidth, targetWidth, targetHeight, scale, padWidth, padHeight);
    ScaleBoxes(nmsResult, originalWidth, originalHeight, scale, padWidth, padHeight);
    // 映射COCO类别ID
    const size_t cocoSize = yolo80ToCoco90.size();
    for (auto& box : nmsResult) {
        if (box.classId >= 0 && static_cast<size_t>(box.classId) < cocoSize) {
            box.cocoClassId = yolo80ToCoco90[box.classId];
        } else {
            box.cocoClassId = -1;
        }
    }
    SortResults(nmsResult);
    if (!saveTxt.empty()) {
        SaveTxtResults(nmsResult, saveTxt, fileName);
    }


    LOG(INFO) << "====== 图像后处理完成 ======";
    return true;
}

}  // namespace Yolo6s
}  // namespace Infer