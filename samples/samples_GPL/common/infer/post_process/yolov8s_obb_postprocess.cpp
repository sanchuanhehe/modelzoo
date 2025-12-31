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

#include "yolov8s_obb_postprocess.h"
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
namespace Yolo8sObb {
using namespace std;

// 常量定义 - 适配DOTAv1和1024输入
constexpr float MIN_VALUE = 0.0f;
constexpr int DOTA_CLASS_COUNT = 15;  // DOTAv1有15个类别
constexpr int NUM_BOXES = 21504;      // YOLOv8s-OBB输出21504个预测框
constexpr int BOX_DIM_OBB = 5;        // 5个OBB参数：cx, cy, w, h, angle

// 索引匹配官方Python代码的数据布局: [cx, cy, w, h, p(cls0), ..., p(cls14), angle]
constexpr int BOX_CLS_START_IDX = 4;  // 类别分数从第4个开始
constexpr int BOX_CX_IDX = 0;
constexpr int BOX_CY_IDX = 1;
constexpr int BOX_W_IDX = 2;
constexpr int BOX_H_IDX = 3;
constexpr int BOX_ANGLE_IDX = 19;     // 角度参数在最后，索引为19
constexpr float HALF_VALUE = 2.0f;
constexpr float PI = 3.14159265358979323846f;
constexpr int FLOAT_SIZE = sizeof(float);
constexpr int NUM_ATTRS = BOX_DIM_OBB + DOTA_CLASS_COUNT; // 20

// DOTAv1类别名称
static const vector<string> DOTA_CLASS_NAMES = {
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
};

// OBB边界框结构体
struct OBBBox {
    float cx, cy, w, h, angle;  // 中心点坐标、宽高、角度(弧度)
    float score;
    int classId;
    string className;
    
    // 计算四个角点坐标
    vector<cv::Point2f> GetCorners() const {
        vector<cv::Point2f> corners(4);
        float cos_val = cos(angle);
        float sin_val = sin(angle);
        
        // 计算旋转后的四个角点
        float w2 = w / 2.0f;
        float h2 = h / 2.0f;
        
        corners[0] = cv::Point2f(-w2, -h2);
        corners[1] = cv::Point2f(w2, -h2);
        corners[2] = cv::Point2f(w2, h2);
        corners[3] = cv::Point2f(-w2, h2);
        
        // 旋转并平移
        for (auto& corner : corners) {
            float x_new = corner.x * cos_val - corner.y * sin_val + cx;
            float y_new = corner.x * sin_val + corner.y * cos_val + cy;
            corner.x = x_new;
            corner.y = y_new;
        }
        
        return corners;
    }
    
    // 获取DOTA格式字符串
    string ToDOTAFormat() const {
        auto corners = GetCorners();
        stringstream ss;
        ss.precision(2);
        ss << fixed;
        for (const auto& corner : corners) {
            ss << corner.x << " " << corner.y << " ";
        }
        ss << className << " " << score;
        return ss.str();
    }
};

// 提取文件名（保持不变）
string GetFileName(const string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == string::npos || end <= start) ? path.substr(start) : path.substr(start, end - start);
}

// 计算两个旋转框的IoU（优化版本）
float CalculateOBBiou(const OBBBox& box1, const OBBBox& box2) {
    // 快速面积过滤
    float area1 = box1.w * box1.h;
    float area2 = box2.w * box2.h;
    
    // 如果面积相差太大，直接返回0
    if (max(area1, area2) / min(area1, area2) > 100.0f) {
        return 0.0f;
    }
    
    auto corners1 = box1.GetCorners();
    auto corners2 = box2.GetCorners();
    
    // 使用OpenCV计算旋转矩形交集
    cv::RotatedRect rotRect1(cv::Point2f(box1.cx, box1.cy), 
                           cv::Size2f(box1.w, box1.h), 
                           box1.angle * 180.0f / PI);
    cv::RotatedRect rotRect2(cv::Point2f(box2.cx, box2.cy), 
                           cv::Size2f(box2.w, box2.h), 
                           box2.angle * 180.0f / PI);
    
    vector<cv::Point2f> intersection;
    cv::rotatedRectangleIntersection(rotRect1, rotRect2, intersection);
    
    if (intersection.empty() || intersection.size() < 3) {
        return 0.0f;
    }
    
    // 计算交集面积
    float interArea = cv::contourArea(intersection);
    if (interArea <= 0.0f) {
        return 0.0f;
    }
    
    // 计算并集面积
    float unionArea = area1 + area2 - interArea;
    
    return unionArea > MIN_VALUE ? interArea / unionArea : MIN_VALUE;
}

// 旋转框的NMS（优化性能）
vector<OBBBox> BatchedNMS(vector<OBBBox>& boxes, float iouThres) {
    if (boxes.empty()) return {};
    
    // 按类别分组
    map<int, vector<OBBBox>> classBoxes;
    for (const auto& box : boxes) {
        classBoxes[box.classId].push_back(box);
    }

    vector<OBBBox> nmsResult;
    for (auto& clsPair : classBoxes) {
        vector<OBBBox>& clsBoxes = clsPair.second;

        // 按置信度降序排序
        sort(clsBoxes.begin(), clsBoxes.end(), [](const OBBBox& a, const OBBBox& b) {
            return a.score > b.score;
        });

        vector<bool> keep(clsBoxes.size(), true);
        for (size_t i = 0; i < clsBoxes.size(); ++i) {
            if (!keep[i]) continue;
            nmsResult.push_back(clsBoxes[i]);

            // 计算当前框与后续框的IoU（使用旋转框IoU）
            const OBBBox& boxI = clsBoxes[i];
            for (size_t j = i + 1; j < clsBoxes.size(); ++j) {
                if (!keep[j]) continue;
                const OBBBox& boxJ = clsBoxes[j];

                // 快速距离过滤
                float dx = boxI.cx - boxJ.cx;
                float dy = boxI.cy - boxJ.cy;
                float distance = sqrt(dx * dx + dy * dy);
                float maxDiag = max(sqrt(boxI.w * boxI.w + boxI.h * boxI.h),
                                  sqrt(boxJ.w * boxJ.w + boxJ.h * boxJ.h));
                
                if (distance > maxDiag * 2.0f) {
                    continue; // 距离太远，跳过IoU计算
                }

                float iou = CalculateOBBiou(boxI, boxJ);
                if (iou > iouThres) {
                    keep[j] = false;
                }
            }
        }
    }
    return nmsResult;
}

// 计算缩放比例和填充量（适配1024输入）
void CalculateScaleAndPad(int originalHeight, int originalWidth, int targetWidth, int targetHeight,
                          float& scale, int& padWidth, int& padHeight) {
    scale = min(static_cast<float>(targetWidth) / originalWidth,
                static_cast<float>(targetHeight) / originalHeight);

    float newWidth = originalWidth * scale;
    float newHeight = originalHeight * scale;

    padWidth = (targetWidth - static_cast<int>(newWidth)) / 2;
    padHeight = (targetHeight - static_cast<int>(newHeight)) / 2;
}

// 缩放旋转框到原始图像尺寸
void ScaleBoxes(vector<OBBBox>& boxes, int originalWidth, int originalHeight,
                float scale, int padWidth, int padHeight) {
    for (auto& box : boxes) {
        // 缩放中心点坐标
        box.cx = (box.cx - padWidth) / scale;
        box.cy = (box.cy - padHeight) / scale;
        
        // 缩放宽高
        box.w = box.w / scale;
        box.h = box.h / scale;
        
        // 角度保持不变
        
        // 边界检查 - 确保在图像范围内
        box.cx = max(MIN_VALUE, min(box.cx, static_cast<float>(originalWidth)));
        box.cy = max(MIN_VALUE, min(box.cy, static_cast<float>(originalHeight)));
        box.w = max(1.0f, min(box.w, static_cast<float>(originalWidth)));
        box.h = max(1.0f, min(box.h, static_cast<float>(originalHeight)));
    }
}

// 按类别和置信度排序
void SortResults(vector<OBBBox>& results) {
    sort(results.begin(), results.end(), [](const OBBBox& a, const OBBBox& b) {
        if (a.classId != b.classId) {
            return a.classId < b.classId;
        } else {
            if (a.score == b.score) {
                return a.cx < b.cx;
            }
            return a.score > b.score;
        }
    });
}

// 保存DOTA格式的TXT结果
void SaveDOTAResults(const vector<OBBBox>& sortedResult, const string& saveTxt, const string& fileName) {
    string txtPath = saveTxt + "/" + fileName + ".txt";
    ofstream txtFile(txtPath);
    if (!txtFile.is_open()) {
        LOG(ERROR) << "无法打开TXT文件: " << txtPath;
        return;
    }
    txtFile.precision(2);
    txtFile << fixed;
    for (const auto& box : sortedResult) {
        // DOTA格式: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
        // 这里用置信度作为difficulty
        txtFile << box.ToDOTAFormat() << endl;
    }
    txtFile.close();
}

// 核心后处理函数
bool Yolov8sObbPostprocess(vector<string>& fileList,
                          vector<TensorBuf>& outBufs,
                          vector<TensorDesc>& outDescs) {
    LOG(INFO) << "====== 开始YOLOv8s-OBB后处理 (DOTAv1) ======";
    auto cfg = ReadCfgFile("../data/cfg.txt");
    float confThres = stof(cfg["conf_threshold"]);
    float nmsThres = stof(cfg["nms_threshold"]);
    string saveBin = cfg["save_result_bin"];
    string saveTxt = cfg["save_result_txt"];
    string preProcDir = cfg["save_preprocess_bin"];
    int targetWidth = 1024;  // 固定1024输入
    int targetHeight = 1024;

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

    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const string& imgPath = fileList[imgIdx];
        string fileName = GetFileName(imgPath);
        // 保存二进制输出（可选）
        if (!saveBin.empty()) {
            string binPath = saveBin + "/" + fileName + "_output_0.bin";
            ofstream binFile(binPath, ios::binary);
            if (binFile.is_open()) {
                binFile.write(reinterpret_cast<char*>(outBufs[imgIdx].data.get()), outBufs[imgIdx].size);
                binFile.close();
            } else {
                LOG(WARNING) << "无法打开bin文件: " << binPath;
            }
        }

        // 解析模型输出 - 适配20×21504输出
        float* original_data = reinterpret_cast<float*>(outBufs[imgIdx].data.get());
        if (original_data == nullptr) {
            LOG(WARNING) << "模型输出缓冲区为空: " << fileName;
            continue;
        }
        
        int64_t bufSize = outBufs[imgIdx].size;
        int totalElements = bufSize / FLOAT_SIZE;
        
        // 验证输出尺寸
        int expectedElements = NUM_BOXES * (BOX_DIM_OBB + DOTA_CLASS_COUNT);
        if (totalElements != expectedElements) {
            LOG(ERROR) << "输出尺寸不匹配: 期望 " << expectedElements 
                      << ", 实际 " << totalElements;
            continue;
        }

        // 使用OpenCV进行高效转置
        // 1. 创建一个cv::Mat头部，包装原始数据，无需复制。
        // 形状: [20, 21504], 类型: CV_32F (32位浮点)
        cv::Mat original_mat(NUM_ATTRS, NUM_BOXES, CV_32F, original_data);

        // 2. 为转置后的数据创建一个缓冲区。
        vector<float> transposed_data(totalElements);

        // 3. 创建一个cv::Mat头部，包装新的缓冲区，用于接收转置结果。
        // 形状: [21504, 20], 类型: CV_32F
        cv::Mat transposed_mat(NUM_BOXES, NUM_ATTRS, CV_32F, transposed_data.data());

        // 4. 执行转置。这是一个高度优化的操作，通常使用SIMD指令。
        cv::transpose(original_mat, transposed_mat);

        // 5. 获取指向转置后数据的指针，用于后续处理。
        float* data = transposed_data.data();

        vector<OBBBox> boxes;
        int processedBoxes = 0;
        
        // 解析21504个预测框
        // data 指向了转置后的 [21504, 20] 数据
        for (int j = 0; j < NUM_BOXES; ++j) {
            int baseIdx = j * (BOX_DIM_OBB + DOTA_CLASS_COUNT);

            // 提取OBB参数
            float cx = data[baseIdx + BOX_CX_IDX];
            float cy = data[baseIdx + BOX_CY_IDX];
            float w = data[baseIdx + BOX_W_IDX];
            float h = data[baseIdx + BOX_H_IDX];
            float angle = data[baseIdx + BOX_ANGLE_IDX];  // 角度(弧度)
            
            // 查找最大类别分数
            int classId = 0;
            float maxClsScore = MIN_VALUE;
            for (int c = 0; c < DOTA_CLASS_COUNT; ++c) {
                float clsScore = data[baseIdx + BOX_CLS_START_IDX + c];
                if (clsScore > maxClsScore) {
                    maxClsScore = clsScore;
                    classId = c;
                }
            }

            // 使用类别分数作为置信度（YOLOv8-OBB没有单独的obj_conf）
            float finalScore = maxClsScore;
            if (finalScore < confThres) continue;

            // 创建OBB框
            OBBBox box;
            box.cx = cx;
            box.cy = cy;
            box.w = w;
            box.h = h;
            box.angle = angle;
            box.score = finalScore;
            box.classId = classId;
            if (classId >= 0 && classId < DOTA_CLASS_COUNT) {
                box.className = DOTA_CLASS_NAMES[classId];
            } else {
                box.className = "unknown";
            }

            boxes.push_back(box);
            processedBoxes++;
        }

        // 执行NMS
        vector<OBBBox> nmsResult = BatchedNMS(boxes, nmsThres);
        LOG(INFO) << "NMS后剩余 " << nmsResult.size() << " 个检测框";

        // 读取原始图像获取尺寸
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            LOG(WARNING) << "无法读取图像: " << imgPath;
            continue;
        }
        int originalHeight = img.rows;
        int originalWidth = img.cols;

        // 计算缩放参数并还原坐标
        float scale;
        int padWidth, padHeight;
        CalculateScaleAndPad(originalHeight, originalWidth, targetWidth, targetHeight, 
                           scale, padWidth, padHeight);
        ScaleBoxes(nmsResult, originalWidth, originalHeight, scale, padWidth, padHeight);

        // 排序结果
        SortResults(nmsResult);

        // 保存DOTA格式结果
        if (!saveTxt.empty()) {
            SaveDOTAResults(nmsResult, saveTxt, fileName);
        }

        // 删除前处理文件, 避免撑爆内存
        string preProcPath = preProcDir + "/" + fileName + ".bin";
        if (ifstream(preProcPath)) {
            remove(preProcPath.c_str());
        }
    }

    LOG(INFO) << "====== YOLOv8s-OBB DOTAv1后处理完成 ======";
    return true;
}

}  // namespace Yolo8sObb
}  // namespace Infer
