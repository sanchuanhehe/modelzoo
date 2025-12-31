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

#include "yolo11s_pose_postprocess.h"
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
#include <stdexcept>  // 新增：异常处理头文件

namespace Infer {
namespace Yolo11sPoseNS {
using namespace std;

constexpr int BYTE_BIT_NUM = 8;
constexpr int FP16_BIT_NUM = 16;
constexpr int FP32_BIT_NUM = 32;

struct BBox {
    float x1, y1, x2, y2, score;
    int classId;  // YOLO11s-pose默认为1（人体）
    vector<float> keypoints;  // 17个关键点(x,y,score)，共51个值
};

// 提取文件名（不含路径和扩展名）
string GetFileName(const string& path) {
    const size_t sep = path.find_last_of("/\\");
    const size_t start = (sep == string::npos) ? 0 : sep + 1;
    const size_t end = path.find_last_of(".");
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
    for (auto& pair : classBoxes) {
        vector<BBox>& clsBoxes = pair.second;

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

                const float interLeft = max(boxI.x1, boxJ.x1);
                const float interTop = max(boxI.y1, boxJ.y1);
                const float interRight = min(boxI.x2, boxJ.x2);
                const float interBottom = min(boxI.y2, boxJ.y2);
                const float interArea = max(0.0f, interRight - interLeft) * max(0.0f, interBottom - interTop);

                const float areaI = (boxI.x2 - boxI.x1) * (boxI.y2 - boxI.y1);
                const float areaJ = (boxJ.x2 - boxJ.x1) * (boxJ.y2 - boxJ.y1);
                const float unionArea = areaI + areaJ - interArea;
                const float iou = unionArea > 0 ? interArea / unionArea : 0.0f;

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

    const float newWidth = originalWidth * scale;
    const float newHeight = originalHeight * scale;

    const int dw = targetWidth - static_cast<int>(newWidth);
    const int dh = targetHeight - static_cast<int>(newHeight);
    padWidth = dw / 2;
    padHeight = dh / 2;
}

// 缩放边界框到原始图像尺寸
void ScaleBoxes(vector<BBox>& boxes, int originalWidth, int originalHeight,
                float scale, int padWidth, int padHeight) {
    const float keyPointScoreThres = 0.1f;
    for (auto& box : boxes) {
        // 调整边界框坐标
        box.x1 = (box.x1 - padWidth) / scale;
        box.y1 = (box.y1 - padHeight) / scale;
        box.x2 = (box.x2 - padWidth) / scale;
        box.y2 = (box.y2 - padHeight) / scale;

        // 边界裁剪
        box.x1 = max(0.0f, min(box.x1, static_cast<float>(originalWidth)));
        box.y1 = max(0.0f, min(box.y1, static_cast<float>(originalHeight)));
        box.x2 = max(0.0f, min(box.x2, static_cast<float>(originalWidth)));
        box.y2 = max(0.0f, min(box.y2, static_cast<float>(originalHeight)));

        // 调整关键点坐标
        for (size_t k = 0; k < box.keypoints.size(); k += 3) {
            float x = box.keypoints[k];
            float y = box.keypoints[k + 1];
            float score = box.keypoints[k + 2];

            if (score > keyPointScoreThres) {
                x = (x - padWidth) / scale;
                y = (y - padHeight) / scale;
                x = max(0.0f, min(x, static_cast<float>(originalWidth)));
                y = max(0.0f, min(y, static_cast<float>(originalHeight)));
            } else {
                x = 0.0f;
                y = 0.0f;
                score = 0.0f;
            }

            box.keypoints[k] = x;
            box.keypoints[k + 1] = y;
            box.keypoints[k + 2] = score;
        }
    }
}

// 按置信度排序结果
void SortResults(vector<BBox>& results) {
    sort(results.begin(), results.end(), [](const BBox& a, const BBox& b) {
        return a.score > b.score;
    });
}

// 保存TXT结果
void SaveTxtResults(const vector<BBox>& results, const string& saveTxt, const string& fileName) {
    try {  // 新增：文件操作异常捕获
        const string txtPath = saveTxt + "/" + fileName + ".txt";
        ofstream txtFile(txtPath);
        if (!txtFile.is_open()) {
            LOG(ERROR) << "无法打开文件用于写入: " << txtPath;
            return;
        }
        txtFile.precision(6);
        txtFile << fixed;

        for (const auto& box : results) {
            const float width = box.x2 - box.x1;
            const float height = box.y2 - box.y1;

            // 格式: category_id score x y width height keypoint1_x ...
            txtFile << box.classId << " "
                    << box.score << " "
                    << box.x1 << " "
                    << box.y1 << " "
                    << width << " "
                    << height << " ";

            // 写入关键点
            for (size_t i = 0; i < box.keypoints.size(); ++i) {
                txtFile << box.keypoints[i];
                if (i != box.keypoints.size() - 1) {
                    txtFile << " ";
                }
            }
            txtFile << "\n";
        }
    } catch (const exception& e) {
        LOG(ERROR) << "保存TXT结果时发生错误: " << e.what() << "，文件名: " << fileName;
    }
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
bool Yolov11sPosePostprocess(vector<string>& fileList,
                            vector<TensorBuf>& outBufs,
                            vector<TensorDesc>& outDescs) {
    try {  // 新增：外层总异常捕获
        LOG(INFO) << "====== 开始YOLO11s-pose后处理 ======";
        auto cfg = ReadCfgFile("../data/cfg.txt");
        if (cfg.empty()) {
            LOG(ERROR) << "配置文件读取为空";
            return false;
        }

        // 新增：配置参数解析异常处理
        float confThres, nmsThres;
        int targetWidth, targetHeight;
        string saveBin, saveTxt;

        try {
            if (cfg.find("conf_threshold") == cfg.end()) throw runtime_error("未找到conf_threshold配置");
            confThres = stof(cfg["conf_threshold"]);

            if (cfg.find("nms_threshold") == cfg.end()) throw runtime_error("未找到nms_threshold配置");
            nmsThres = stof(cfg["nms_threshold"]);

            if (cfg.find("save_result_bin") == cfg.end()) throw runtime_error("未找到save_result_bin配置");
            saveBin = cfg["save_result_bin"];

            if (cfg.find("save_result_txt") == cfg.end()) throw runtime_error("未找到save_result_txt配置");
            saveTxt = cfg["save_result_txt"];

            if (cfg.find("img_width") == cfg.end()) throw runtime_error("未找到img_width配置");
            targetWidth = stoi(cfg["img_width"]);

            if (cfg.find("img_height") == cfg.end()) throw runtime_error("未找到img_height配置");
            targetHeight = stoi(cfg["img_height"]);
        } catch (const exception& e) {
            LOG(ERROR) << "解析配置参数时出错: " << e.what();
            return false;
        }

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

        const int numBoxes = 8400;
        const int boxDim = 56;  // 4(bbox) + 1(score) + 51(keypoints)
        const int keypointNum = 17;
        const int topk = 1000;   // 添加topk参数，与Python代码保持一致
        const int coordinateOffset = 4;
        const int keypointStep = 3;

        size_t imgIdx = 0;
        const string& imgPath = fileList[imgIdx];
        const string fileName = GetFileName(imgPath);
        // 保存原始bin文件（可选）
        if (!saveBin.empty()) {
            const string binPath = saveBin + "/" + fileName + "_output_0.bin";
            ofstream binFile(binPath, ios::binary);
            if (!binFile.is_open()) {
                LOG(WARNING) << "无法打开bin文件用于写入: " << binPath << "，跳过保存bin文件";
            } else {
                binFile.write(reinterpret_cast<char*>(outBufs[imgIdx].data.get()), outBufs[imgIdx].size);
            }
        }
        // 解析模型输出
        if (outBufs[imgIdx].data == nullptr) {  // 新增：空指针检查
            LOG(ERROR) << "输出缓冲区为空，图片: " << imgPath;
            return false;
        }
        // float* data = reinterpret_cast<float*>(outBufs[imgIdx].data.get());
        const int64_t bufSize = outBufs[imgIdx].size;
        const int totalElements = bufSize / sizeof(float);
        if (totalElements != numBoxes * boxDim) {
            LOG(ERROR) << "输出尺寸异常! 预期" << numBoxes * boxDim
                      << "元素，实际" << totalElements << "元素，图片: " << imgPath;
            return false;
        }
        std::vector<float> detectionsPreds;  
        const float* data = GetOutDataAsFP32(outBufs[imgIdx], outDescs[imgIdx], detectionsPreds);
        vector<BBox> boxes;
        // 注意：修复维度解析，模型输出是(56, 8400)，需要转置为(8400, 56)
        for (int j = 0; j < numBoxes; ++j) {
            // 计算转置后的索引，对应Python中的detections.transpose()
            const float cx = data[0 * numBoxes + j];    // 原(0, j) → 转置后(j, 0)
            const float cy = data[1 * numBoxes + j];    // 原(1, j) → 转置后(j, 1)
            const float w = data[2 * numBoxes + j];     // 原(2, j) → 转置后(j, 2)
            const float h = data[3 * numBoxes + j];     // 原(3, j) → 转置后(j, 3)
            const float score = data[coordinateOffset * numBoxes + j]; // 原(4, j) → 转置后(j, 4)
            // 过滤低置信度目标
            if (score < confThres) continue;
            // 解析关键点 (原(5+3k, j) → 转置后(j, 5+3k))
            vector<float> keypoints;
            for (int k = 0; k < keypointNum; ++k) {
                const int kpBase = coordinateOffset + 1 + k * keypointStep;
                keypoints.push_back(data[kpBase * numBoxes + j]);     // x
                keypoints.push_back(data[(kpBase + 1) * numBoxes + j]); // y
                keypoints.push_back(data[(kpBase + 2) * numBoxes + j]); // score
            }
            // 转换为xyxy格式
            BBox box;
            const float half = 2.0f;
            box.x1 = cx - w / half;
            box.y1 = cy - h / half;
            box.x2 = cx + w / half;
            box.y2 = cy + h / half;
            box.score = score;
            box.classId = 1;  // YOLO11s-pose默认为人体检测
            box.keypoints = keypoints;
            boxes.push_back(box);
        }
        // 添加Top-k筛选，与Python代码保持一致
        if (boxes.size() > topk) {
            sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
                return a.score > b.score;
            });
            boxes.resize(topk);
        }
        // 执行NMS
        vector<BBox> nmsResult = BatchedNMS(boxes, nmsThres);
        // 读取原图获取尺寸
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            LOG(WARNING) << "无法读取图像: " << imgPath;
            return false;
        }
        const int originalHeight = img.rows;
        const int originalWidth = img.cols;
        // 计算缩放和填充参数
        float scale;
        int padWidth, padHeight;
        CalculateScaleAndPad(originalHeight, originalWidth, targetWidth, targetHeight, scale, padWidth, padHeight);
        // 缩放边界框和关键点到原始图像尺寸
        ScaleBoxes(nmsResult, originalWidth, originalHeight, scale, padWidth, padHeight);
        // 按置信度排序
        SortResults(nmsResult);
        // 保存结果到TXT
        if (!saveTxt.empty()) {
            SaveTxtResults(nmsResult, saveTxt, fileName);
        }

        LOG(INFO) << "====== YOLO11s-pose后处理完成 ======";
        return true;
    } catch (const exception& e) {
        LOG(ERROR) << "后处理总流程发生致命错误: " << e.what();
        return false;
    }
}

}  // namespace Yolo11sPoseNS
}  // namespace Infer