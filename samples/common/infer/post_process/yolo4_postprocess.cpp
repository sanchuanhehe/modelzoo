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

#include "yolo4_postprocess.h"
#include "log.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace Infer {
namespace Yolo4 {
    using namespace std;

    // -------------------------- 模型核心参数 --------------------------
    const vector<int> STRIDES = { 8, 16, 32 };
    const vector<int> ANCHORS = { 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401 };
    const vector<vector<int>> MASK_GROUPS = { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };
    const int CLASS_NUM = 80;
    const int SCALE_SIZE = 3; // 每个输出层锚点数量
    const int OUT_PARM_NUM = 85; // 每个锚点的参数数（4+1+80）
    const int TOTAL_CHANNELS = SCALE_SIZE * OUT_PARM_NUM; // 每个输出层总通道数（3*85=255）
    constexpr int BYTE_BIT_NUM = 8;
    constexpr int FP16_BIT_NUM = 16;
    constexpr int FP32_BIT_NUM = 32;

    // YOLO80到COCO90类别ID映射表
    const vector<int> yolo80_to_coco90 = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90
    };

    // -------------------------- 工具函数 --------------------------
    string GetFileName(const string& path)
    {
        size_t sep = path.find_last_of("/\\");
        size_t start = (sep == string::npos) ? 0 : sep + 1;
        size_t end = path.find_last_of(".");
        return (end == string::npos || end <= start) ? path.substr(start) : path.substr(start, end - start);
    }

    inline float Sigmod(float a)
    {
        return 1.0f / (1.0f + exp(-a));
    }

    // -------------------------- 核心功能函数 --------------------------
    vector<BBox> BatchedNMS(vector<BBox>& boxes, float iouThres)
    {
        LOG(INFO) << "NMS处理：输入框数量=" << boxes.size() << ", IOU阈值=" << iouThres;
        map<int, vector<BBox>> classBoxes;
        for (const auto& box : boxes) {
            classBoxes[box.classId].push_back(box);
        }

        vector<BBox> nmsResult;
        for (auto& pair : classBoxes) {
            auto& clsBoxes = pair.second;
            // 按置信度降序排序
            sort(clsBoxes.begin(), clsBoxes.end(), [](const BBox& a, const BBox& b) {
                return a.score > b.score;
            });

            // 执行NMS
            vector<bool> keep(clsBoxes.size(), true);
            for (size_t i = 0; i < clsBoxes.size(); ++i) {
                if (!keep[i])
                    continue;
                const auto& box_i = clsBoxes[i];
                for (size_t j = i + 1; j < clsBoxes.size(); ++j) {
                    if (!keep[j])
                        continue;
                    const auto& box_j = clsBoxes[j];

                    // 计算IOU
                    float inter_left = max(box_i.x1, box_j.x1);
                    float inter_top = max(box_i.y1, box_j.y1);
                    float inter_right = min(box_i.x2, box_j.x2);
                    float inter_bottom = min(box_i.y2, box_j.y2);
                    float inter_area = max(0.0f, inter_right - inter_left) * max(0.0f, inter_bottom - inter_top);
                    float area_i = (box_i.x2 - box_i.x1) * (box_i.y2 - box_i.y1);
                    float area_j = (box_j.x2 - box_j.x1) * (box_j.y2 - box_j.y1);
                    float iou = (area_i + area_j - inter_area) > 0 ? inter_area / (area_i + area_j - inter_area) : 0.0f;

                    if (iou > iouThres)
                        keep[j] = false;
                }
            }

            // 保留有效框
            for (size_t i = 0; i < clsBoxes.size(); ++i) {
                if (keep[i])
                    nmsResult.push_back(clsBoxes[i]);
            }
        }
        LOG(INFO) << "NMS处理完成：保留框数量=" << nmsResult.size();
        return nmsResult;
    }

    void CalculateScaleAndPad(int original_h, int original_w, int target_w, int target_h,
        float& scale, int& pad_w, int& pad_h)
    {
        scale = min(static_cast<float>(target_w) / original_w, static_cast<float>(target_h) / original_h);
        float new_w = original_w * scale;
        float new_h = original_h * scale;
        pad_w = (target_w - static_cast<int>(new_w)) / 2;
        pad_h = (target_h - static_cast<int>(new_h)) / 2;
        LOG(INFO) << "图像缩放配置：原始尺寸(" << original_w << "," << original_h
                  << "), 目标尺寸(" << target_w << "," << target_h
                  << "), 缩放=" << scale << ", pad_w=" << pad_w << ", pad_h=" << pad_h;
    }

    void ScaleBoxes(vector<BBox>& boxes, int original_w, int original_h,
        float scale, int pad_w, int pad_h)
    {
        for (auto& box : boxes) {
            // 反向抵消填充和缩放
            box.x1 = (box.x1 - pad_w) / scale;
            box.y1 = (box.y1 - pad_h) / scale;
            box.x2 = (box.x2 - pad_w) / scale;
            box.y2 = (box.y2 - pad_h) / scale;

            // 确保坐标在图像范围内
            box.x1 = max(0.0f, min(box.x1, static_cast<float>(original_w - 1)));
            box.y1 = max(0.0f, min(box.y1, static_cast<float>(original_h - 1)));
            box.x2 = max(0.0f, min(box.x2, static_cast<float>(original_w - 1)));
            box.y2 = max(0.0f, min(box.y2, static_cast<float>(original_h - 1)));
        }
        LOG(INFO) << "框坐标缩放完成：共" << boxes.size() << "个框";
    }

    void SortResults(vector<BBox>& results)
    {
        sort(results.begin(), results.end(), [](const BBox& a, const BBox& b) {
            if (a.cocoClassId != b.cocoClassId)
                return a.cocoClassId < b.cocoClassId;
            return a.score > b.score;
        });
    }

    void SaveTxtResults(const vector<BBox>& results, const string& saveDir, const string& fileName)
    {
        string txtPath = saveDir + "/" + fileName + ".txt";
        ofstream txtFile(txtPath);
        if (!txtFile.is_open()) { // 增加文件打开失败的判断
            LOG(ERROR) << "无法打开TXT文件: " << txtPath;
            return;
        }
        txtFile.precision(6);
        txtFile << fixed;

        for (const auto& box : results) {
            if (box.cocoClassId == -1)
                continue;
            float w = box.x2 - box.x1;
            float h = box.y2 - box.y1;
            txtFile << box.cocoClassId << " " << box.score << " "
                    << box.x1 << " " << box.y1 << " " << w << " " << h << "\n";
        }
        // 显式关闭文件，释放文件句柄
        txtFile.close();
        LOG(INFO) << "结果保存完成：" << txtPath << "（" << results.size() << "个检测框）";
    }

    const float* GetOutDataAsFP32(const TensorBuf& buf, const TensorDesc& desc, std::vector<float>& preds)
    {
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

    // -------------------------- 内部辅助函数：直接解码原始NCHW数据 --------------------------
    static bool DecodeRawOutputLayer(TensorBuf& buf, TensorDesc& desc, int layerIdx, float confThres,
        int imgWidth, int imgHeight, vector<BBox>& boxes)
    {
        int gridSize = imgWidth / STRIDES[layerIdx];
        size_t totalElements = buf.size / sizeof(float);
        std::vector<float> detectionsPreds;
        const float* rawData = GetOutDataAsFP32(buf, desc, detectionsPreds);
        if (rawData == nullptr) {
            LOG(ERROR) << "输出层" << layerIdx << "数据为空";
            return false;
        }

        // 1. 计算原始NCHW数据的实际宽度（可能存在字节对齐的冗余填充）
        // 公式：总元素数 = 通道数（255） * 高度（gridSize） * 实际宽度（actualWidth）
        int actualWidth = static_cast<int>(totalElements / (TOTAL_CHANNELS * gridSize));
        if (actualWidth < gridSize) {
            LOG(ERROR) << "输出层" << layerIdx << "尺寸异常：实际宽度=" << actualWidth << " < 期望=" << gridSize;
            return false;
        }
        LOG(INFO) << "输出层" << layerIdx << "配置：网格尺寸=" << gridSize << "x" << gridSize
                  << ", 实际宽度=" << actualWidth << ", 总元素数=" << totalElements;

        // 2. 获取当前输出层的锚点（根据MASK_GROUPS）
        auto& mask = MASK_GROUPS[layerIdx];
        vector<int> anchors = {
            ANCHORS[mask[0] * 2], ANCHORS[mask[0] * 2 + 1],
            ANCHORS[mask[1] * 2], ANCHORS[mask[1] * 2 + 1],
            ANCHORS[mask[2] * 2], ANCHORS[mask[2] * 2 + 1]
        };

        size_t initialBoxCount = boxes.size();
        // 3. 遍历每个锚点、每个网格，直接读取原始数据解码
        for (int anchorIdx = 0; anchorIdx < SCALE_SIZE; ++anchorIdx) {
            int anchorW = anchors[anchorIdx * 2];
            int anchorH = anchors[anchorIdx * 2 + 1];
            // 每个锚点对应的通道起始偏移（锚点0→0~84，锚点1→85~169，锚点2→170~254）
            int anchorChannelOffset = anchorIdx * OUT_PARM_NUM;

            for (int y = 0; y < gridSize; ++y) { // 遍历网格高度（y轴）
                for (int x = 0; x < gridSize; ++x) { // 遍历网格宽度（x轴，仅取前gridSize列，跳过冗余填充）
                    // 4. 计算原始NCHW数据中当前参数的地址偏移
                    // 公式：地址 = 通道偏移 + 高度偏移 + 宽度偏移
                    // 通道偏移：当前锚点的通道起始（anchorChannelOffset）
                    // 高度偏移：当前行 * 实际宽度（actualWidth）
                    // 宽度偏移：当前列（x）
                    auto getRawData = [&](int paramIdx) -> float {
                        // paramIdx：0=tx,1=ty,2=tw,3=th,4=objConf,5~84=类别置信度
                        int channel = anchorChannelOffset + paramIdx;
                        size_t dataIdx = channel * gridSize * actualWidth + y * actualWidth + x;
                        return rawData[dataIdx];
                    };

                    // 5. 解码坐标和置信度（直接读取原始数据，无复制）
                    float tx = getRawData(0);
                    float ty = getRawData(1);
                    float tw = getRawData(2);
                    float th = getRawData(3);
                    float objConf = Sigmod(getRawData(4));

                    // 6. 计算最高类别置信度
                    float maxClsConf = 0.0f;
                    int maxClsIdx = 0;
                    for (int c = 0; c < CLASS_NUM; ++c) {
                        float clsConf = Sigmod(getRawData((OUT_PARM_NUM - CLASS_NUM) + c));
                        if (clsConf > maxClsConf) {
                            maxClsConf = clsConf;
                            maxClsIdx = c;
                        }
                    }

                    // 7. 过滤低置信度框
                    float finalConf = objConf * maxClsConf;
                    if (finalConf < confThres)
                        continue;

                    // 8. 计算最终坐标（相对于模型输入尺寸）
                    float cx = (Sigmod(tx) + x) * STRIDES[layerIdx];
                    float cy = (Sigmod(ty) + y) * STRIDES[layerIdx];
                    float w = exp(tw) * anchorW;
                    float h = exp(th) * anchorH;

                    // 9. 添加候选框
                    BBox box;
                    box.x1 = cx - w / 2;
                    box.y1 = cy - h / 2;
                    box.x2 = cx + w / 2;
                    box.y2 = cy + h / 2;
                    box.score = finalConf;
                    box.classId = maxClsIdx;
                    box.cocoClassId = -1;
                    boxes.push_back(box);
                }
            }
        }

        LOG(INFO) << "输出层" << layerIdx << "解码完成：新增候选框=" << boxes.size() - initialBoxCount;
        return true;
    }

    // -------------------------- 对外接口函数 --------------------------
    bool Yolov4Postprocess(vector<string>& fileList,
        vector<TensorBuf>& outBufs,
        vector<TensorDesc>& outDescs)
    {
        LOG(INFO) << "\n====== YOLO4后处理启动 ======";

        // 读取配置参数
        auto cfg = ReadCfgFile("../data/cfg.txt");
        float confThres = stof(cfg["conf_threshold"]);
        float nmsThres = stof(cfg["nms_threshold"]);
        string saveBin = cfg["save_result_bin"];
        string saveTxt = cfg["save_result_txt"];
        // 从cfg读取图像宽高（与preprocess.cpp保持一致）
        int imgWidth = stoi(cfg["img_width"]);
        int imgHeight = stoi(cfg["img_height"]);

        // 创建输出目录
        if (!saveBin.empty() && !CreateDirectoryRecursive(saveBin)) {
            LOG(ERROR) << "创建BIN目录失败：" << saveBin;
            return false;
        }
        if (!saveTxt.empty() && !CreateDirectoryRecursive(saveTxt)) {
            LOG(ERROR) << "创建TXT目录失败：" << saveTxt;
            return false;
        }

        // 校验输入合法性
        if (fileList.empty() || outBufs.size() != 3) {
            LOG(ERROR) << "输入异常：文件列表空或输出层数量≠3（实际=" << outBufs.size() << "）";
            return false;
        }

        size_t imgIdx = 0;
        const string& imgPath = fileList[imgIdx];
        string fileName = GetFileName(imgPath);
        // 保存原始推理结果（可选）
        if (!saveBin.empty()) {
            for (size_t i = 0; i < outBufs.size(); ++i) {
                string binPath = saveBin + "/" + fileName + "_output_" + to_string(i) + ".bin";
                ofstream(binPath, ios::binary).write(reinterpret_cast<char*>(outBufs[i].GetRawPtr()), outBufs[i].size);
            }
            LOG(INFO) << "推理结果保存：" << saveBin << "（3个输出层）";
        }
        // 1. 直接解码原始数据生成候选框
        vector<BBox> boxes;
        bool valid = true;
        for (int i = 0; i < 3; ++i) {
            if (!DecodeRawOutputLayer(outBufs[i], outDescs[i], i, confThres, imgWidth, imgHeight, boxes)) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            LOG(WARNING) << "跳过异常图像：" << imgPath;
            return false;
        }
        LOG(INFO) << "所有输出层处理完成：候选框总数=" << boxes.size();
        // 2. 执行NMS
        if (boxes.empty()) {
            LOG(WARNING) << "无候选框，跳过NMS：" << imgPath;
            return false;
        }
        vector<BBox> nmsResult = BatchedNMS(boxes, nmsThres);
        // 3. 读取原始图像，计算缩放参数
        cv::Mat img = cv::imread(imgPath);
        if (img.empty()) {
            LOG(WARNING) << "读取图像失败：" << imgPath;
            return false;
        }
        int original_w = img.cols;
        int original_h = img.rows;
        float scale;
        int pad_w, pad_h;
        CalculateScaleAndPad(original_h, original_w, imgWidth, imgHeight, scale, pad_w, pad_h);
        // 4. 框坐标缩放回原始图像尺寸
        ScaleBoxes(nmsResult, original_w, original_h, scale, pad_w, pad_h);
        // 5. 类别ID映射（YOLO80 -> COCO90）
        for (auto& box : nmsResult) {
            if (box.classId >= 0 && box.classId < static_cast<int>(yolo80_to_coco90.size())) {
                box.cocoClassId = yolo80_to_coco90[box.classId];
            } else {
                box.cocoClassId = -1;
                LOG(WARNING) << "无效类别ID：" << box.classId;
            }
        }
        // 6. 排序并保存结果
        SortResults(nmsResult);
        if (!saveTxt.empty()) {
            SaveTxtResults(nmsResult, saveTxt, fileName);
        }

        LOG(INFO) << "\n====== YOLO4后处理完成 ======";
        return true;
    }

} // namespace Yolo4
} // namespace Infer