/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
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

#include "yolov9s_postprocess.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "log.h"
#include "utils.h"

namespace Infer {

constexpr int NUM_CLASSES = 80;
constexpr int BBOX_SIZE = 84;
constexpr int BBOX_NUM = 8400;
constexpr int MAX_DET = 300;
constexpr int BBOX_COORD_NUM = 4;

static int originWidth = 640;
static int originHeight = 640;

// 检测框结构体
struct BBox {
    float x1, y1, x2, y2; // 左上和右下坐标
    float score; // 置信度
    int classId; // 类别ID
};

struct ContentRect {
    int x1;
    int y1;
    int x2;
    int y2;
};

struct PostCfg {
    float confThres;
    float nmsThres;
    int imgHeight;
    int imgWidth;
    std::string saveBin;
    std::string saveTxt;
};

// 提取无后缀文件名
static std::string GetFileName(const std::string& path) {
    size_t sep = path.find_last_of("/\\");
    size_t start = (sep == std::string::npos) ? 0 : sep + 1;
    size_t end = path.find_last_of(".");
    return (end == std::string::npos || end <= start) ? 
           path.substr(start) : path.substr(start, end - start);
}

static float CalculateScale(int originalHeight, int originalWidth, int targetWidth, int targetHeight)
{
    // 等比缩放比：取宽/高方向的较小值，确保缩放后不超出目标框
    return std::min(static_cast<float>(targetWidth) / originalWidth,
                    static_cast<float>(targetHeight) / originalHeight);
}

static ContentRect CalculateContentRect(
    int originalHeight, int originalWidth, int targetWidth, int targetHeight, float scale)
{
    // 先算缩放后区域
    const int resizedWidth = std::min(
        static_cast<int>(std::ceil(static_cast<float>(originalWidth) * scale)), targetWidth);
    const int resizedHeight = std::min(
        static_cast<int>(std::ceil(static_cast<float>(originalHeight) * scale)), targetHeight);

    // 再算目标区域矩形
    const int x1 = static_cast<int>(std::round(
        (static_cast<float>(targetWidth) - static_cast<float>(resizedWidth)) / 2.0f - 0.1f));
    const int y1 = static_cast<int>(std::round(
        (static_cast<float>(targetHeight) - static_cast<float>(resizedHeight)) / 2.0f - 0.1f));
    return { x1, y1, x1 + resizedWidth, y1 + resizedHeight };
}

static BBox ScaleBboxToOriginal(const BBox& box, float scale, const ContentRect& contentRect)
{
    BBox boxNew;
    // 先去掉内容区域左上偏移
    boxNew.x1 = std::max(0.0f, std::min((box.x1 - contentRect.x1) / scale, (float)originWidth));
    boxNew.y1 = std::max(0.0f, std::min((box.y1 - contentRect.y1) / scale, (float)originHeight));
    boxNew.x2 = std::max(0.0f, std::min((box.x2 - contentRect.x1) / scale, (float)originWidth));
    boxNew.y2 = std::max(0.0f, std::min((box.y2 - contentRect.y1) / scale, (float)originHeight));
    boxNew.score = box.score;
    boxNew.classId = box.classId;

    return boxNew;
}

static void SaveBboxResult(std::vector<BBox> bboxs,
    const std::string& fileName, const std::string& saveTxt, float scale, const ContentRect& contentRect)
{
    mode_t oldUmask = umask(0);
    
    // 保存检测框结果
    std::string txtFile = saveTxt + "/" + fileName + "_result.txt";
    std::ofstream file1(txtFile, std::ios::out);
    if (file1.is_open()) {
        for (auto& box1 : bboxs) {
            auto box = ScaleBboxToOriginal(box1, scale, contentRect);
            file1 << "Class " << box.classId << " | Score: " << box.score
                  << " | Box: [" << box.x1 << ", " << box.y1 << ", "
                  << box.x2 << ", " << box.y2 << "]\n";
        }
        file1.close();
    } else {
        LOG(ERROR) << "open result txt failed, " << txtFile;
    }

    umask(0);
}

static void SaveResultBin(const TensorBuf& outBuf, const std::string& imgPath, const std::string& saveBin)
{
    const std::string fileName = GetFileName(imgPath);
    std::string binPath = saveBin + "/" + fileName + ".bin";
    std::ofstream binFile(binPath, std::ios::binary);
    if (binFile.is_open()) {
        binFile.write(reinterpret_cast<const char*>(outBuf.GetRawPtr()), outBuf.size);
        binFile.close();
        LOG(INFO) << "Saved result bin: " << binPath;
    } else {
        LOG(WARNING) << "无法打开bin文件: " << binPath;
    }
}

static void SaveResultTxt(const std::vector<BBox>& bboxs, const std::string& imgPath, const PostCfg& cfg)
{
    // 读取原图获取尺寸
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        LOG(WARNING) << "无法读取图像: " << imgPath;
        return;
    }
    originHeight = img.rows;
    originWidth = img.cols;

    const float scale = CalculateScale(originHeight, originWidth, cfg.imgWidth, cfg.imgHeight);
    const ContentRect contentRect = CalculateContentRect(originHeight, originWidth, cfg.imgWidth, cfg.imgHeight, scale);
    const std::string fileName = GetFileName(imgPath);
    SaveBboxResult(bboxs, fileName, cfg.saveTxt, scale, contentRect);
}

static PostCfg LoadPostCfg()
{
    // 从cfg读取后处理参数
    auto cfg = ReadCfgFile("../data/cfg.txt");
    return {
        std::stof(cfg["conf_threshold"]),
        std::stof(cfg["nms_threshold"]),
        std::stoi(cfg["img_height"]),
        std::stoi(cfg["img_width"]),
        cfg["save_result_bin"],
        cfg["save_result_txt"],
    };
}

static bool PreparePostRun(const std::vector<std::string>& fileList, const std::vector<TensorBuf>& outBufs,
    const PostCfg& cfg)
{
    if (!cfg.saveBin.empty() && !CreateDirectoryRecursive(cfg.saveBin)) {
        LOG(ERROR) << "创建bin目录失败: " << cfg.saveBin;
        return false;
    }
    if (!cfg.saveTxt.empty() && !CreateDirectoryRecursive(cfg.saveTxt)) {
        LOG(ERROR) << "创建txt目录失败: " << cfg.saveTxt;
        return false;
    }

    if (fileList.size() != outBufs.size()) {
        LOG(ERROR) << "文件列表与输出缓冲区数量不匹配";
        return false;
    }
    return true;
}

// 计算两个框的IoU（交并比）
static float CalculateIoU(const BBox& box1, const BBox& box2)
{
    // 计算交集区域
    float interX1 = std::max(box1.x1, box2.x1);
    float interY1 = std::max(box1.y1, box2.y1);
    float interX2 = std::min(box1.x2, box2.x2);
    float interY2 = std::min(box1.y2, box2.y2);

    // 计算交集面积
    float interArea = std::max(0.0f, interX2 - interX1) * std::max(0.0f, interY2 - interY1);

    // 计算各自面积
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    // 计算并集面积
    float unionArea = area1 + area2 - interArea;

    if (unionArea <= 0.0f) {
        return 0.0f;
    }
    return interArea / unionArea;
}


// NMS主函数
static std::vector<BBox> NMS(std::vector<BBox>& boxes, float iouThres)
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
            if (CalculateIoU(boxes[i], boxes[j]) > iouThres) {
                keep[j] = false; // 标记为移除
            }
        }
    }

    return result;
}

// 按类别分组NMS
static std::vector<BBox> MulticlassNms(std::vector<BBox>& boxes, float iouThres)
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
                auto nmsResult = NMS(classBoxes, iouThres);
                result.insert(result.end(), nmsResult.begin(), nmsResult.end());
                classBoxes.clear();
            }
            currentClass = box.classId;
        }
        classBoxes.push_back(box);
    }

    // 处理最后一个类别
    if (!classBoxes.empty()) {
        auto nmsResult = NMS(classBoxes, iouThres);
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

static std::vector<BBox> GetBBox(TensorBuf& outBuf, float confThres)
{
    std::vector<BBox> bboxs;
    float* data = static_cast<float*>(outBuf.GetRawPtr());

    for (int i = 0; i < BBOX_NUM; i++) {
        // 提取类别分数
        const float* scores = data + BBOX_COORD_NUM;
        int classId = std::max_element(scores, scores + NUM_CLASSES) - scores;
        float conf = scores[classId];

        // 获取x y x y
        float x1 = data[0] - data[2] / 2;
        float y1 = data[1] - data[3] / 2;
        float x2 = data[0] + data[2] / 2;
        float y2 = data[1] + data[3] / 2;

        // 置信度过滤
        if (conf >= confThres) {
            bboxs.push_back({ x1, y1, x2, y2, conf, classId });
        }
        data += BBOX_SIZE;
    }
    return bboxs;
}

static std::vector<BBox> GetNmsBboxs(TensorBuf& outBuf, float confThres, float iouThres)
{
    std::vector<BBox> bboxs = std::move(GetBBox(outBuf, confThres));
    std::vector<BBox> result = MulticlassNms(bboxs, iouThres);
    std::sort(result.begin(), result.end(), [](const BBox& a, const BBox& b) { return a.score > b.score; });
    if (result.size() > MAX_DET) {
        result.resize(MAX_DET);
    }
    return result;
}

static void ProcessSingleImageResult(TensorBuf& outBuf, const std::string& imgPath, const PostCfg& cfg)
{
    // 先保存原始输出供评估使用
    if (!cfg.saveBin.empty()) {
        SaveResultBin(outBuf, imgPath, cfg.saveBin);
    }

    // 转成8400x84便于逐框解析
    float* data = static_cast<float*>(outBuf.GetRawPtr());
    Transpose(data, BBOX_SIZE, BBOX_NUM);

    // 解码并做NMS
    std::vector<BBox> bboxs = std::move(GetNmsBboxs(outBuf, cfg.confThres, cfg.nmsThres));

    // 保存后处理结果
    if (!cfg.saveTxt.empty()) {
        SaveResultTxt(bboxs, imgPath, cfg);
    }
}

bool GetYoloV9sBox(std::vector<std::string>& fileList,
                   std::vector<TensorBuf>& outBufs,
                   std::vector<TensorDesc>& outDescs) {
    const PostCfg cfg = LoadPostCfg();
    if (!PreparePostRun(fileList, outBufs, cfg)) {
        return false;
    }

    for (size_t imgIdx = 0; imgIdx < fileList.size(); ++imgIdx) {
        const std::string& imgPath = fileList[imgIdx];
        TensorBuf& outBuf = outBufs[imgIdx];
        ProcessSingleImageResult(outBuf, imgPath, cfg);
    }

    LOG(INFO) << "Finished data postprocess successfully, " << fileList[0];
    return true;
}

} // namespace Infer
