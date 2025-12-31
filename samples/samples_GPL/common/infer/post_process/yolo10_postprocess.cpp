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

#include "yolo10_postprocess.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <climits>
#include <libgen.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <getopt.h>
#include "log.h"
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <map>
#include <cstdio>
#include <numeric>

namespace Infer
{
    using namespace std;
    constexpr float CONF_THRES = 0.25f;
    constexpr float IOU_THRES = 0.45f;
    constexpr int BYTE_BIT_NUM = 8;
    constexpr uint8_t SCALE_SIZE = 3;
    constexpr uint8_t CLASS_NUM = 80;
    constexpr uint8_t OUT_PARM_NUM = 85; /* x, y, w,h, obj , class(80) */
    constexpr int IMG_SIZE = 640;
    constexpr int BBOX_SIZE = 84;
    constexpr int BBOX_NUM = 8400;
    constexpr int BBOX_MAX_NUM = 300;

    std::vector<int> cocoCategoryIdsChangeTo90{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
        54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
        74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

    struct BBox
    {
        float x1, y1, x2, y2; // 边界框坐标
        float score;          // 置信度
        int label;            // 类别标签
    };

    static void ProcessPerDectectionInner(float *outData, int maxDet, int nc, std::vector<BBox> &detections)
    {
        // 直接模拟Python的v10postprocess逻辑
        int numBoxes = BBOX_NUM;
        int numClasses = CLASS_NUM;

        // 分割boxes和scores (更接近Python实现)
        std::vector<float> boxes(numBoxes * 4);
        std::vector<float> scores(numBoxes * numClasses);

        // 直接复制数据，避免复杂索引
        for (int i = 0; i < numBoxes; ++i) {
            // 复制boxes (x, y, w, h),坐标数共为4个
            for (int j = 0; j < 4; ++j) {
                boxes[i * 4 + j] = outData[i * (4 + numClasses) + j];
            }
            // 复制scores
            for (int j = 0; j < numClasses; ++j) {
                scores[i * numClasses + j] = outData[i * (4 + numClasses) + 4 + j];
            }
        }

        // 第一步：找到每个框的最大分数
        std::vector<float> maxScores(numBoxes);
        std::vector<int> maxIndices(numBoxes);

        for (int i = 0; i < numBoxes; ++i)
        {
            float maxScore = -1000.0f;
            int maxIdx = 0;
            for (int j = 0; j < numClasses; ++j)
            {
                if (scores[i * numClasses + j] > maxScore)
                {
                    maxScore = scores[i * numClasses + j];
                    maxIdx = j;
                }
            }
            maxScores[i] = maxScore;
            maxIndices[i] = maxIdx;
        }

        // 第二步：第一次TopK选择maxDet个框
        std::vector<int> firstIndices(numBoxes);
        std::iota(firstIndices.begin(), firstIndices.end(), 0);
        std::stable_sort(firstIndices.begin(), firstIndices.end(),
                         [&](int a, int b)
                         { return maxScores[a] > maxScores[b]; });

        // 第三步：第二次TopK在所有类别分数上
        std::vector<std::pair<float, int>> allScores;
        for (int i = 0; i < maxDet; ++i) {
            int boxIdx = firstIndices[i];
            for (int j = 0; j < numClasses; ++j) {
                allScores.emplace_back(scores[boxIdx * numClasses + j], boxIdx * numClasses + j);
            }
        }

        // 取前maxDet个
        int finalCount = std::min(maxDet, static_cast<int>(allScores.size()));
        std::stable_sort(allScores.begin(), allScores.end(),
                         [](const auto &a, const auto &b)
                         { return a.first > b.first; });

        // 收集最终结果
        detections.clear();
        for (int i = 0; i < finalCount; ++i) {
            float score = allScores[i].first;
            int flatIdx = allScores[i].second;
            int boxIdx = flatIdx / numClasses;
            int classIdx = flatIdx % numClasses;

            BBox det;
            // 复制坐标信息(x, y, x, y),坐标数共为4个
            det.x1 = boxes[boxIdx * 4 + 0];
            det.y1 = boxes[boxIdx * 4 + 1];
            det.x2 = boxes[boxIdx * 4 + 2];
            det.y2 = boxes[boxIdx * 4 + 3];
            det.score = score;
            det.label = classIdx;

            detections.push_back(det);
        }
    }

    // 裁剪边界框到图像范围内
    static void clipBoxes(std::vector<BBox> detections, const cv::Size shape)
    {
        int height = shape.height;
        int width = shape.width;

        for (auto &box : detections) {
            box.x1 = std::max(0.0f, std::min(box.x1, static_cast<float>(width)));
            box.y1 = std::max(0.0f, std::min(box.y1, static_cast<float>(height)));
            box.x2 = std::max(0.0f, std::min(box.x2, static_cast<float>(width)));
            box.y2 = std::max(0.0f, std::min(box.y2, static_cast<float>(height)));
        }
    }

    // 缩放边界框
    static void ScaleBoxes(std::vector<BBox> &detections, const string &filePath)
    {
        int h1 = IMG_SIZE;
        int w1 = IMG_SIZE;
        cv::Mat im0 = cv::imread(filePath);

        // 获取原始尺寸
        cv::Size shape = im0.size();
        int h0 = shape.height;
        int w0 = shape.width;

        // 计算增益 (gain)
        double gain = std::min(static_cast<double>(h1) / h0,
                               static_cast<double>(w1) / w0);

        // 计算填充 (pad)
        double padWidth = (w1 - w0 * gain) / 2.0f;
        double padHeight = (h1 - h0 * gain) / 2.0f;
        // 四舍五入并转换为整数
        int padX = static_cast<int>(std::round(padWidth));
        int padY = static_cast<int>(std::round(padHeight));
        for (BBox &box : detections) {
            // 减去填充
            box.x1 -= padX;
            box.x2 -= padX;
            box.y1 -= padY;
            box.y2 -= padY;

            // 除以增益进行缩放
            box.x1 /= gain;
            box.x2 /= gain;
            box.y1 /= gain;
            box.y2 /= gain;
        }
        clipBoxes(detections, shape);
        return;
    }

    void SaveResultBbox(std::vector<Infer::TensorBuf> &outBufs, std::vector<BBox> bboxs, const std::string &filePath)
    {
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");

        std::string fileName = filePath.substr(start + 1, end - start - 1);
        std::string resultPath = "../out/result";
        std::string binPath = resultPath + "/bin/";
        std::string txtPath = resultPath + "/txt/";
        std::string txtFile = txtPath + fileName + "_result.txt";
        std::ofstream file(txtFile, std::ios::out);
        if (file.is_open()) {
            for (auto &box : bboxs) {
                file << "Class " << cocoCategoryIdsChangeTo90[box.label]
                     << " | Score: " << box.score << " | Box: [" << std::fixed
                     << std::setprecision(6) << box.x1 << ", " << box.y1 << ", "
                     << box.x2 << ", " << box.y2 << "]\n";
            }
            file.close();
        } else {
            LOG(ERROR) << "Open " << txtFile.c_str() << " result txt failed\n";
        }
    }

    void SaveResultBin(std::vector<Infer::TensorBuf> &outBufs,
                       std::vector<BBox> bboxs, const std::string &filePath)
    {
        size_t start = filePath.find_last_of("/");
        size_t end = filePath.find_last_of(".");

        std::string fileName = filePath.substr(start + 1, end - start - 1);
        std::string resultPath = "../out/result";
        std::string binPath = resultPath + "/bin/";
        std::string txtPath = resultPath + "/txt/";
        struct stat info; 
        if (stat(resultPath.c_str(), &info) != 0) {
            mkdir(resultPath.c_str(), 0777);
            INFO_LOG("create file success");
        }

        if (stat(txtPath.c_str(), &info) != 0) {
            mkdir(txtPath.c_str(), 0777);
        }

        if (stat(binPath.c_str(), &info) != 0) {
            mkdir(binPath.c_str(), 0777);
        }

        for (size_t j = 0; j < outBufs.size(); j++) {
            std::string outputFileName =
                binPath + fileName + "_" + std::to_string(j) + ".bin";
            LOG(INFO) << " outputFileName: " << outputFileName;
            std::ofstream binfout(outputFileName, std::ios::out | std::ios::trunc);
            binfout.write(static_cast<char *>(outBufs[j].GetRawPtr()), outBufs[j].size);
            binfout.close();
        }
    }

    static void Transpose(float *data, int rows, int cols)
    {
        float *temp = new float[rows * cols];
        // 执行转置操作
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                temp[j * rows + i] = data[i * cols + j];
            }
        }
        memcpy(data, temp, rows * cols * sizeof(float));
        delete[] temp;
    }

    // 针对单个BBox的版本
    static void Bbox2xywh(std::vector<BBox> &detections)
    {
        for (BBox &bbox : detections) {
            std::vector<float> xywh_box(4);
            xywh_box[0] = (bbox.x1 + bbox.x2) / 2.0f; // x_center
            xywh_box[1] = (bbox.y1 + bbox.y2) / 2.0f; // y_center
            xywh_box[2] = bbox.x2 - bbox.x1;          // width
            xywh_box[3] = bbox.y2 - bbox.y1;          // height

            xywh_box[0] -= xywh_box[2] / 2.0f;
            xywh_box[1] -= xywh_box[3] / 2.0f;
            bbox.x1 = xywh_box[0];
            bbox.x2 = xywh_box[2];
            bbox.y1 = xywh_box[1];
            bbox.y2 = xywh_box[3];
        }
    }

    bool GetYolo10Box(std::vector<std::string> &fileList, std::vector<TensorBuf> &tensorBufs, std::vector<TensorDesc> &tensorDescs)
    {
        if (tensorBufs.size() == 0 || fileList.size() == 0) {
            LOG(INFO) << " modelOuput.size() == 0 || inputFileList.size() == 0  NO OUT";
            return false;
        }
        vector<vector<float>> bboxes;
        vector<vector<float>> vaildBox;
        TensorDesc desc = tensorDescs[0];
        TensorBuf buf = tensorBufs[0];
        if (desc.defaultStride == 0) {
            desc.defaultStride = desc.dims[desc.dimCount - 1] * desc.typeSize / BYTE_BIT_NUM;
            buf.stride = desc.defaultStride;
        }
        // 对矩阵进行转置，从84*8400变为8400*84，方便逐行处理
        float *data = static_cast<float *>(buf.GetRawPtr());

        std::vector<BBox> detections;
        SaveResultBin(tensorBufs, detections, fileList[0]);
        Transpose(data, BBOX_SIZE, BBOX_NUM);

        ProcessPerDectectionInner(data, BBOX_MAX_NUM, CLASS_NUM, detections);
        ScaleBoxes(detections, fileList[0]);
        Bbox2xywh(detections);
        SaveResultBbox(tensorBufs, detections, fileList[0]);
        LOG(INFO) << "dump final data success " << fileList[0];
        return true;
    }
}