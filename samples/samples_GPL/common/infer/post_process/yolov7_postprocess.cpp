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

#include "yolov7_postprocess.h"

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
constexpr int numClasses = 80;
constexpr float confThres = 0.001f;
constexpr float iouThres = 0.65f;
constexpr int yolov7InputSize = 640;
int originWidth = 640;
int originHeight = 640;

constexpr int numBoxes = 25200;
constexpr int dpicoBoxAttrs = 88;
int boxAttrs = 85;

static const int BYTE_BIT_NUM = 8;

std::vector<int> cocoCategoryIds{
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17,
    18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53,
    54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

// 定义检测框结构体
struct BBox {
  float x1, y1, x2, y2;  // 左上和右下坐标
  float score;           // 置信度
  int classId;           // 类别ID
};

void SaveBboxResult(std::vector<Infer::TensorBuf>& outBufs,
                    std::vector<BBox> bboxs, const std::string& filePath) {
  size_t start = filePath.find_last_of("/");
  size_t end = filePath.find_last_of(".");

  // std::string txtPath = filePath.substr(0, start) + "/result/txt/";
  std::string fileName = filePath.substr(start + 1, end - start - 1);
  std::string resultPath = "../out/result";
  std::string binPath = resultPath + "/bin/";
  std::string txtPath = resultPath + "/txt/";
  for (auto& path : {resultPath, binPath, txtPath}) {
    if (stat(path.c_str(), nullptr) != 0) {
      mkdir(path.c_str(), 0755);
    }
  }

  for (size_t j = 0; j < outBufs.size(); j++) {
    std::string outputFileName =
        binPath + fileName + "_" + std::to_string(j) + ".bin";
    LOG(INFO) << " outputFileName: " << outputFileName;
    std::ofstream binfout(outputFileName, std::ios::out | std::ios::trunc);
    binfout.write(static_cast<char*>(outBufs[j].GetRawPtr()), outBufs[j].size);
    binfout.close();
  }

  std::string txtFile = txtPath + fileName + "_result.txt";
  std::ofstream file(txtFile, std::ios::out);
  if (file.is_open()) {
    for (auto& box : bboxs) {
      file << "Class " << cocoCategoryIds[box.classId]
           << " | Score: " << box.score << " | Box: [" << std::fixed
           << std::setprecision(6) << box.x1 << ", " << box.y1 << ", "
           << box.x2 - box.x1 << ", " << box.y2 - box.y1 << "]\n";
    }
    file.close();
    LOG(INFO) << "Saved " << bboxs.size() << " detection results to "
              << txtFile.c_str();
  } else {
    LOG(ERROR) << "Open " << txtFile.c_str() << " result txt failed\n";
  }
};

BBox ScaleBboxToOriginal(const BBox& box) {
  BBox boxNew;
  float scale = std::min((float)yolov7InputSize / originHeight,
                         (float)yolov7InputSize / originWidth);
  scale = std::min(scale, 1.0f);
  float newW = originWidth * scale;
  float newH = originHeight * scale;
  float dx = (yolov7InputSize - newW) / 2;
  float dy = (yolov7InputSize - newH) / 2;

  boxNew.x1 =
      std::max(0.0f, std::min((box.x1 - dx) / scale, (float)originWidth));
  boxNew.y1 =
      std::max(0.0f, std::min((box.y1 - dy) / scale, (float)originHeight));
  boxNew.x2 =
      std::max(0.0f, std::min((box.x2 - dx) / scale, (float)originWidth));
  boxNew.y2 =
      std::max(0.0f, std::min((box.y2 - dy) / scale, (float)originHeight));
  boxNew.score = box.score;
  boxNew.classId = box.classId;

  return boxNew;
}

float CalculateIoU(const BBox& box1, const BBox& box2) {
  float interX1 = std::max(box1.x1, box2.x1);
  float interY1 = std::max(box1.y1, box2.y1);
  float interX2 = std::min(box1.x2, box2.x2);
  float interY2 = std::min(box1.y2, box2.y2);

  float interArea =
      std::max(0.0f, interX2 - interX1) * std::max(0.0f, interY2 - interY1);

  float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

  float unionArea = area1 + area2 - interArea;

  if (unionArea <= 0) return 0.0f;

  float ovr = interArea / unionArea;
  return ovr;
}

std::vector<BBox> NMS(std::vector<BBox>& boxes) {
  std::vector<BBox> keep;
  if (boxes.empty()) return keep;

  std::sort(boxes.begin(), boxes.end(),
            [](const BBox& a, const BBox& b) { return a.score > b.score; });
  while (boxes.size() > 0) {
    auto current = boxes.front();
    keep.push_back(ScaleBboxToOriginal(current));
    boxes.erase(boxes.begin());
    std::vector<int> toRemove;
    for (int i = 0; i < boxes.size(); i++) {
      if (boxes[i].classId == current.classId &&
          CalculateIoU(ScaleBboxToOriginal(boxes[i]),
                       ScaleBboxToOriginal(current)) > iouThres) {
        toRemove.push_back(i);
      }
    }

    for (int j = toRemove.size() - 1; j >= 0; j--) {
      boxes.erase(boxes.begin() + toRemove[j]);
    }
  }
  return keep;
}

std::vector<BBox> ProcessYOLOv7Output(std::vector<Infer::TensorBuf>& outBufs) {
  std::vector<BBox> allBboxs;

  if (outBufs.empty()) {
    LOG(ERROR) << "No output buffers";
    return allBboxs;
  }

  float* data = static_cast<float*>(outBufs[0].GetRawPtr());

  LOG(INFO) << "Processing  " << numBoxes << " boxes with " << boxAttrs
            << " attributes each";

  int validBoxes = 0;

  for (int i = 0; i < numBoxes; ++i) {
    float* boxData = data + i * boxAttrs;

    // 索引: [0]:xCenter, [1]:yCenter, [2]:width, [3]:height, [4]:objConf
    float xCenter = boxData[0];
    float yCenter = boxData[1];
    float width = boxData[2];
    float height = boxData[3];
    float objConf = boxData[4];

    if (objConf < confThres) {
      continue;
    }

    float maxClassCrob = 0.000000f;
    int classId = -1;

    for (int j = 0; j < numClasses; ++j) {
      float classProb = boxData[5 + j];
      if (classProb > maxClassCrob) {
        maxClassCrob = classProb;
        classId = j;
      }
    }

    float final_score = objConf * maxClassCrob;

    if (final_score >= confThres) {
      float x1 = xCenter - width / 2;
      float y1 = yCenter - height / 2;
      float x2 = xCenter + width / 2;
      float y2 = yCenter + height / 2;

      if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1) {
        allBboxs.push_back({x1, y1, x2, y2, final_score, classId});
        validBoxes++;
      }
    }
  }
  LOG(INFO) << "Found " << validBoxes
            << "valid boxes after confidence filtering";
  return allBboxs;
}

Result PostProcess(std::vector<Infer::TensorBuf>& outBufs,
                   const std::string& filePath) {
  size_t expected_size = numBoxes * boxAttrs * sizeof(float);
  if (outBufs[0].size != expected_size) {
    LOG(ERROR) << "Output size mismatch: expected " << expected_size << ", got "
               << outBufs[0].size;
    return FAILED;
  }

  LOG(INFO) << "Starting YOLOv7 postprocess for " << numBoxes << "boxes...";
  std::vector<BBox> allBboxs = ProcessYOLOv7Output(outBufs);
  LOG(INFO) << "Before NMS:" << allBboxs.size() << "boxes";
  std::vector<BBox> finalBboxs = NMS(allBboxs);
  LOG(INFO) << "After NMS:" << finalBboxs.size() << "boxes";
  SaveBboxResult(outBufs, finalBboxs, filePath);
  return SUCCESS;
}

bool GetYoloV7Box(std::vector<std::string>& fileList,
                  std::vector<TensorBuf>& tensorBufs,
                  std::vector<TensorDesc>& tensorDescs, bool isDpico) {
  if (isDpico) {
    LOG(INFO) << "detected dpico ";
  } else {
    LOG(INFO) << "detected dlite ";
  }
  if (tensorBufs.size() == 0 || fileList.size() == 0) {
    LOG(INFO) << " modelOuput size is 0 or inputFileList size is 0";
    return false;
  }
  if (isDpico) boxAttrs = dpicoBoxAttrs;
  cv::Mat img = cv::imread(fileList[0]);
  cv::Size imgSize = img.size();
  originWidth = imgSize.width;
  originHeight = imgSize.height;
  (void)PostProcess(tensorBufs, fileList[0]);
  LOG(INFO) << "Finished data postprocess successfully, " << fileList[0];
  return true;
}
}  // namespace Infer