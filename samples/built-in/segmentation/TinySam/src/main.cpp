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

#include "model.h"
#include "log.h"
#include "dev_interface_adapter.h"
#include "sam_predictor.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <variant>
#include <sys/stat.h>

using json = nlohmann::json;
using namespace cv;
using namespace std;
using namespace Infer;

static std::vector<json> readResFromJson(const std::string& jsonFilePath) {
    // 1. 打开JSON文件
    std::ifstream ifs(jsonFilePath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonFilePath);
    }

    // 2. 解析JSON内容
    json jsonData;
    try {
        ifs >> jsonData;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    // 3. 检查是否为数组格式
    if (!jsonData.is_array()) {
        throw std::runtime_error("JSON file is not a valid array format");
    }

    // 4. 转换为vector<json::json>
    std::vector<json> resList = jsonData.get<std::vector<json>>();
    return resList;
}

static std::vector<std::string> SplitString(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    // 按分隔符循环读取子串
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            result.push_back(item);
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        LOG(ERROR) << "【使用错误】参数数量不符";
        LOG(INFO) << "正确用法: " << argv[0] << " <json文件路径> <图片目录> <模型路径1;模型路径2;模型路径3>";
        return -1;
    }
    std::string jsonFilePath = argv[1];
    std::string imageDir = argv[2];

    // 解析分号分隔的模型路径参数
    std::vector<std::string> modelPaths = SplitString(argv[3], ':');
    // 验证模型路径数量是否为3个
    if (modelPaths.size() != 3) {
        LOG(ERROR) << "【使用错误】模型路径数量错误";
        LOG(INFO) << "需传入3个模型路径(用:分隔)，实际传入: " << modelPaths.size() << "个";
        return -1;
    }
    EnvInit();
    SAMBoxPredictor predictor;
    predictor.loadModel(modelPaths);
    int preImgId = -1;
    struct stat info;
    if (stat("result", &info) != 0) {
        mkdir("result", 0777);
    }
    std::vector<json> resList = readResFromJson(jsonFilePath);
    for (size_t i = 0; i < resList.size(); ++i) {
        const json& resIns = resList[i];
        int imgId = 0;
        if (resIns.contains("image_id") && resIns["image_id"].is_number_integer()) {
            imgId = resIns["image_id"].get<int>();
        } else {
            LOG(ERROR) << "Warning: Element " << i << " has no valid image_id, skip";
            continue;
        }
        char filenameBuf[20];
        sprintf(filenameBuf, "%012d.jpg", imgId);
        std::string imgFileName = filenameBuf;

        if (preImgId != imgId) {
            std::string fullImgPath = imageDir + "/" + imgFileName;
            Mat image = cv::imread(fullImgPath);
            if (image.empty()) {
                preImgId = imgId;  // 更新ID，避免重复报错
                continue;
            }
            cvtColor(image, image, COLOR_BGR2RGB);
            predictor.setImage(image);
            preImgId = imgId;
        }
        // 转换为float数组（bbox通常是x(左上)、y(左上)、宽、高）
        float x = resIns["bbox"][0].get<float>();
        float y = resIns["bbox"][1].get<float>();
        float width = resIns["bbox"][2].get<float>();
        float height = resIns["bbox"][3].get<float>();
        vector<float> boxes = {x, y, x + width, y + height};
        // 预测掩码
        pair<float, Mat> result = predictor.predict(boxes);
        float iou = result.first;
        LOG(INFO) << "bbox " << i << " process successfully";
    }
    LOG(INFO) << "completed";
    EnvDeinit();
    return 0;
}