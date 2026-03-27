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

#include "deepsort.h"
#include <algorithm>
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

const int MIN_EXT_LEN = 4;    // 最短文件扩展名长度 (.jpg/.png/.bmp)
const int JPEG_EXT_LEN = 5;   // .jpeg 扩展名长度

// 从目录中获取所有图片文件路径并排序
static std::vector<std::string> GetImageFiles(const std::string &dirPath)
{
    std::vector<std::string> files;
    DIR *dir = opendir(dirPath.c_str());
    if (!dir) {
        return files;
    }
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name.size() > MIN_EXT_LEN) {
            std::string ext = name.substr(name.size() - MIN_EXT_LEN);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".png" || ext == ".bmp") {
                files.push_back(dirPath + "/" + name);
            }
        }
        if (name.size() > JPEG_EXT_LEN) {
            std::string ext = name.substr(name.size() - JPEG_EXT_LEN);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpeg") {
                files.push_back(dirPath + "/" + name);
            }
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end());
    return files;
}

// 从外置 JSON 文件中加载用户自定义的超参数配置
static void LoadConfigFromJson(const std::string &configPath,
                               DeepSORTConfig &config)
{
    std::ifstream f(configPath);
    if (!f.is_open()) {
        std::cerr << "Could not open config file " << configPath
                  << ". Using defaults." << std::endl;
        return;
    }
    json data = json::parse(f);
    if (data.contains("YOLO_CONF_THRES")) {
        config.yoloConfThres = data["YOLO_CONF_THRES"];
    }
    if (data.contains("YOLO_NMS_THRES")) {
        config.yoloNmsThres = data["YOLO_NMS_THRES"];
    }
    if (data.contains("MAX_COSINE_DISTANCE")) {
        config.maxCosineDistance = data["MAX_COSINE_DISTANCE"];
    }
    if (data.contains("NN_BUDGET")) {
        config.nnBudget = data["NN_BUDGET"];
    }
    if (data.contains("MAX_IOU_DISTANCE")) {
        config.maxIouDistance = data["MAX_IOU_DISTANCE"];
    }
    if (data.contains("MAX_AGE")) {
        config.maxAge = data["MAX_AGE"];
    }
    if (data.contains("N_INIT")) {
        config.nInit = data["N_INIT"];
    }
    if (data.contains("TEST_REID_FPS")) {
        config.testReidFps = data["TEST_REID_FPS"];
    }
    std::cout << "Loaded custom config from " << configPath << std::endl;
}

// 根据输入路径动态生成 MOT 评测结果的输出文件名
static std::string BuildOutputFilename(const std::string &inputPath)
{
    std::string normPath = inputPath;
    if (!normPath.empty() && normPath.back() == '/') {
        normPath.pop_back();
    }

    size_t pos = normPath.find_last_of('/');
    if (pos == std::string::npos) {
        return normPath + "-npu.txt";
    }

    std::string folderName = normPath.substr(pos + 1);
    if (folderName == "img1" || folderName == "img") {
        size_t prevPos = normPath.find_last_of('/', pos - 1);
        if (prevPos != std::string::npos) {
            return normPath.substr(prevPos + 1, pos - prevPos - 1) + "-npu.txt";
        }
        return normPath.substr(0, pos) + "-npu.txt";
    }
    return folderName + "-npu.txt";
}

// 判断输入路径是文件还是目录，获取待处理帧路径列表
static std::vector<std::string> GetFramePaths(const std::string &inputPath)
{
    std::vector<std::string> framePaths;
    DIR *dir = opendir(inputPath.c_str());
    if (dir) {
        closedir(dir);
        framePaths = GetImageFiles(inputPath);
    } else {
        framePaths.push_back(inputPath);
    }
    return framePaths;
}

// 核心推理循环：逐帧执行 DeepSort 跟踪并输出 MOT 评测结果
static int RunTrackingPipeline(DeepSortController &deepSort,
                               const std::vector<std::string> &framePaths,
                               const std::string &outFilename)
{
    std::ofstream outTxt(outFilename);
    if (!outTxt.is_open()) {
        std::cerr << "Warning: Could not open " << outFilename
                  << " for writing." << std::endl;
    }

    std::cout << "Total frames to process: " << framePaths.size() << std::endl;

    for (size_t frameIdx = 0; frameIdx < framePaths.size(); ++frameIdx) {
        cv::Mat frame = cv::imread(framePaths[frameIdx]);
        if (frame.empty()) {
            std::cerr << "Failed to read image: " << framePaths[frameIdx]
                      << std::endl;
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();
        deepSort.ProcessFrame(frame);
        auto end = std::chrono::high_resolution_clock::now();
        float ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        // 获取当前帧确认的追踪目标并写入 MOT 评测文件
        auto confirmedTracks = deepSort.tracker_->GetConfirmedTracks();
        for (const auto &t : confirmedTracks) {
            // MOT16 格式: <frame_id>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1
            if (outTxt.is_open()) {
                outTxt << (frameIdx + 1) << "," << t.id << ","
                       << t.x1 << "," << t.y1 << ","
                       << t.width << "," << t.height << ",1,-1,-1,-1\n";
            }
            cv::Rect rect(t.x1, t.y1, t.width, t.height);
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID " + std::to_string(t.id);
            cv::putText(frame, label, cv::Point(t.x1, t.y1 - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }

        std::cout << "Frame " << frameIdx + 1
                  << " (" << framePaths[frameIdx] << ") processed in " << ms
                  << " ms. Active tracks: " << confirmedTracks.size()
                  << std::endl;
    }

    std::cout << "Successfully saved tracking results to: " << outFilename
              << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    const int minArgCount = 4;  // 最少需要: 程序名 + yolo模型 + resnet模型 + 输入路径
    if (argc < minArgCount) {
        std::cerr << "Usage: ./main <yolov5.om> <resnet18.om> "
                     "<image_dir_or_file> [config.json]" << std::endl;
        return -1;
    }

    // 初始化 NPU 驱动设备
    if (Infer::DevInit("") != Infer::SUCCESS) {
        std::cerr << "Device Init Failed." << std::endl;
        return -1;
    }

    // 构建默认超参数配置
    DeepSORTConfig config;
    config.yoloModelPath = argv[1];
    config.resnetModelPath = argv[2];
    config.yoloConfThres = 0.3f;    // YOLO 目标置信度阈值
    config.yoloNmsThres = 0.4f;     // YOLO NMS 重叠阈值
    config.maxCosineDistance = 0.15f;  // ReID 余弦距离最大匹配阈值
    config.nnBudget = 100;          // 每个 ID 最多缓存的特征向量数量
    config.testReidFps = false;

    // 如果提供了外置 JSON 配置文件则解析覆盖默认值
    const int configArgIdx = 4;  // config.json 在 argv 中的索引
    std::string configPath = (argc > configArgIdx) ? argv[configArgIdx] : "";
    if (!configPath.empty()) {
        LoadConfigFromJson(configPath, config);
    }

    std::string inputPath = argv[3];
    std::string outFilename = BuildOutputFilename(inputPath);

    // 局部作用域确保 deepSort 在 DevDeInit 之前析构
    {
        DeepSortController deepSort(config);
        if (deepSort.Init() != Infer::SUCCESS) {
            std::cerr << "DeepSort Init Failed." << std::endl;
            Infer::DevDeInit();
            return -1;
        }

        std::vector<std::string> framePaths = GetFramePaths(inputPath);
        if (framePaths.empty()) {
            std::cerr << "No image files found in: " << inputPath << std::endl;
            Infer::DevDeInit();
            return -1;
        }

        RunTrackingPipeline(deepSort, framePaths, outFilename);
    }

    Infer::DevDeInit();
    return 0;
}
