#pragma once

#include <vector>
#include <string>

// 定义关键点结构，包含坐标、分数和描述子
struct KeyPoint {
    float x, y, score;
    std::vector<float> descriptor; // 256维特征向量
};

std::string ExtractFilename(const std::string &filePath);

// 返回 NCHW 顺序的 float 向量，尺寸 1×1×OUT_H×OUT_W
std::vector<float> SuperpointPreprocess(const std::string& imageFile);

std::vector<KeyPoint> ProcessKeypoints(const std::vector<float> &semi, const std::vector<float> &desc);

void VisualizeKeypoints(const std::vector<KeyPoint>& keypoints, const std::string& imagePath, 
                        const std::string& outputPath);

