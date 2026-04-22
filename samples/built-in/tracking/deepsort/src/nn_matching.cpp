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

#include "nn_matching.h"
#include <algorithm>
#include <cmath>
#include <numeric>

const float DISTANCE_INIT_MAX = 1e5f;  // 初始化最小距离搜索时的极大值

NearestNeighborDistanceMetric::NearestNeighborDistanceMetric(
    float maxCosineDistance, int nnBudget)
    : maxCosineDistance_(maxCosineDistance), nnBudget_(nnBudget) {}

float NearestNeighborDistanceMetric::CosineDistance(
    const std::vector<float> &u, const std::vector<float> &v) const
{
    float dot = 0.0f, normU = 0.0f, normV = 0.0f;
    for (size_t i = 0; i < u.size(); ++i) {
        dot += u[i] * v[i];
        normU += u[i] * u[i];
        normV += v[i] * v[i];
    }
    if (normU == 0 || normV == 0) {
        return 0.0f;
    }
    return 1.0f - dot / (std::sqrt(normU) * std::sqrt(normV));
}

std::vector<std::vector<float>> NearestNeighborDistanceMetric::Distance(
    const std::vector<std::vector<float>> &features,
    const std::vector<int> &targets)
{
    std::vector<std::vector<float>> costMatrix(
        targets.size(), std::vector<float>(features.size(), 0.0f));

    for (size_t i = 0; i < targets.size(); ++i) {
        int targetId = targets[i];
        auto &gallery = samples_[targetId];

        for (size_t j = 0; j < features.size(); ++j) {
            float minDist = DISTANCE_INIT_MAX;
            for (auto &trackFeat : gallery) {
                float dist = CosineDistance(features[j], trackFeat);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            costMatrix[i][j] = minDist;
        }
    }
    return costMatrix;
}

void NearestNeighborDistanceMetric::PartialFit(
    const std::vector<std::vector<float>> &features,
    const std::vector<int> &targets, const std::vector<int> &activeTargets)
{
    for (size_t i = 0; i < features.size(); ++i) {
        samples_[targets[i]].push_back(features[i]);
        if (nnBudget_ > 0 && samples_[targets[i]].size() > (size_t)nnBudget_) {
            samples_[targets[i]].erase(samples_[targets[i]].begin());
        }
    }

    // 清除不在活跃列表中的过期轨迹特征
    for (auto it = samples_.begin(); it != samples_.end();) {
        if (std::find(activeTargets.begin(), activeTargets.end(), it->first) ==
            activeTargets.end()) {
            it = samples_.erase(it);
        } else {
            ++it;
        }
    }
}

float MahalanobisDistance(const BoundingBox &state, const cv::Mat &covariance,
                          const BoundingBox &measure)
{
    cv::Mat d(4, 1, CV_32F);  // 4维观测空间残差向量
    d.at<float>(0, 0) = measure.x - state.x;
    d.at<float>(1, 0) = measure.y - state.y;
    d.at<float>(2, 0) = measure.a - state.a;
    d.at<float>(3, 0) = measure.h - state.h;

    cv::Mat covInv = covariance.inv();
    cv::Mat res = d.t() * covInv * d;
    return std::sqrt(res.at<float>(0, 0));
}
