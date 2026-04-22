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

#ifndef NN_MATCHING_H
#define NN_MATCHING_H

#include "track.h"
#include <map>
#include <opencv2/core.hpp>
#include <vector>

class NearestNeighborDistanceMetric {
public:
    NearestNeighborDistanceMetric(float maxCosineDistance, int nnBudget);

    std::vector<std::vector<float>>
    Distance(const std::vector<std::vector<float>> &features,
             const std::vector<int> &targets);

    void PartialFit(const std::vector<std::vector<float>> &features,
                    const std::vector<int> &targets,
                    const std::vector<int> &activeTargets);

    float GetMaxCosineDistance() const { return maxCosineDistance_; }

private:
    float maxCosineDistance_;
    int nnBudget_;
    std::map<int, std::vector<std::vector<float>>> samples_;

    float CosineDistance(const std::vector<float> &u,
                         const std::vector<float> &v) const;
};

// 外部声明，用于辅助计算特征协方差边界的马氏距离函数
float MahalanobisDistance(const BoundingBox &state, const cv::Mat &covariance,
                          const BoundingBox &measure);

#endif // NN_MATCHING_H
