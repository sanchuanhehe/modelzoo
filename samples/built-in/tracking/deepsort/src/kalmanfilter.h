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

#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <opencv2/core.hpp>
#include <vector>

struct BoundingBox {
    float x; // 中心坐标 x
    float y; // 中心坐标 y
    float a; // 纵横比 (宽/高)
    float h; // 高度
};

class KalmanFilter {
public:
    KalmanFilter();

    void Initiate(const BoundingBox &measurement);
    void Predict();
    void Update(const BoundingBox &measurement);

    BoundingBox GetState() const;
    BoundingBox GetPredictState() const {
        BoundingBox box;
        box.x = statePre_.at<float>(0, 0);
        box.y = statePre_.at<float>(1, 0);
        box.a = statePre_.at<float>(2, 0);
        box.h = statePre_.at<float>(3, 0);
        return box;
    }
    cv::Mat GetStateMat() const { return statePost_; }
    cv::Mat GetCovarianceMat() const { return errorCovPost_; }
    float GetMahalanobisDistance(const BoundingBox &measurement) const;

private:
    cv::Mat transitionMatrix_;    // F: 8x8
    cv::Mat measurementMatrix_;   // H: 4x8
    cv::Mat processNoiseCov_;     // Q: 8x8
    cv::Mat measurementNoiseCov_; // R: 4x4
    cv::Mat statePost_;           // x: 8x1 (后验状态)
    cv::Mat statePre_;            // x': 8x1 (先验状态)
    cv::Mat errorCovPost_;        // P: 8x8 (后验协方差)
    cv::Mat errorCovPre_;         // P': 8x8 (先验协方差)

    float dt_;
    float stdWeightPosition_;
    float stdWeightVelocity_;
};

#endif // KALMANFILTER_H
