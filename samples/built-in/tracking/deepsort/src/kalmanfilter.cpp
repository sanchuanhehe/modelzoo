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

#include "kalmanfilter.h"
#include <cmath>

const int STATE_DIM = 8;        // 卡尔曼状态向量维度: (x, y, a, h, vx, vy, va, vh)
const int MEASURE_DIM = 4;      // 卡尔曼观测向量维度: (x, y, a, h)

KalmanFilter::KalmanFilter()
{
    dt_ = 1.0f;
    stdWeightPosition_ = 1.0f / 20;   // 位置标准差权重，参照 Python 原版 deep_sort
    stdWeightVelocity_ = 1.0f / 160;  // 速度标准差权重，参照 Python 原版 deep_sort

    // 状态转移矩阵 F (8x8)
    transitionMatrix_ = (cv::Mat_<float>(STATE_DIM, STATE_DIM) <<
                         1, 0, 0, 0, dt_, 0, 0, 0,
                         0, 1, 0, 0, 0, dt_, 0, 0,
                         0, 0, 1, 0, 0, 0, dt_, 0,
                         0, 0, 0, 1, 0, 0, 0, dt_,
                         0, 0, 0, 0, 1, 0, 0, 0,
                         0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 0, 0, 1, 0,
                         0, 0, 0, 0, 0, 0, 0, 1);

    // 观测矩阵 H (4x8)
    measurementMatrix_ =
        (cv::Mat_<float>(MEASURE_DIM, STATE_DIM) <<
         1, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0);

    statePost_ = cv::Mat::zeros(STATE_DIM, 1, CV_32F);
    statePre_ = cv::Mat::zeros(STATE_DIM, 1, CV_32F);
    errorCovPost_ = cv::Mat::eye(STATE_DIM, STATE_DIM, CV_32F);
    errorCovPre_ = cv::Mat::eye(STATE_DIM, STATE_DIM, CV_32F);
    processNoiseCov_ = cv::Mat::eye(STATE_DIM, STATE_DIM, CV_32F);
    measurementNoiseCov_ = cv::Mat::eye(MEASURE_DIM, MEASURE_DIM, CV_32F);
}

void KalmanFilter::Initiate(const BoundingBox &measurement)
{
    cv::Mat state = cv::Mat::zeros(STATE_DIM, 1, CV_32F);
    state.at<float>(0, 0) = measurement.x;
    state.at<float>(1, 0) = measurement.y;
    state.at<float>(2, 0) = measurement.a;
    state.at<float>(3, 0) = measurement.h;
    // 速度项初始化为0 (已默认)

    statePost_ = state;

    // 2x: 初始位置不确定性放大因子，补偿首帧无历史观测的信息缺失
    float stdPosition[MEASURE_DIM] = {
        2 * stdWeightPosition_ * measurement.h,
        2 * stdWeightPosition_ * measurement.h,
        1e-2f,  // 纵横比 aspect ratio 的初始方差（经验值）
        2 * stdWeightPosition_ * measurement.h};
    // 10x: 初始速度不确定性放大因子，首帧无运动历史故需要更大容差
    float stdVelocity[MEASURE_DIM] = {
        10 * stdWeightVelocity_ * measurement.h,
        10 * stdWeightVelocity_ * measurement.h,
        1e-5f,  // 纵横比变化率的初始方差（经验值）
        10 * stdWeightVelocity_ * measurement.h};

    cv::Mat covariance = cv::Mat::eye(STATE_DIM, STATE_DIM, CV_32F);
    for (int i = 0; i < MEASURE_DIM; ++i) {
        covariance.at<float>(i, i) = stdPosition[i] * stdPosition[i];
        covariance.at<float>(i + MEASURE_DIM, i + MEASURE_DIM) = stdVelocity[i] * stdVelocity[i];
    }
    errorCovPost_ = covariance;
}

void KalmanFilter::Predict()
{
    // 动态过程噪声矩阵建模
    float h = statePost_.at<float>(3, 0);
    // 2x: 抗 INT8 量化位置抖动放大因子
    float stdPosition[MEASURE_DIM] = {
        2 * stdWeightPosition_ * h,
        2 * stdWeightPosition_ * h,
        1e-2f,  // 纵横比过程噪声（经验值）
        2 * stdWeightPosition_ * h};
    // 10x: 抗 INT8 量化速度抖动放大因子
    float stdVelocity[MEASURE_DIM] = {
        10 * stdWeightVelocity_ * h,
        10 * stdWeightVelocity_ * h,
        1e-5f,  // 纵横比变化率过程噪声（经验值）
        10 * stdWeightVelocity_ * h};

    cv::Mat Q = cv::Mat::eye(STATE_DIM, STATE_DIM, CV_32F);
    for (int i = 0; i < MEASURE_DIM; ++i) {
        Q.at<float>(i, i) = stdPosition[i] * stdPosition[i];
        Q.at<float>(i + MEASURE_DIM, i + MEASURE_DIM) = stdVelocity[i] * stdVelocity[i];
    }
    processNoiseCov_ = Q;

    // 预测: x' = F * x
    statePre_ = transitionMatrix_ * statePost_;
    // 预测协方差: P' = F * P * F^T + Q
    errorCovPre_ = transitionMatrix_ * errorCovPost_ * transitionMatrix_.t() +
                   processNoiseCov_;

    // DeepSORT 标准对齐:
    // 在未得到外部检测更新(Update)前，将预测先验值暂存在后验变量中以支持纯空转滑推 (Coast)
    statePost_ = statePre_.clone();
    errorCovPost_ = errorCovPre_.clone();
}

void KalmanFilter::Update(const BoundingBox &measurement)
{
    cv::Mat measureMat(MEASURE_DIM, 1, CV_32F);
    measureMat.at<float>(0, 0) = measurement.x;
    measureMat.at<float>(1, 0) = measurement.y;
    measureMat.at<float>(2, 0) = measurement.a;
    measureMat.at<float>(3, 0) = measurement.h;

    float h = statePre_.at<float>(3, 0);
    float stdPosition[MEASURE_DIM] = {
        stdWeightPosition_ * h, stdWeightPosition_ * h,
        1e-1f,  // 纵横比观测噪声（经验值）
        stdWeightPosition_ * h};

    cv::Mat R = cv::Mat::eye(MEASURE_DIM, MEASURE_DIM, CV_32F);
    for (int i = 0; i < MEASURE_DIM; ++i) {
        R.at<float>(i, i) = stdPosition[i] * stdPosition[i];
    }
    measurementNoiseCov_ = R;

    // 新息 (残差): y = z - H * x'
    cv::Mat innovation = measureMat - measurementMatrix_ * statePre_;
    // 新息协方差: S = H * P' * H^T + R
    cv::Mat S = measurementMatrix_ * errorCovPre_ * measurementMatrix_.t() +
                measurementNoiseCov_;
    // 卡尔曼增益: K = P' * H^T * S^{-1}
    cv::Mat K = errorCovPre_ * measurementMatrix_.t() * S.inv();

    // 更新状态: x = x' + K * y
    statePost_ = statePre_ + K * innovation;
    // 更新协方差: P = (I - K * H) * P'
    cv::Mat I = cv::Mat::eye(STATE_DIM, STATE_DIM, CV_32F);
    errorCovPost_ = (I - K * measurementMatrix_) * errorCovPre_;
}

BoundingBox KalmanFilter::GetState() const
{
    BoundingBox box;
    box.x = statePost_.at<float>(0, 0);
    box.y = statePost_.at<float>(1, 0);
    box.a = statePost_.at<float>(2, 0);
    box.h = statePost_.at<float>(3, 0);
    return box;
}

float KalmanFilter::GetMahalanobisDistance(
    const BoundingBox &measurement) const
{
    cv::Mat measureMat(MEASURE_DIM, 1, CV_32F);
    measureMat.at<float>(0, 0) = measurement.x;
    measureMat.at<float>(1, 0) = measurement.y;
    measureMat.at<float>(2, 0) = measurement.a;
    measureMat.at<float>(3, 0) = measurement.h;

    cv::Mat mean = measurementMatrix_ * statePre_;

    float h = statePre_.at<float>(3, 0);
    float stdPosition[MEASURE_DIM] = {
        stdWeightPosition_ * h, stdWeightPosition_ * h,
        1e-1f,  // 纵横比观测噪声（经验值）
        stdWeightPosition_ * h};

    cv::Mat R = cv::Mat::eye(MEASURE_DIM, MEASURE_DIM, CV_32F);
    for (int i = 0; i < MEASURE_DIM; ++i) {
        R.at<float>(i, i) = stdPosition[i] * stdPosition[i];
    }

    cv::Mat S = measurementMatrix_ * errorCovPre_ * measurementMatrix_.t() + R;
    cv::Mat innovation = measureMat - mean;

    cv::Mat res = innovation.t() * S.inv() * innovation;
    // deep_sort gating 使用平方马氏距离
    // 4 自由度卡方分布在 0.05 显著性水平的临界值为 9.4877
    return res.at<float>(0, 0);
}
