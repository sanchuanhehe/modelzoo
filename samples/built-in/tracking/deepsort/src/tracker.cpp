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

#include "tracker.h"
#include "hungarian.h"
#include <algorithm>
#include <iostream>

const float GATING_THRESHOLD = 9.4877f;  // 卡方分布 4 自由度 95% 置信度门限值
const double GATING_COST = 1e5;          // 门控截断时赋予的极大惩罚代价，等效禁止匹配

Tracker::Tracker(NearestNeighborDistanceMetric *metric, float maxIouDistance,
                 int maxAge, int nInit)
    : metric_(metric), maxIouDistance_(maxIouDistance), maxAge_(maxAge),
      nInit_(nInit) {}

Tracker::~Tracker()
{
    if (metric_) {
        delete metric_;
    }
}

void Tracker::Predict()
{
    for (auto &track : tracks_) {
        track.Predict();
    }
}

float Tracker::CalculateIouDistance(const BoundingBox &a,
                                    const BoundingBox &b)
{
    float w1 = a.a * a.h, h1 = a.h;
    float x1 = a.x - w1 / 2.0f, y1 = a.y - h1 / 2.0f;
    float w2 = b.a * b.h, h2 = b.h;
    float x2 = b.x - w2 / 2.0f, y2 = b.y - h2 / 2.0f;

    float interW = std::max(0.0f, std::min(x1 + w1, x2 + w2) - std::max(x1, x2));
    float interH = std::max(0.0f, std::min(y1 + h1, y2 + h2) - std::max(y1, y2));

    float intersectionArea = interW * interH;
    float unionArea = w1 * h1 + w2 * h2 - intersectionArea;
    if (unionArea <= 0.0f) {
        return 1.0f;  // 无交集时返回最大距离
    }
    return 1.0f - (intersectionArea / unionArea);
}

// ==========================================
// 阶段一匹配器 (主匹配)：表观特征与运动状态的联合级联匹配
// ==========================================
void Tracker::MatchFeatures(const std::vector<BoundingBox> &measurements,
                            const std::vector<std::vector<float>> &features,
                            const std::vector<int> &confirmedTracks,
                            std::vector<std::pair<int, int>> &matches,
                            std::vector<int> &unmatchedTracks,
                            std::vector<int> &unmatchedDetections)
{
    if (confirmedTracks.empty() || measurements.empty()) {
        for (size_t d = 0; d < measurements.size(); ++d) {
            unmatchedDetections.push_back(d);
        }
        unmatchedTracks = confirmedTracks;
        return;
    }

    std::vector<int> trackIndices(confirmedTracks.size(), 0);
    for (size_t i = 0; i < confirmedTracks.size(); ++i) {
        trackIndices[i] = tracks_[confirmedTracks[i]].trackId_;
    }

    // 使用余弦距离构建特征表观代价矩阵 (Appearance Cost Matrix)
    std::vector<std::vector<float>> costMatrix =
        metric_->Distance(features, trackIndices);
    std::vector<std::vector<double>> costMatrixDouble(
        costMatrix.size(), std::vector<double>(measurements.size(), 0.0));

    for (size_t r = 0; r < costMatrix.size(); ++r) {
        for (size_t c = 0; c < costMatrix[0].size(); ++c) {
            // 通过卡尔曼滤波器预测的状态空间，计算马氏距离 (Mahalanobis Distance)
            float mDist = tracks_[confirmedTracks[r]].kf_.GetMahalanobisDistance(
                measurements[c]);
            // 当候选框严重偏离卡尔曼预测的轨迹期望分布时，视为不可信离群值进行强制截断阻断
            costMatrixDouble[r][c] = (mDist > GATING_THRESHOLD) ? GATING_COST : costMatrix[r][c];
        }
    }

    std::vector<int> detectionIndices(measurements.size());
    for (size_t d = 0; d < measurements.size(); ++d) {
        detectionIndices[d] = d;
    }

    LinearAssignment(costMatrixDouble, metric_->GetMaxCosineDistance(),
                     confirmedTracks, detectionIndices, matches, unmatchedTracks,
                     unmatchedDetections);
}

// ==========================================
// 阶段二匹配器 (接力匹配)：基于物理坐标系空间的 IoU 兜底匹配
// 主要针对第一阶段因为表观特征残缺、形变或被遮挡而漏掉的目标，进行基于物理惯性的强行"拉一把"关联
// ==========================================
void Tracker::MatchIoU(const std::vector<BoundingBox> &measurements,
                       const std::vector<int> &iouCandidates,
                       const std::vector<int> &unmatchedDetections,
                       std::vector<std::pair<int, int>> &matches,
                       std::vector<int> &finalUnmatchedTracks,
                       std::vector<int> &finalUnmatchedDetections)
{
    if (iouCandidates.empty() || unmatchedDetections.empty()) {
        finalUnmatchedDetections.insert(finalUnmatchedDetections.end(), unmatchedDetections.begin(), unmatchedDetections.end());
        finalUnmatchedTracks.insert(finalUnmatchedTracks.end(), iouCandidates.begin(), iouCandidates.end());
        return;
    }

    std::vector<std::vector<double>> iouCostMatrix(
        iouCandidates.size(),
        std::vector<double>(unmatchedDetections.size(), 0.0));

    for (size_t r = 0; r < iouCandidates.size(); ++r) {
        int tIdx = iouCandidates[r];
        for (size_t c = 0; c < unmatchedDetections.size(); ++c) {
            int dIdx = unmatchedDetections[c];
            // 计算边界框交并比 (IoU) 的重叠代价距离: cost = 1.0 - IoU
            float cost =
                CalculateIouDistance(tracks_[tIdx].mean_, measurements[dIdx]);
            iouCostMatrix[r][c] = (cost > maxIouDistance_) ? GATING_COST : cost;
        }
    }

    LinearAssignment(iouCostMatrix, maxIouDistance_, iouCandidates,
                     unmatchedDetections, matches, finalUnmatchedTracks,
                     finalUnmatchedDetections);
}

// ==========================================
// 全局生命周期调度入口
// 承担级联路由、新航迹注卷分离、僵尸航迹剪枝等管家任务
// ==========================================
void Tracker::Update(const std::vector<BoundingBox> &measurements,
                     const std::vector<std::vector<float>> &features,
                     const std::vector<int> &classIds)
{
    std::vector<int> confirmedTracks;
    std::vector<int> unconfirmedTracks;

    // 1. 分流处理：将久经考验的已确认目标与新兵蛋子暂定目标分类处置
    for (size_t i = 0; i < tracks_.size(); i++) {
        if (tracks_[i].IsConfirmed()) {
            confirmedTracks.push_back(i);
        } else {
            unconfirmedTracks.push_back(i);
        }
    }

    std::vector<std::pair<int, int>> matches;
    std::vector<int> unmatchedTracksA;
    std::vector<int> unmatchedDetections;

    // --- 第一级联阶段：特征深度学习主匹配 (严苛匹配) ---
    MatchFeatures(measurements, features, confirmedTracks, matches,
                  unmatchedTracksA, unmatchedDetections);

    // --- 第二备用阶段：IoU 空间相交补漏兜底 (宽容接力匹配) ---
    // 收集候选：尚未确认的 Tentative 航迹 + 第一阶段仅失联 1 帧的航迹
    std::vector<int> iouCandidates = unconfirmedTracks;
    for (int tIdx : unmatchedTracksA) {
        if (tracks_[tIdx].timeSinceUpdate_ == 1) {
            iouCandidates.push_back(tIdx);
        }
    }

    std::vector<int> finalUnmatchedTracks;
    for (int tIdx : unmatchedTracksA) {
        // 如果丢了超过两帧，不用 IoU 匹配，彻底被剥离
        if (tracks_[tIdx].timeSinceUpdate_ != 1) {
            finalUnmatchedTracks.push_back(tIdx);
        }
    }

    std::vector<int> finalUnmatchedDetections;
    MatchIoU(measurements, iouCandidates, unmatchedDetections, matches,
             finalUnmatchedTracks, finalUnmatchedDetections);

    // 4. 更新最终通过匈牙利角斗胜出匹配的大名单状态
    for (auto &match : matches) {
        tracks_[match.first].Update(measurements[match.second],
                                    features[match.second], classIds[match.second]);
    }

    // 5. 对无任何匹配接壤的新生观测单独实施建档初始化 (新进入探测视域)
    for (int dIdx : finalUnmatchedDetections) {
        TrackObj newTrack(measurements[dIdx], features[dIdx], classIds[dIdx],
                          maxAge_, nInit_);
        tracks_.push_back(newTrack);
    }

    // 6. 冷板凳标记丢失并集中销毁所有已超过寿命期限的僵尸目标
    for (int tIdx : finalUnmatchedTracks) {
        tracks_[tIdx].MarkMissed();
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const TrackObj &t) { return t.IsDeleted(); }),
                  tracks_.end());

    // 7. 在相似度滑窗存储器中提交局部增量特征，保持最新的记忆样貌模板
    std::vector<int> activeTargets;
    std::vector<std::vector<float>> activeFeats;
    std::vector<int> tgtIds;
    for (auto &t : tracks_) {
        activeTargets.push_back(t.trackId_);
        activeFeats.push_back(t.features_);
        tgtIds.push_back(t.trackId_);
    }
    metric_->PartialFit(activeFeats, tgtIds, activeTargets);
}

std::vector<Track> Tracker::GetConfirmedTracks() const
{
    std::vector<Track> out;
    for (const auto &t : tracks_) {
        if (t.IsConfirmed() && t.timeSinceUpdate_ <= 1) {
            Track tOut;
            tOut.id = t.trackId_;

            // 将中心宽高格式逆向推导转换为左上点坐标 (即左上角x1/y1和宽/高格式)
            tOut.x1 = t.mean_.x - (t.mean_.a * t.mean_.h) / 2.0f;
            tOut.y1 = t.mean_.y - t.mean_.h / 2.0f;
            tOut.width = t.mean_.a * t.mean_.h;
            tOut.height = t.mean_.h;

            out.push_back(tOut);
        }
    }
    return out;
}

// --- 封装：将匈牙利解与合规判定归纳到 LinearAssignment 模型中 ---
void Tracker::LinearAssignment(
    const std::vector<std::vector<double>> &costMatrix, float maxDistance,
    const std::vector<int> &tracksCandidates,
    const std::vector<int> &detectionsCandidates,
    std::vector<std::pair<int, int>> &matches,
    std::vector<int> &unmatchedTracks, std::vector<int> &unmatchedDetections)
{
    if (tracksCandidates.empty() || detectionsCandidates.empty()) {
        unmatchedTracks.insert(unmatchedTracks.end(), tracksCandidates.begin(),
                               tracksCandidates.end());
        unmatchedDetections.insert(unmatchedDetections.end(),
                                   detectionsCandidates.begin(),
                                   detectionsCandidates.end());
        return;
    }

    // 拷贝代价矩阵，因为匈牙利算法 Solve 可能修改原矩阵
    std::vector<std::vector<double>> costMatrixCopy = costMatrix;
    HungarianAlgorithm hungAlgo;
    std::vector<int> assignment;
    hungAlgo.Solve(costMatrixCopy, assignment);

    for (size_t d = 0; d < detectionsCandidates.size(); ++d) {
        bool matched = false;
        for (size_t t = 0; t < assignment.size(); ++t) {
            if (assignment[t] == (int)d && costMatrix[t][d] <= maxDistance) {
                matches.push_back({tracksCandidates[t], detectionsCandidates[d]});
                matched = true;
                break;
            }
        }
        if (!matched) {
            unmatchedDetections.push_back(detectionsCandidates[d]);
        }
    }

    for (size_t t = 0; t < tracksCandidates.size(); ++t) {
        bool matched = false;
        for (size_t d = 0; d < assignment.size(); ++d) {
            if (assignment[t] == (int)d && costMatrix[t][d] <= maxDistance) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            unmatchedTracks.push_back(tracksCandidates[t]);
        }
    }
}
