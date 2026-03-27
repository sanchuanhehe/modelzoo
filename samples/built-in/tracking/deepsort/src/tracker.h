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

#ifndef TRACKER_H
#define TRACKER_H

#include "nn_matching.h"
#include "track.h"
#include <map>
#include <vector>

// 前置声明对外界暴露的单个追踪目标结构体
struct Track {
    int id;
    float x1, y1, width, height;
};

class Tracker {
public:
    Tracker(NearestNeighborDistanceMetric *metric, float maxIouDistance,
            int maxAge, int nInit);
    ~Tracker();

    void Predict();
    void Update(const std::vector<BoundingBox> &measurements,
                const std::vector<std::vector<float>> &features,
                const std::vector<int> &classIds);

    std::vector<Track> GetConfirmedTracks() const;

private:
    NearestNeighborDistanceMetric *metric_;
    float maxIouDistance_;
    int maxAge_;
    int nInit_;

    std::vector<TrackObj> tracks_;

    void LinearAssignment(const std::vector<std::vector<double>> &costMatrix,
                          float maxDistance,
                          const std::vector<int> &tracksCandidates,
                          const std::vector<int> &detectionsCandidates,
                          std::vector<std::pair<int, int>> &matches,
                          std::vector<int> &unmatchedTracks,
                          std::vector<int> &unmatchedDetections);

    void MatchFeatures(const std::vector<BoundingBox> &measurements,
                       const std::vector<std::vector<float>> &features,
                       const std::vector<int> &confirmedTracks,
                       std::vector<std::pair<int, int>> &matches,
                       std::vector<int> &unmatchedTracks,
                       std::vector<int> &unmatchedDetections);

    void MatchIoU(const std::vector<BoundingBox> &measurements,
                  const std::vector<int> &iouCandidates,
                  const std::vector<int> &unmatchedDetections,
                  std::vector<std::pair<int, int>> &matches,
                  std::vector<int> &finalUnmatchedTracks,
                  std::vector<int> &finalUnmatchedDetections);

    float CalculateIouDistance(const BoundingBox &a, const BoundingBox &b);
};

#endif // TRACKER_H
