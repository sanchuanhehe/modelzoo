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

#ifndef TRACK_H
#define TRACK_H

#include "kalmanfilter.h"
#include <vector>

enum TrackState { Tentative, Confirmed, Deleted };

class TrackObj {
public:
    TrackObj(const BoundingBox &means, const std::vector<float> &feature,
             int classId, int maxAge, int nInit);

    void Update(const BoundingBox &measurement, const std::vector<float> &feature,
                int classId);
    void Predict();
    void MarkMissed();
    bool IsConfirmed() const;
    bool IsDeleted() const;
    bool IsTentative() const;

    int trackId_;
    int classId_;
    int hits_;
    int maxAge_;
    int nInit_;
    int timeSinceUpdate_;
    TrackState state_;

    std::vector<float> features_;
    BoundingBox mean_;
    KalmanFilter kf_;

private:
    static int nextId_;
};

#endif // TRACK_H
