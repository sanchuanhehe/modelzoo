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

#include "track.h"

int TrackObj::nextId_ = 1;  // 全局自增 ID 计数器

TrackObj::TrackObj(const BoundingBox &means, const std::vector<float> &feature,
                   int classId, int maxAge, int nInit)
    : trackId_(nextId_++), classId_(classId), hits_(1), maxAge_(maxAge),
      nInit_(nInit), timeSinceUpdate_(0), state_(Tentative), mean_(means)
{
    features_ = feature;
    kf_.Initiate(means);
    mean_ = kf_.GetState();
}

void TrackObj::Update(const BoundingBox &measurement,
                      const std::vector<float> &feature, int classId)
{
    kf_.Update(measurement);
    mean_ = kf_.GetState(); // 由卡尔曼滤波器计算出的最新后验状态覆盖
    features_ = feature;
    classId_ = classId;
    hits_++;
    timeSinceUpdate_ = 0;

    if (state_ == Tentative && hits_ >= nInit_) {
        state_ = Confirmed;
    }
}

void TrackObj::Predict()
{
    kf_.Predict();
    mean_ = kf_.GetPredictState();
    timeSinceUpdate_++;
}

void TrackObj::MarkMissed()
{
    if (state_ == Tentative) {
        state_ = Deleted;
    } else if (timeSinceUpdate_ > maxAge_) {
        state_ = Deleted;
    }
}

bool TrackObj::IsConfirmed() const { return state_ == Confirmed; }
bool TrackObj::IsDeleted() const { return state_ == Deleted; }
bool TrackObj::IsTentative() const { return state_ == Tentative; }
