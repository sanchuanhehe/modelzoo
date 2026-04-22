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

#pragma once

#include "dev_interface_adapter.h"
#include <vector>

struct DetectBox {
    int classID;
    float confidence;
    float x1, y1, x2, y2;
    float width() const { return std::max(0.0f, x2 - x1); }
    float height() const { return std::max(0.0f, y2 - y1); }
};

namespace Infer {
namespace DeepSort {

struct YoloImageInfo {
    int origWidth;
    int origHeight;
    float confThres;
    float nmsThres;
};

void NMS(std::vector<DetectBox> &inputBoxes, float nmsThresh);
int PostProcessYOLO(const std::vector<TensorBuf> &outBufs,
                    const std::vector<TensorDesc> &outDescs,
                    std::vector<DetectBox> &detectedBoxes,
                    const YoloImageInfo &imgInfo);

} // namespace DeepSort
} // namespace Infer
