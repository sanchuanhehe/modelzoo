/*
 * Copyright (c) ModelZoo. 2025-2025. All rights reserved.
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

#include "model.h"

namespace Infer {
struct ImageprocessOptions {
    ImageprocessOptions(int resize, int centerCrop, bool chw) : resizeValue(resize), centerCropValue(centerCrop), chwFlag(chw) {}
    int resizeValue;
    int centerCropValue;
    bool chwFlag = false;
    std::string dtypeStr = "uint8";
};
bool ImageProcess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs, const ImageprocessOptions& options);
}