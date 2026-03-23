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

#include "dev_interface_adapter.h"
#include "PillowResize/PillowResize.hpp"

namespace Infer {
struct VitPrerocessOptions {
    VitPrerocessOptions(int resize, int centerCrop, bool chw, int32_t interpolation = PillowResize::INTERPOLATION_BILINEAR) : resizeValue(resize), centerCropValue(centerCrop), chwFlag(chw), interpolationValue(interpolation) {}
    int resizeValue;
    int centerCropValue;
    bool chwFlag = false;
    std::string dtypeStr = "uint8";
    int32_t interpolationValue = PillowResize::INTERPOLATION_BILINEAR;
};
bool VitPreprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs, const VitPrerocessOptions& options);
}