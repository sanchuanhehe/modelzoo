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
#include <vector>
#include <string>

namespace Infer {
// 子命名空间：Yolo4
namespace Yolo4 {
// 声明结构体
struct BBox {
    float x1, y1, x2, y2, score;
    int classId;
    int cocoClassId;
};

// 声明核心后处理函数
bool Yolov4Postprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& outBufs, std::vector<TensorDesc>& outDescs);

}  // namespace Yolo4
}  // namespace Infer