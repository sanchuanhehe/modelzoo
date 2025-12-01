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

namespace Infer {
// 新增子命名空间：将YOLO4相关代码全部放入Infer::Yolo4
namespace Yolo4 {
bool Yolov4Preprocess(std::vector<std::string>& fileList, std::vector<TensorBuf>& inBufs, std::vector<TensorDesc>& inDescs);
}
}