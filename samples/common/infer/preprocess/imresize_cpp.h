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
#include "image_process.h"
#include "log.h"
#include "dev_interface_adapter.h"
#include <opencv2/opencv.hpp>
#include "PillowResize/PillowResize.hpp"
#include <string>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <fstream>
#include "utils.h"

namespace Infer {
    cv::Mat MatlabImresize(const cv::Mat& mat, const int& r);
}