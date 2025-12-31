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

#include <string>
#include <unordered_map>
#include "dev_interface_adapter.h"

namespace Infer {
struct InferParam {
    std::string omModelPath;
    std::string aclConfigPath;
    std::string imglistPath;
};


bool PathToRealPath(const std::string &path, std::string &realPath);
bool ParseParamFromCmd(int argc, char *argv[], InferParam &inferParam);
bool PadDataToTensorBuf(void* data, size_t size, TensorDesc& desc, TensorBuf& buf);
std::unordered_map<std::string, std::string> ReadCfgFile(const std::string& cfgPath);
bool CreateDirectoryRecursive(const std::string& path);
std::vector<std::vector<std::string>> ParseFileList(const std::string& fileListPaths);
std::string BuildInputString(std::vector<std::string>& fileListPaths);
std::vector<std::string> GetInputList(std::string inputString);
uint16_t FloatToHalf(float value);
float HalfToFloat(uint16_t value);
}