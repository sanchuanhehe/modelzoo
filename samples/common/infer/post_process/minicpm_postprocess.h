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
#include "log.h"
#include <string>
#include <unordered_map>
#include <vector>

int PrefillPostprocess(const std::vector<Infer::TensorBuf>& outBuf, const std::vector<Infer::TensorDesc>& outDesc, int prefillLen, int& prefillTokenId);

int DecodePostprocess(const std::vector<Infer::TensorBuf>& outBuf, const std::vector<Infer::TensorDesc>& outDesc, int& tokenId);

int DecodeTokenIds(const std::vector<int>& tokenIdVec, std::string& result, std::unordered_map<int, std::string>& idToToken);