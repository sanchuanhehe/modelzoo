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
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <thread>
#include "acl.h"

namespace Infer {
class AclMdl : public MdlBase {
public:
    virtual ~AclMdl() = default;
    int32_t LoadModel(const std::string& modelPath) override;
    int32_t LoadModel(const char* modelBuf, size_t modelSize) override;
    int32_t UnLoadModel() override;

    size_t GetInTensorNum() override;
    int32_t GetInTensorDescByIdx(size_t index, TensorDesc& desc) override;
    int32_t GetInTensorDescByName(const std::string& name, TensorDesc& tensorDesc) override;
    size_t GetOutTensorNum() override;
    int32_t GetOutTensorDescByIdx(size_t index, TensorDesc& desc) override;
    int32_t GetOutTensorDescByName(const std::string& name, TensorDesc& tensorDesc) override;

    int32_t Execute(std::vector<TensorBuf>& inBufs, std::vector<TensorBuf>& outBufs,
        RunMode runMode = RunMode::Sync) override;

    int32_t Wait() override;

private:
    int32_t UpdateDatasetBuffer(size_t index, TensorBuf& tensorBuf, aclmdlDataset* dataset);
    int32_t CreateModelDesc();
    int32_t InitInput();
    int32_t InitOutput();
    void DestroyDataset(aclmdlDataset* dataset);
    bool loadFlag_ = false;
    std::mutex loadFlagMtx_;
    std::mutex executeMtx_;
    uint32_t modelId_ = 0;
    aclmdlDesc* modelDesc_ = nullptr;
    aclmdlDataset* input_ = nullptr;
    aclmdlDataset* output_ = nullptr;
    TensorBuf defaultBuf_ = TensorBuf(1, 0);
};
}