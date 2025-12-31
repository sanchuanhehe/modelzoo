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
#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include <mutex>

namespace Infer {
class SvpAclMdl : public MdlBase{
public:
    virtual ~SvpAclMdl() = default;
    // debug模式时打印 模型TensorDesc
    int32_t LoadModel(const std::string& modelPath) override;
    int32_t LoadModel(const char* modelBuf, size_t modelSize) override;
    int32_t UnLoadModel() override;

    size_t GetInTensorNum() override;
    int32_t GetInTensorDescByIdx(size_t index, TensorDesc& desc) override;
    int32_t GetInTensorDescByName(const std::string& name, TensorDesc& desc) override;
    size_t GetOutTensorNum() override;
    int32_t GetOutTensorDescByIdx(size_t index, TensorDesc& desc) override;
    int32_t GetOutTensorDescByName(const std::string& name, TensorDesc& desc) override;

    int32_t Execute(std::vector<TensorBuf> &inBufs, std::vector<TensorBuf> &outBufs,
        RunMode runMode = RunMode::Sync);
    int32_t Wait() override;

private:
    int32_t ReadFile(const std::string& filePath, void *&ptr, uint32_t &size);
    int32_t LoadModelFromMem(const void* modelBuf, size_t modelSize);
    int32_t CreateDesc();
    void DestroyDesc();
    int32_t CreateDataset();
    void DestroyDataset();
    int32_t GetInOutputNum();
    int32_t CreateWorkAndTaskBuf();
    int32_t SetInBuf(size_t index, TensorBuf& buf);
    int32_t SetOutBuf(size_t index, TensorBuf& buf);
    int32_t SetTaskBuf(int &taskBufIndex);

    constexpr static uint32_t workTaskBufNum = 2;
    bool loadFlag_{false};
    std::mutex loadFlagMtx_;
    std::mutex executeMtx_;
    TensorBuf modelBuf_;
    uint32_t modelId_{-1U};
    svp_acl_mdl_desc *modelDesc_{nullptr};
    svp_acl_mdl_dataset *inputDataset_{nullptr};
    svp_acl_mdl_dataset *outputDataset_{nullptr};
    uint32_t inputNum_{0};
    uint32_t outputNum_{0};
    size_t workBufSize_{0};
    size_t workBufStride_{0};
    TensorBuf workBuf_;
    size_t taskBufSize_{0};
    size_t taskBufStride_{0};
    TensorBuf taskBuf_;
};
}