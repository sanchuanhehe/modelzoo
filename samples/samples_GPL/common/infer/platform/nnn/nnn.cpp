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

#include "nnn.h"
#include "log.h"

namespace Infer {
using namespace Infer;

TensorBuf::TensorBuf(size_t dataSize, size_t dataStride): data(nullptr), size(dataSize), stride(dataStride) 
{
    void* ptr = nullptr;
    if (dataSize > 0) {
        DevMalloc(&ptr, dataSize);
    }
    data.reset(ptr, [](void* p) {
        if (p != nullptr) {
            DevFree(p);
        } 
    });
}

TensorBuf::TensorBuf(void* externalData, size_t dataSize, size_t dataStride) 
{
    data.reset(externalData, [](void* p) {}); // 空删除器，不释放外部内存
    size = dataSize;
    stride = dataStride;
}

TensorBuf TensorBuf::DeepCopy() 
{
    TensorBuf newBuf;
    void* newData = nullptr;
    if (DevMalloc(&newData, size) != 0) {
        return newBuf;
    }
    if (data == nullptr || DevMemcpy(newData, size, data.get(), size) != 0) {
        DevFree(newData);
        return newBuf;
    }
    newBuf.data.reset(newData, [](void* p) {
        if (p != nullptr) {
            DevFree(p);
        }
    });
    newBuf.size = size;
    newBuf.stride = stride;
    return newBuf;
}

void TensorBuf::ResetData(void* newData)
{ 
    data.reset(newData, [](void* p) {
        if (p != nullptr) {
            DevFree(p);
        } 
    });
}

int32_t DevInit(const std::string& configPath)
{
    const char *aclConfigPath = configPath.empty() ? NULL : configPath.c_str();
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "acl init failed, errorCode is " << ret;
        return -1;
    }

    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "acl set device failed, errorCode is " << ret;
        return -1;
    }
    LOG(INFO) << "acl init success";
    return 0;
}

int32_t DevDeInit()
{
    aclError ret = aclrtResetDevice(0);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "acl reset device failed, errorCode is " << ret;
        return -1;
    }

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "acl finalize failed, errorCode is " << ret;
        return -1;
    }
    LOG(INFO) << "acl finalize success";
    return 0;
}

int32_t DevMalloc(void **devPtr, size_t size)
{
    aclrtSetDevice(0);
    aclError ret = aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "malloc mem failed, size is " << size  << ", errorCode is " << ret;
        aclrtResetDevice(0);
        return -1;
    }
    aclrtResetDevice(0);
    return 0;
}

int32_t DevFlush(void *devPtr, size_t size)
{
    return 0;
}

int32_t DevFree(void *devPtr)
{
    aclrtSetDevice(0);
    aclError ret = aclrtFree(devPtr);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "free mem failed, errorCode is " << ret;
        aclrtResetDevice(0);
        return -1;
    }
    aclrtResetDevice(0);
    return 0;
}

int32_t DevMemcpy(void *dst, size_t destMax, const void *src, size_t count)
{
    aclrtSetDevice(0);
    aclError ret = aclrtMemcpy(dst, destMax, src, count, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "copy mem failed, errorCode is " << ret;
        aclrtResetDevice(0);
        return -1;
    }
    aclrtResetDevice(0);
    return 0;
}

std::shared_ptr<MdlBase> MdlCreate()
{
    return std::make_shared<AclMdl>();
}

int32_t AclMdl::LoadModel(const std::string& modelPath)
{
    std::lock_guard<std::mutex> lock(loadFlagMtx_);
    if (loadFlag_) {
        LOG(ERROR) << "has already loaded a model";
        return -1;
    }

    aclError ret = aclmdlLoadFromFile(modelPath.c_str(), &modelId_);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "load model failed, model file is " << modelPath << ", errorCode is " << ret;
        return -1;
    }

    if (CreateModelDesc() != 0) {
        LOG(ERROR) << "create model description failed";
        return -1;
    }

    if (InitInput() != 0) {
        LOG(ERROR) << "init model input failed";
        return -1;
    }
    if (InitOutput() != 0) {
        LOG(ERROR) << "init model output failed";
        return -1;
    }

    loadFlag_ = true;
    LOG(INFO) << "load model success, model file is " << modelPath;
    PrintModelDesc();
    return 0;
}

int32_t AclMdl::LoadModel(const char* modelBuf, size_t modelSize)
{
    std::lock_guard<std::mutex> lock(loadFlagMtx_);
    if (loadFlag_) {
        LOG(ERROR) << "has already loaded a model";
        return -1;
    }

    aclError ret = aclmdlLoadFromMem(modelBuf, modelSize, &modelId_);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "load model failed, errorCode is " << ret;
        return -1;
    }

    if (CreateModelDesc() != 0) {
        LOG(ERROR) << "create model description failed";
        return -1;
    }

    if (InitInput() != 0) {
        LOG(ERROR) << "init model input failed";
        return -1;
    }
    if (InitOutput() != 0) {
        LOG(ERROR) << "init model output failed";
        return -1;
    }

    loadFlag_ = true;
    LOG(INFO) << "load model success";
    PrintModelDesc();
    return 0;
}

int32_t AclMdl::CreateModelDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        LOG(ERROR) << "create model description failed";
        return -1;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "get model description failed, errorCode is " << ret;
        return -1;
    }
    return 0;
}

int32_t AclMdl::InitInput()
{
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        LOG(ERROR) << "can't create input dataset";
        return -1;
    }
    size_t inputSize = aclmdlGetNumInputs(modelDesc_);
    for (size_t i = 0; i < inputSize; ++i) {
        aclDataBuffer* inputData = aclCreateDataBuffer(defaultBuf_.GetRawPtr(), defaultBuf_.size);
        aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << "can't add data buffer";
            aclDestroyDataBuffer(inputData);
            return -1;
        }
    }
    return 0;
}
int32_t AclMdl::InitOutput()
{
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        LOG(ERROR) << "can't create output dataset";
        return -1;
    }
    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        aclDataBuffer* outputData = aclCreateDataBuffer(defaultBuf_.GetRawPtr(), defaultBuf_.size);
        aclError ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << "can't add data buffer";
            aclDestroyDataBuffer(outputData);
            return -1;
        }
    }
    return 0;   
}

int32_t AclMdl::UnLoadModel()
{
    std::lock_guard<std::mutex> lock(loadFlagMtx_);
    if (!loadFlag_) {
        LOG(WARNING) << "no model had been loaded, unload failed";
        return 0;
    }
    defaultBuf_.ResetData();
    DestroyDataset(input_);
    DestroyDataset(output_);

    if (modelDesc_ != nullptr) {
        aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "unload model failed, modelId is " << modelId_ << ", errorCode is " << ret;
        return -1;
    }
    loadFlag_ = false;
    LOG(INFO) << "unload model success, modelId is " << modelId_;
    modelId_ = 0;
    return 0;
}

void AclMdl::DestroyDataset(aclmdlDataset* dataset)
{
    if (dataset == nullptr) {
        return;
    }
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(dataset, i);
        aclDestroyDataBuffer(dataBuffer);
    }
    aclmdlDestroyDataset(dataset);
    return;
}

size_t AclMdl::GetInTensorNum()
{
    if (!loadFlag_) {
        LOG(ERROR) << "model had been unloaded or is not loaded, please load model first";
        return 0;
    }
    return aclmdlGetNumInputs(modelDesc_);
}

int32_t AclMdl::GetInTensorDescByIdx(size_t index, TensorDesc& tensorDesc)
{
    if (!loadFlag_) {
        LOG(ERROR) << "model had been unloaded or is not loaded, please load model first";
        return -1;
    }
    aclDataType dataType = aclmdlGetInputDataType(modelDesc_, index);
    tensorDesc.type = static_cast<TensorType>(static_cast<int>(dataType));
    tensorDesc.typeSize = aclDataTypeSize(dataType) * 8; // 1Byte = 8bits

    aclFormat dataFormat = aclmdlGetInputFormat(modelDesc_, index);
    tensorDesc.format = static_cast<TensorFormat>(static_cast<int>(dataFormat));

    tensorDesc.defaultSize = aclmdlGetInputSizeByIndex(modelDesc_, index);

    aclmdlIODims mdlIODims;
    aclError ret = aclmdlGetInputDims(modelDesc_, index, &mdlIODims);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "get model input dims failed, errorCode is " << ret;
        return -1;  
    }
    tensorDesc.dimCount = mdlIODims.dimCount;
    for (size_t i = 0; i < mdlIODims.dimCount && i < MAX_TENSOR_DIM; ++i) {
        tensorDesc.dims[i] = mdlIODims.dims[i];
    }
    return 0;
}

int32_t AclMdl::GetInTensorDescByName(const std::string& name, TensorDesc& tensorDesc)
{
    if (!loadFlag_) {
        LOG(ERROR) << "model had been unloaded or is not loaded, please load model first";
        return -1;
    }
    size_t index = 0;
    aclError ret = aclmdlGetInputIndexByName(modelDesc_, name.c_str(), &index);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "get model input index failed, errorCode is " << ret;
        return -1;
    }
    if (GetInTensorDescByIdx(index, tensorDesc) != 0) {
        LOG(ERROR) << "get model input tensor description failed";
        return -1;
    }
    return 0;
}

size_t AclMdl::GetOutTensorNum()
{
    if (!loadFlag_) {
        LOG(ERROR) << "model had been unloaded or is not loaded, please load model first";
        return 0;
    }
    return aclmdlGetNumOutputs(modelDesc_);
}

int32_t AclMdl::GetOutTensorDescByIdx(size_t index, TensorDesc& tensorDesc)
{
    if (!loadFlag_) {
        LOG(ERROR) << "model had been unloaded or is not loaded, please load model first";
        return -1;
    }
    aclDataType dataType = aclmdlGetOutputDataType(modelDesc_, index);
    tensorDesc.type = static_cast<TensorType>(static_cast<int>(dataType));
    tensorDesc.typeSize = aclDataTypeSize(dataType) * 8; // 1Byte = 8bits

    aclFormat dataFormat = aclmdlGetOutputFormat(modelDesc_, index);
    tensorDesc.format = static_cast<TensorFormat>(static_cast<int>(dataFormat));

    tensorDesc.defaultSize = aclmdlGetOutputSizeByIndex(modelDesc_, index);

    aclmdlIODims mdlIODims;
    aclError ret = aclmdlGetOutputDims(modelDesc_, index, &mdlIODims);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "get model output dims failed, index is " << index << ", errorCode is " << ret;
        return -1;  
    }
    tensorDesc.dimCount = mdlIODims.dimCount;
    for (size_t i = 0; i < mdlIODims.dimCount && i < MAX_TENSOR_DIM; ++i) {
        tensorDesc.dims[i] = mdlIODims.dims[i];
    }
    return 0;
}

int32_t AclMdl::GetOutTensorDescByName(const std::string& name, TensorDesc& tensorDesc)
{
    if (!loadFlag_) {
        LOG(ERROR) << "model had been unloaded or is not loaded, please load model first";
        return -1;
    }
    size_t index = 0;
    aclError ret = aclmdlGetOutputIndexByName(modelDesc_, name.c_str(), &index);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "get model output index failed, name is " << name << ", errorCode is " << ret;
        return -1;
    }
    if (GetOutTensorDescByIdx(index, tensorDesc) != 0) {
        LOG(ERROR) << "get model output tensor description failed, index is " << index;
        return -1;
    }
    return 0;
}

int32_t AclMdl::UpdateDatasetBuffer(size_t index, TensorBuf& tensorBuf, aclmdlDataset* dataset)
{
    aclDataBuffer* datasetBuffer = aclmdlGetDatasetBuffer(dataset, index);
    if (datasetBuffer == nullptr) {
        LOG(ERROR) << "get dataset buffer failed";
        return -1;
    }
    aclError ret = aclUpdateDataBuffer(datasetBuffer, tensorBuf.GetRawPtr(), tensorBuf.size);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "update dataset buffer failed, errorCode is " << ret;
        return -1;
    }
    return 0;
}

int32_t AclMdl::Execute(std::vector<TensorBuf>& inBufs, std::vector<TensorBuf>& outBufs, RunMode runMode)
{
    size_t inputBufSize = inBufs.size();
    bool timeCostFlag = false;
    static std::chrono::microseconds dur(0);
    std::chrono::high_resolution_clock::time_point start;
    if (inputBufSize == GetInTensorNum()  + 1) {
        inputBufSize--;
        timeCostFlag = true;
    }

    if (inputBufSize != GetInTensorNum() || outBufs.size() != GetOutTensorNum()) {
        LOG(ERROR) << "numbers of model inputs or outputs are not correct";
        return -1;
    }
    std::lock_guard<std::mutex> lock(executeMtx_);
    for (size_t i = 0; i < inputBufSize; ++i) {
         if (UpdateDatasetBuffer(i, inBufs[i], input_) != 0) {
            LOG(ERROR) << "update input buf failed";
            return -1;
        }
    }
    for (size_t i = 0; i < outBufs.size(); ++i) {
        if (UpdateDatasetBuffer(i, outBufs[i], output_) != 0) {
            LOG(ERROR) << "update output buf failed";
            return -1;
        }
    }
    aclrtSetDevice(0);
    aclError ret = ACL_SUCCESS;
    if (runMode == RunMode::Sync) {
        if (timeCostFlag) {
            start = std::chrono::high_resolution_clock::now();
        }
        ret = aclmdlExecute(modelId_, input_, output_);
        if (timeCostFlag) {
            auto end = std::chrono::high_resolution_clock::now();
            dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            *(static_cast<float*>(inBufs[inBufs.size() - 1].GetRawPtr())) = dur.count();
        }
    } else if (runMode == RunMode::Async) {
        ret = aclmdlExecuteAsync(modelId_, input_, output_, NULL);
    }
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "execute model failed, run mode is " << runMode << ", errorCode is " << ret;
        aclrtResetDevice(0);
        return -1;
    }
    aclrtResetDevice(0);
    return 0;
}

int32_t AclMdl::Wait()
{
    aclrtSetDevice(0);
    aclError ret = aclrtSynchronizeStream(NULL);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << "synchronize stream failed, errorCode is " << ret;
        aclrtResetDevice(0);
        return -1;
    }
    aclrtResetDevice(0);
    return 0;
}
}