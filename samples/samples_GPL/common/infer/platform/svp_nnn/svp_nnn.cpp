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

#include "svp_nnn.h"
#include "log.h"
#include <fstream>
#include <cstring>

namespace Infer {
int32_t DevInit(const std::string& configPath)
{
    int ret;
    svp_acl_rt_run_mode runMode;
    ret = svp_acl_init(configPath.c_str());
    if (ret != 0) {
        ERROR_LOG("acl init failed");
        return -1;
    }

    ret = svp_acl_rt_set_device(0);
    if (ret != 0) {
        ERROR_LOG("acl set device failed");
        return -1;
    }

    ret = svp_acl_rt_get_run_mode(&runMode);
    if (ret != 0 || runMode != SVP_ACL_DEVICE) {
        ERROR_LOG("get runmode failed");
        return -1;
    }
    return 0;
}

int32_t DevDeInit()
{
    int ret = svp_acl_rt_reset_device(0);
    if (ret != 0 && ret != SVP_ACL_ERROR_UNINITIALIZE) {
        ERROR_LOG("acl reset device failed");
        return -1;
    }

    ret = svp_acl_finalize();
    if (ret != 0 && ret != SVP_ACL_ERROR_REPEAT_FINALIZE) {
        ERROR_LOG("acl deinit failed");
        return -1;
    }
    return 0;
}

int32_t InitData(int8_t * data, size_t size)
{
    memset(data, 0, size);
    return 0;
}

int32_t DevMalloc(void **devPtr, size_t size)
{
#ifdef ENABLE_SVP_ACCELERATE
    int ret = svp_acl_rt_malloc_cached(devPtr, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
#else
    int ret = svp_acl_rt_malloc(devPtr, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
#endif
    
    if (ret != 0) {
        ERROR_LOG("malloc mem failed. size is %u", size);
        return -1;
    }
    return 0;
}

int32_t DevFlush(void *devPtr, size_t size)
{
#ifdef ENABLE_SVP_ACCELERATE
    if(devPtr == nullptr) {
        INFO_LOG("flush mem empty");
        return -1;
    }
    int ret = svp_acl_rt_mem_flush(devPtr, size);
    if (ret != 0) {
        DevFree(devPtr);
        return -1;
    }
#endif
    return 0;
}

int32_t DevFree(void *devPtr)
{
    int ret = svp_acl_rt_free(devPtr);
    if (ret != 0) {
        ERROR_LOG("free mem failed");
        return -1;
    }
    return 0;
}
int32_t DevMemcpy(void *dst, size_t destMax, const void *src, size_t count)
{
    if (destMax < count) {
        return -1;
    }
    std::memcpy(dst, src, count);
    return 0;
}

TensorBuf::TensorBuf(size_t dataSize, size_t dataStride): data(nullptr), size(dataSize), stride(dataStride) 
{
    void* ptr = nullptr;
    if (dataSize > 0) {
        int ret = DevMalloc(&ptr, dataSize);
        if (ret == 0) {
            InitData(static_cast<int8_t*>(ptr), size);
            DevFlush(ptr, size);
        }
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
    DevFlush(data.get(), size);
}

TensorBuf TensorBuf::DeepCopy() 
{
    TensorBuf newBuf;
    newBuf.size = size;
    newBuf.stride = stride;
    if (data != nullptr) {
        void* newData = nullptr;
        DevMalloc(&newData, size);
        DevMemcpy(newData, size, data.get(), size);    
        newBuf.data.reset(newData, [](void* p) {
            if (p != nullptr) {
                DevFree(p);
            } 
        });
        DevFlush(newData, size); 
    }
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

std::shared_ptr<MdlBase> MdlCreate()
{
    return std::make_shared<SvpAclMdl>();
}

int32_t SvpAclMdl::CreateDesc()
{
    modelDesc_ = svp_acl_mdl_create_desc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("svp_acl_mdl_create_desc failed");
        return -1;
    }

    int ret = svp_acl_mdl_get_desc(modelDesc_, modelId_);
    if (ret != 0) {
        ERROR_LOG("svp_acl_mdl_get_desc failed");
        return -1;
    }

    return 0;
}

void SvpAclMdl::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)svp_acl_mdl_destroy_desc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

int32_t SvpAclMdl::CreateDataset()
{
    int ret;
    svp_acl_data_buffer *dataBuf = nullptr;

    inputDataset_ = svp_acl_mdl_create_dataset();
    if (inputDataset_ == nullptr) {
        ERROR_LOG("create input dataset failed");
        return -1;
    }

    outputDataset_ = svp_acl_mdl_create_dataset();
    if (outputDataset_ == nullptr) {
        ERROR_LOG("create output dataset failed");
        (void)svp_acl_mdl_destroy_dataset(inputDataset_);
        inputDataset_ = nullptr;
        return -1;
    }

    // add dataBuf to dataset
    for (size_t i = 0; i < inputNum_; i++) {
        dataBuf = svp_acl_create_data_buffer(workBuf_.data.get(), workBufSize_, workBufStride_);
        if (dataBuf == nullptr) {
            ERROR_LOG("get input data buffer failed");
            DestroyDataset();
            return -1;
        }
        ret = svp_acl_mdl_add_dataset_buffer(inputDataset_, dataBuf);
        if (ret != 0) {
            ERROR_LOG("add input data buffer failed");
            DestroyDataset();
            return -1;
        }
    }

    for (size_t i = 0; i < outputNum_; i++) {
        dataBuf = svp_acl_create_data_buffer(workBuf_.data.get(), workBufSize_, workBufStride_);

        if (dataBuf == nullptr) {
            ERROR_LOG("get output data buffer failed");
            DestroyDataset();
            return -1;
        }
        ret = svp_acl_mdl_add_dataset_buffer(outputDataset_, dataBuf);
        if (ret != 0) {
            ERROR_LOG("add output data buffer failed");
            DestroyDataset();
            return -1;
        }
    }

    return 0;
}

void SvpAclMdl::DestroyDataset()
{
    svp_acl_data_buffer *dataBuf = nullptr;
    if (inputDataset_ != nullptr) {
        for (int i = 0; i < inputNum_; i++) {
            dataBuf = svp_acl_mdl_get_dataset_buffer(inputDataset_, i);
            if (dataBuf != nullptr) {
                (void)svp_acl_destroy_data_buffer(dataBuf);
            }
        }
        (void)svp_acl_mdl_destroy_dataset(inputDataset_);
        inputDataset_ = nullptr;
    }
    if (outputDataset_ != nullptr) {
        for (int i = 0; i < outputNum_; i++) {
            dataBuf = svp_acl_mdl_get_dataset_buffer(outputDataset_, i);
            if (dataBuf != nullptr) {
                (void)svp_acl_destroy_data_buffer(dataBuf);
            }
        }
        (void)svp_acl_mdl_destroy_dataset(outputDataset_);
        outputDataset_ = nullptr;
    }
}

int32_t SvpAclMdl::GetInOutputNum()
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model desc, get input and output num failed");
        return -1;
    }
    inputNum_ = svp_acl_mdl_get_num_inputs(modelDesc_);
    if (inputNum_ <= 2) { /* 2：workBuf taskBuf */
        ERROR_LOG("svp_acl_mdl_get_num_inputs failed");
        return -1;
    }
    outputNum_ = svp_acl_mdl_get_num_outputs(modelDesc_);
    if (outputNum_ <= 0) {
        ERROR_LOG("svp_acl_mdl_get_num_outputs failed");
        return -1;
    }
    return 0;
}

int32_t SvpAclMdl::CreateWorkAndTaskBuf()
{
    int ret;
    void *taskBuf, *workBuf;
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model desc, create workBuf and taskBuf failed");
        return -1;
    }
    // create task buf
    taskBufSize_ = svp_acl_mdl_get_input_size_by_index(modelDesc_, inputNum_ - workTaskBufNum);
    taskBufStride_ = svp_acl_mdl_get_input_default_stride(modelDesc_, inputNum_ - workTaskBufNum);
    taskBuf_ = TensorBuf(taskBufSize_, taskBufStride_);

    // create work buf
    workBufSize_ = svp_acl_mdl_get_input_size_by_index(modelDesc_, inputNum_ - 1);
    workBufStride_ = svp_acl_mdl_get_input_default_stride(modelDesc_, inputNum_ - 1);
    workBuf_ = TensorBuf(workBufSize_, workBufStride_);

    return 0;

}

int32_t SvpAclMdl::ReadFile(const std::string& filePath, void *&ptr, uint32_t &size)
{
    std::ifstream file(filePath, std::ifstream::binary);
    if (!file.is_open()) {
        ERROR_LOG("open model file[%s] failed", filePath.c_str());
        return -1;
    }
    file.seekg(0, file.end);
    size = file.tellg();
    if (size == 0) {
        ERROR_LOG("model file:%s size is 0", filePath.c_str());
        file.close();
        return -1;
    }
#ifdef ENABLE_SVP_ACCELERATE
    auto ret = svp_acl_rt_malloc_cached(&ptr, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
#else
    auto ret = svp_acl_rt_malloc(&ptr, size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
#endif
    if (ret != 0) {
        ERROR_LOG("svp_acl_rt_malloc model buffer failed. size is %u", size);
        file.close();
        return -1;
    }

    file.seekg(0, file.beg);
    file.read(static_cast<char *>(ptr), size);
    file.close();
    DevFlush(ptr, size);
    return 0;
}

int32_t SvpAclMdl::LoadModelFromMem(const void* modelBuf, size_t modelSize)
{
    int ret;
    ret = svp_acl_mdl_load_from_mem(modelBuf, modelSize, &modelId_);
    if (ret != 0) {
        svp_acl_rt_free(modelBuf);
        ERROR_LOG("svp_acl_mdl_load_from_mem call failed");
        return -1;
    }

    ret = CreateDesc();
    if (ret != 0) {
        ERROR_LOG("create desc failed");
        return -1;
    }

    ret = GetInOutputNum();
    if (ret != 0) {
        DestroyDesc();
        ERROR_LOG("get input and output Num failed");
        return -1;
    }

    ret = CreateWorkAndTaskBuf();
    if (ret != 0) {
        DestroyDesc();
        ERROR_LOG("CreateWorkAndTaskBuf failed");
        return -1;
    }

    ret = CreateDataset();
    if (ret != 0) {
        DestroyDesc();
        ERROR_LOG("CreateDataset failed");
        return -1;
    }
    return 0;
}

int32_t SvpAclMdl::LoadModel(const std::string& modelPath)
{
    std::lock_guard<std::mutex> lock(loadFlagMtx_);
    if (loadFlag_) {
        ERROR_LOG("has already loaded a model");
        return -1;
    }

    void *modelBuf = nullptr;
    uint32_t modelSize = 0;
    if (ReadFile(modelPath, modelBuf, modelSize) != 0) {
        ERROR_LOG("read model file failed");
        return -1;
    }
    // modelPtr_ = UniquePtr{modelBuf, &deleter};
    modelBuf_.ResetData(modelBuf);

    if (LoadModelFromMem(modelBuf, modelSize) != 0) {
        ERROR_LOG("load model:[%s] failed", modelPath.c_str());
        return -1;
    }
    loadFlag_ = true;
    PrintModelDesc();
    return 0;
}

int32_t SvpAclMdl::LoadModel(const char* modelBuf, size_t modelSize)
{
    std::lock_guard<std::mutex> lock(loadFlagMtx_);
    if (loadFlag_) {
        ERROR_LOG("has already loaded a model");
        return -1;
    }
    if (LoadModelFromMem(modelBuf, modelSize) != 0) {
        ERROR_LOG("load model failed");
        return -1;
    }
    loadFlag_ = true;
    PrintModelDesc();
    return 0;
}

int32_t SvpAclMdl::UnLoadModel()
{
    std::lock_guard<std::mutex> lock(loadFlagMtx_);

    if (!loadFlag_) {
        WARN_LOG("model had been unloaded or is not loaded");
        return -1;
    }

    DestroyDataset();
    DestroyDesc();
    int ret = svp_acl_mdl_unload(modelId_);
    if (ret != 0) {
        LOG(ERROR) << "unload model failed, modelId is " << modelId_ << ", errorCode is " << ret;
        return -1;
    }
    loadFlag_ = false;
    return 0;
}

size_t SvpAclMdl::GetInTensorNum()
{
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    return static_cast<size_t>(inputNum_ - workTaskBufNum);
}

size_t SvpAclMdl::GetOutTensorNum()
{
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    return static_cast<size_t>(outputNum_);
}

int32_t SvpAclMdl::GetInTensorDescByIdx(size_t index, TensorDesc& desc)
{
    int ret;
    svp_acl_mdl_io_dims ioDims;
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    ret = svp_acl_mdl_get_input_dims(modelDesc_, index, &ioDims);
    if (ret != 0) {
        ERROR_LOG("get model input dims failed");
        return -1;  
    }
    desc.dimCount = ioDims.dim_count;
    for (size_t i = 0; i < ioDims.dim_count && i < MAX_TENSOR_DIM; i++) {
        desc.dims[i] = ioDims.dims[i];
    }
    desc.type = static_cast<TensorType>(svp_acl_mdl_get_input_data_type(modelDesc_, index));

    desc.typeSize = svp_acl_data_type_size(static_cast<svp_acl_data_type>(desc.type));

    desc.format = static_cast<TensorFormat>(svp_acl_mdl_get_input_format(modelDesc_, index));

    desc.defaultStride = svp_acl_mdl_get_input_default_stride(modelDesc_, index);

    desc.defaultSize = svp_acl_mdl_get_input_size_by_index(modelDesc_, index);

    return 0;
}

int32_t SvpAclMdl::GetInTensorDescByName(const std::string& name, TensorDesc& desc)
{
    int ret;
    size_t index;
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    ret = svp_acl_mdl_get_input_index_by_name(modelDesc_, name.c_str(), &index);
    if (ret != 0) {
        ERROR_LOG("get input index by name failed");
        return -1;
    }

    ret = GetInTensorDescByIdx(index, desc);
    if (ret != 0) {
        ERROR_LOG("get in tensor desc by idx failed");
        return -1;
    }
    return 0;
}

int32_t SvpAclMdl::GetOutTensorDescByIdx(size_t index, TensorDesc& desc)
{
    int ret;
    svp_acl_mdl_io_dims ioDims;
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    ret = svp_acl_mdl_get_output_dims(modelDesc_, index, &ioDims);
    if (ret != 0) {
        ERROR_LOG("get model output dims failed");
        return -1;  
    }
    desc.dimCount = ioDims.dim_count;
    for (size_t i = 0; i < ioDims.dim_count && i < MAX_TENSOR_DIM; i++) {
        desc.dims[i] = ioDims.dims[i];
    }
    desc.type = static_cast<TensorType>(svp_acl_mdl_get_output_data_type(modelDesc_, index));

    desc.typeSize = svp_acl_data_type_size(static_cast<svp_acl_data_type>(desc.type));

    desc.format = static_cast<TensorFormat>(svp_acl_mdl_get_output_format(modelDesc_, index));

    desc.defaultStride = svp_acl_mdl_get_output_default_stride(modelDesc_, index);

    desc.defaultSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, index);

    return 0;
}

int32_t SvpAclMdl::GetOutTensorDescByName(const std::string& name, TensorDesc& desc)
{
    int ret;
    size_t index;
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    ret = svp_acl_mdl_get_output_index_by_name(modelDesc_, name.c_str(), &index);
    if (ret != 0) {
        ERROR_LOG("get output index by name failed");
        return -1;
    }

    ret = GetOutTensorDescByIdx(index, desc);
    if (ret != 0) {
        ERROR_LOG("get Out tensor desc by idx failed");
        return -1;
    }
    return 0;
}

int32_t SvpAclMdl::SetInBuf(size_t index, TensorBuf& buf)
{
    int ret;
    svp_acl_data_buffer *dataBuf = nullptr;
    size_t inputSize, inputStride;

    inputSize = svp_acl_mdl_get_input_size_by_index(modelDesc_, index);
    if (buf.size != inputSize) {
        ERROR_LOG("buf size[%zu] is small than model %zuth input size[%zu]", buf.size, index, inputSize);
        return -1;
    }

    inputStride = svp_acl_mdl_get_input_default_stride(modelDesc_, index);
    if (buf.stride != inputStride) {
        ERROR_LOG("buf stride[%zu] is small than model %zuth input stride[%zu]", buf.stride, index, inputStride);
        return -1;
    }

    dataBuf = svp_acl_mdl_get_dataset_buffer(inputDataset_, index);
    if (dataBuf == nullptr) {
        ERROR_LOG("get input dataset %zuth buffer failed", index);
        return -1;
    }
    ret = svp_acl_update_data_buffer(dataBuf, buf.data.get(), buf.size, buf.stride);
    if (ret != 0) {
        ERROR_LOG("update input %zuth databuffer failed", index);
        return -1;
    }
    DevFlush(buf.data.get(), buf.size);
    return 0;
}

int32_t SvpAclMdl::SetOutBuf(size_t index, TensorBuf& buf)
{
    int ret;
    svp_acl_data_buffer *dataBuf = nullptr;
    size_t outputSize, outputStride;

    outputSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, index);
    if (buf.size != outputSize) {
        ERROR_LOG("buf size[%zu] is small than model %zuth output size[%zu]", buf.size, index, outputSize);
        return -1;
    }

    outputStride = svp_acl_mdl_get_output_default_stride(modelDesc_, index);
    if (buf.stride != outputStride) {
        ERROR_LOG("buf stride[%zu] is small than model %zuth output stride[%zu]", buf.stride, index, outputStride);
        return -1;
    }

    dataBuf = svp_acl_mdl_get_dataset_buffer(outputDataset_, index);
    if (dataBuf == nullptr) {
        ERROR_LOG("get output dataset %zuth buffer failed", index);
        return -1;
    }
    ret = svp_acl_update_data_buffer(dataBuf, buf.data.get(), buf.size, buf.stride);
    if (ret != 0) {
        ERROR_LOG("update output %zuth databuffer failed", index);
        return -1;
    }

    DevFlush(buf.data.get(), buf.size);
    return 0;
}

int32_t SvpAclMdl::Execute(std::vector<TensorBuf> &inBufs, std::vector<TensorBuf> &outBufs,
    RunMode runMode)
{
    int ret;
    int taskBufIndex;

    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }

    size_t inputBufSize = inBufs.size();
    if (inputBufSize == (inputNum_  + 1 - workTaskBufNum)) {
        inputBufSize--;
    }

    if (inputBufSize != (inputNum_ - workTaskBufNum)) {
        ERROR_LOG("input Tensor Buf size[%zu] is not equal to model inputNum[%u]", inputBufSize, inputNum_);
        return -1;
    }

    if (outBufs.size() != (outputNum_)) {
        ERROR_LOG("output Tensor Buf size[%zu] is not equal to model outputNum[%u]", inBufs.size(), outputNum_);
        return -1;
    }

    /* 加锁保证多线程安全 */
    std::lock_guard<std::mutex> locker(executeMtx_);
    for (size_t i = 0; i < inputBufSize; i++) {
        ret = SetInBuf(i, inBufs[i]);
        if (ret != SUCCESS) {
            ERROR_LOG("set %zuth input buf faile", i);
            return -1;
        }
    }
    // set task buf
    ret = SetInBuf(inputNum_ - workTaskBufNum, taskBuf_);
    if (ret != SUCCESS) {
        ERROR_LOG("set taskbuf faile");
        return -1;
    }

    for (size_t i = 0; i < outBufs.size(); i++) {
        ret = SetOutBuf(i, outBufs[i]);
        if (ret != SUCCESS) {
            ERROR_LOG("set %zuth output buf faile", i);
            return -1;
        }
    }
    static std::chrono::microseconds dur(0);
    (void)svp_acl_rt_set_device(0);
    if (runMode == RunMode::Sync) {
        auto start = std::chrono::high_resolution_clock::now();
        int ret = svp_acl_mdl_execute(modelId_, inputDataset_, outputDataset_);
        auto end = std::chrono::high_resolution_clock::now();
        (void)svp_acl_rt_reset_device(0);
        if (ret != SUCCESS) {
            ERROR_LOG("model execute failed");
            return -1;
        }
        dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        *(static_cast<float*>(inBufs[inBufs.size() - 1].GetRawPtr())) = dur.count();
    } else if (runMode == RunMode::Async) {
        int ret = svp_acl_mdl_execute_async(modelId_, inputDataset_, outputDataset_, nullptr);
        (void)svp_acl_rt_reset_device(0);
        if (ret != SUCCESS) {
            ERROR_LOG("model execute async failed");
            return -1;
        }
    } else {
        ERROR_LOG("invalid runmode");
        return -1;
    }
    return 0;
}

int32_t SvpAclMdl::Wait()
{
    if (!loadFlag_) {
        ERROR_LOG("model had been unloaded or is not loaded, please load model first");
        return -1;
    }
    std::lock_guard<std::mutex> locker(executeMtx_);
    (void)svp_acl_rt_set_device(0);
    int ret = svp_acl_rt_synchronize_stream(nullptr);
    (void)svp_acl_rt_reset_device(0);
    if (ret != 0) {
        ERROR_LOG("synchronize stream failed");
        return -1;
    }
    return 0;
}
}