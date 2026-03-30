 /*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd
 * This file is part of [Hispark/modelzoo].
 *
 * [Hispark/modelzoo] is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, version 3 of the License only.
 *
 * [Hispark/modelzoo] is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with [Hispark/modelzoo].  If not, see <https://www.gnu.org/licenses/>.
 */

#include "model_process.h"

#include <algorithm>
#include <map>
#include <sstream>
#include <fstream>
#include <cstring> 
#include "utils.h"
#include "worker_types.h"

using namespace std;

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

namespace {

uint32_t MapElementType(svp_acl_data_type dataType)
{
    switch (dataType) {
        case SVP_ACL_FLOAT:
            return WORKER_ELEM_FLOAT32;
        case SVP_ACL_FLOAT16:
            return WORKER_ELEM_FLOAT16;
        case SVP_ACL_INT8:
            return WORKER_ELEM_INT8;
        case SVP_ACL_UINT8:
            return WORKER_ELEM_UINT8;
        case SVP_ACL_INT32:
            return WORKER_ELEM_INT32;
        case SVP_ACL_INT64:
            return WORKER_ELEM_INT64;
        default:
            return WORKER_ELEM_UNKNOWN;
    }
}

}  // namespace

ModelProcess::ModelProcess()
{
}

ModelProcess::~ModelProcess()
{
    // Resource ownership is managed explicitly by SampleProcess::DestroyResource().
    // Avoid calling ACL-related cleanup here because SampleProcess finalizes ACL
    // before this member destructor runs.
}

void ModelProcess::DestroyResource()
{
    DestroyInput();
    DestroyOutput();
    DestroyDesc();
    Unload();
}

Result ModelProcess::LoadModelFromFileWithMem(const std::string& modelPath)
{
    uint32_t fileSize = 0;
    modelMemPtr_ = Utils::ReadBinFile(modelPath, fileSize);
    modelMemSize_ = fileSize;
    if (modelMemPtr_ == nullptr || modelMemSize_ == 0) {
        ERROR_LOG("read model file failed, model file is %s", modelPath.c_str());
        return FAILED;
    }

    svp_acl_mdl_config_handle* handle = svp_acl_mdl_create_config_handle();
    if (handle != nullptr) {
        const size_t loadType = SVP_ACL_MDL_LOAD_FROM_MEM;
        svp_acl_error cfgRet = svp_acl_mdl_set_config_opt(
            handle, SVP_ACL_MDL_LOAD_TYPE_SIZET, &loadType, sizeof(loadType));
        if (cfgRet == SVP_ACL_SUCCESS) {
            cfgRet = svp_acl_mdl_set_config_opt(
                handle, SVP_ACL_MDL_MEM_ADDR_PTR, &modelMemPtr_, sizeof(modelMemPtr_));
        }
        if (cfgRet == SVP_ACL_SUCCESS) {
            cfgRet = svp_acl_mdl_set_config_opt(
                handle, SVP_ACL_MDL_MEM_SIZET, &modelMemSize_, sizeof(modelMemSize_));
        }
        if (cfgRet == SVP_ACL_SUCCESS) {
            cfgRet = svp_acl_mdl_load_with_config(handle, &modelId_);
        }
        (void)svp_acl_mdl_destroy_config_handle(handle);

        if (cfgRet == SVP_ACL_SUCCESS) {
            loadFlag_ = true;
            INFO_LOG("load model %s success by config(mem), modelId=%u", modelPath.c_str(), modelId_);
            return SUCCESS;
        }
        WARN_LOG("load model by config(mem) failed, ret=%d, fallback to load_from_mem", static_cast<int>(cfgRet));
    } else {
        WARN_LOG("create model config handle failed, fallback to load_from_mem");
    }

    svp_acl_error ret = svp_acl_mdl_load_from_mem(static_cast<uint8_t* >(modelMemPtr_), modelMemSize_, &modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        svp_acl_rt_free(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
        ERROR_LOG("load model from file failed, model file is %s, ret=%d", modelPath.c_str(), static_cast<int>(ret));
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath.c_str());
    return SUCCESS;
}

Result ModelProcess::CreateDesc()
{
    modelDesc_ = svp_acl_mdl_create_desc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    svp_acl_error ret = svp_acl_mdl_get_desc(modelDesc_, modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)svp_acl_mdl_destroy_desc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::InitInput()
{
    input_ = svp_acl_mdl_create_dataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize, int stride)
{
    svp_acl_data_buffer* inputData = svp_acl_create_data_buffer(inputDataBuffer, bufferSize, stride);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    svp_acl_error ret = svp_acl_mdl_add_dataset_buffer(input_, inputData);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed");
        svp_acl_destroy_data_buffer(inputData);
        inputData = nullptr;
        return FAILED;
    }

    return SUCCESS;
}

size_t ModelProcess::GetInputDataSize(int index) const
{
    svp_acl_data_type dataType = svp_acl_mdl_get_input_data_type(modelDesc_, index);
    return svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
}

size_t ModelProcess::GetOutputDataSize(int index) const
{
    svp_acl_data_type dataType = svp_acl_mdl_get_output_data_type(modelDesc_, index);
    return svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
}

Result ModelProcess::GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const
{
    svp_acl_error ret = svp_acl_mdl_get_output_dims(modelDesc_, index, &dims);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_mdl_get_output_dims error!");
        return FAILED;
    }
    stride = svp_acl_mdl_get_output_default_stride(modelDesc_, index);
    if (stride == 0) {
        ERROR_LOG("svp_acl_mdl_get_output_default_stride error!");
        return FAILED;
    }
    bufSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("svp_acl_mdl_get_output_size_by_index error!");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateOutput()
{
    output_ = svp_acl_mdl_create_dataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t outputSize = svp_acl_mdl_get_num_outputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t stride = svp_acl_mdl_get_output_default_stride(modelDesc_, i);
        if (stride == 0) {
            ERROR_LOG("Error, output default stride is %lu.", stride);
            return FAILED;
        }
        size_t bufferSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, i);
        if (bufferSize == 0) {
            ERROR_LOG("Error, output size is %lu.", bufferSize);
            return FAILED;
        }

        void *outputBuffer = nullptr;
        svp_acl_error ret = svp_acl_rt_malloc(&outputBuffer, bufferSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", bufferSize);
            return FAILED;
        }
        Utils::InitData(static_cast<int8_t*>(outputBuffer), bufferSize);

        svp_acl_data_buffer* outputData = svp_acl_create_data_buffer(outputBuffer, bufferSize, stride);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            svp_acl_rt_free(outputBuffer);
            return FAILED;
        }
        ret = svp_acl_mdl_add_dataset_buffer(output_, outputData);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed");
            svp_acl_rt_free(outputBuffer);
            svp_acl_destroy_data_buffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

Result ModelProcess::ClearOutputStrideInvalidBuf(std::vector<int8_t>& buffer, size_t index) const
{
    size_t bufSize = 0;
    size_t bufStride = 0;
    svp_acl_mdl_io_dims dims;
    svp_acl_error ret = GetOutputStrideParam(index, bufSize, bufStride, dims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetOutputStrideParam failed");
        return FAILED;
    }
    if ((bufSize == 0) || (bufStride == 0)) {
        ERROR_LOG("Error, bufSize(%zu) bufStride(%zu) invalid", bufSize, bufStride);
        return FAILED;
    }
    if ((dims.dim_count == 0) || (dims.dims[dims.dim_count - 1] <= 0)) {
        ERROR_LOG("Error, dims para invalid");
        return FAILED;
    }
    int64_t lastDim = dims.dims[dims.dim_count - 1];

    size_t dataSize = GetOutputDataSize(index);
    if (dataSize == 0) {
        ERROR_LOG("Error, dataSize == 0 invalid");
        return FAILED;
    }
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = bufSize / bufStride;
    size_t invalidSize = bufStride - lastDimSize;
    if (invalidSize == 0) {
        // not stride invalid space, return directly.
        return SUCCESS;
    }

    for (size_t i = 0; i < loopNum; ++i) {
        size_t offset = bufStride * i + lastDimSize;
        int8_t* ptr = &buffer[offset];
        for (size_t index = 0; index < invalidSize; index++) {
            ptr[index] = 0;
        }
    }
    return SUCCESS;
}

void ModelProcess::WriteOutput(const string& outputFileName, size_t index) const
{
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, index);
    if (dataBuffer == nullptr) {
        ERROR_LOG("output[%zu] dataBuffer nullptr invalid", index);
        return;
    }
    int8_t* outData = (int8_t*)svp_acl_get_data_buffer_addr(dataBuffer);
    size_t outSize = svp_acl_get_data_buffer_size(dataBuffer);
    if (outData == nullptr || outSize == 0) {
        ERROR_LOG("output[%zu] data or size(%zu) invalid", index, outSize);
        return;
    }

    // malloc temp buffer to clear stride useless temp buffer to help output.bin compare
    std::vector<int8_t> tempBuf(outData, outData + outSize);
    if (tempBuf.empty()) {
        ERROR_LOG("tempBuf malloc fail");
        return;
    }

    Result ret = ClearOutputStrideInvalidBuf(tempBuf, index);
    if (ret != SUCCESS) {
        ERROR_LOG("ClearStrideInvalidBuf fail");
        return;
    }

    ofstream fout(outputFileName, ios::out|ios::binary);
    if (fout.good() == false) {
        ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
        return;
    }
    fout.write((char*)&tempBuf[0], tempBuf.size() * sizeof(int8_t));
    fout.close();
    return;
}

// 新增：获取模型输入个数
size_t ModelProcess::GetInputNum() const {
    if (modelDesc_ == nullptr) return 0;
    return svp_acl_mdl_get_num_inputs(modelDesc_);
}

size_t ModelProcess::GetOutputNum() const
{
    if (modelDesc_ == nullptr) {
        return 0;
    }
    return svp_acl_mdl_get_num_outputs(modelDesc_);
}

// 新增：获取指定索引输入的参数（大小、stride、维度）
Result ModelProcess::GetInputStrideParam(int index, size_t& buf_size, size_t& stride, svp_acl_mdl_io_dims& dims) const {
    if (modelDesc_ == nullptr || index < 0 || static_cast<size_t>(index) >= GetInputNum()) {
        ERROR_LOG("Invalid input index or model desc");
        return FAILED;
    }

    // 获取输入维度
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, index, &dims);
    if (ret != SVP_ACL_SUCCESS) { ERROR_LOG("Get input dims failed"); return FAILED; }

    // 获取输入大小（单batch）
    buf_size = svp_acl_mdl_get_input_size_by_index(modelDesc_, index);
    // 获取输入默认stride
    stride = svp_acl_mdl_get_input_default_stride(modelDesc_, index);

    return SUCCESS;
}


// 新增：多输入数据创建实现
Result ModelProcess::CreateInputFromData(const std::vector<const void*>& input_datas, 
                                         const std::vector<size_t>& input_sizes) {
    // 初始化输入数据集
    if (input_ != nullptr) { DestroyInput(); }
    input_ = svp_acl_mdl_create_dataset();
    if (input_ == nullptr) {
        ERROR_LOG("Create input dataset failed");
        for (size_t i = 0; i < input_datas.size(); ++i) {
            if (input_datas[i] != nullptr) {
                svp_acl_rt_free(const_cast<void*>(input_datas[i]));
            }
        }
        return FAILED;
    }

    // 为每个输入创建缓冲区并绑定数据
    for (size_t i = 0; i < input_datas.size(); ++i) {
        size_t buf_size = 0;
        size_t stride = 0;
        svp_acl_mdl_io_dims dims;

        // 获取当前输入的参数
        if (GetInputStrideParam(i, buf_size, stride, dims) != SUCCESS) {
            ERROR_LOG("Get input %zu param failed", i);
            DestroyInput();
            for (size_t rest = i; rest < input_datas.size(); ++rest) {
                if (input_datas[rest] != nullptr) {
                    svp_acl_rt_free(const_cast<void*>(input_datas[rest]));
                }
            }
            return FAILED;
        }
        if (i >= input_sizes.size() || input_sizes[i] > buf_size) {
            ERROR_LOG("Input %zu size mismatch, input=%zu expected<=%zu", i, i < input_sizes.size() ? input_sizes[i] : 0U, buf_size);
            DestroyInput();
            for (size_t rest = i; rest < input_datas.size(); ++rest) {
                if (input_datas[rest] != nullptr) {
                    svp_acl_rt_free(const_cast<void*>(input_datas[rest]));
                }
            }
            return FAILED;
        }

        // 创建数据缓冲区（绑定输入数据）
        svp_acl_data_buffer* input_buf = svp_acl_create_data_buffer(
            const_cast<void*>(input_datas[i]), buf_size, stride);
        if (input_buf == nullptr) {
            ERROR_LOG("Create input %zu buffer failed", i);
            DestroyInput();
            for (size_t rest = i; rest < input_datas.size(); ++rest) {
                if (input_datas[rest] != nullptr) {
                    svp_acl_rt_free(const_cast<void*>(input_datas[rest]));
                }
            }
            return FAILED;
        }

        // 将缓冲区添加到输入数据集
        if (svp_acl_mdl_add_dataset_buffer(input_, input_buf) != SVP_ACL_SUCCESS) {
            ERROR_LOG("Add input %zu buffer to dataset failed", i);
            svp_acl_destroy_data_buffer(input_buf);
            DestroyInput();
            for (size_t rest = i; rest < input_datas.size(); ++rest) {
                if (input_datas[rest] != nullptr) {
                    svp_acl_rt_free(const_cast<void*>(input_datas[rest]));
                }
            }
            return FAILED;
        }
    }

    return SUCCESS;
}

// 修改：销毁输入（适配多输入缓冲区释放）
void ModelProcess::DestroyInput() {
    if (input_ == nullptr) return;

    // 遍历所有输入缓冲区，释放内存并销毁缓冲区对象
    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(input_); ++i) {
        svp_acl_data_buffer* buf = svp_acl_mdl_get_dataset_buffer(input_, i);
        if (buf != nullptr) {
            void* data_addr = svp_acl_get_data_buffer_addr(buf);
            if (data_addr != nullptr) {
                svp_acl_rt_free(data_addr); // 释放输入数据内存
            }
            svp_acl_destroy_data_buffer(buf); // 销毁缓冲区对象
        }
    }

    svp_acl_mdl_destroy_dataset(input_);
    input_ = nullptr;
}

Result ModelProcess::CreateInputFromData(const void* data, size_t data_size) 
{
    // 1. 初始化输入数据集
    if (InitInput() != SUCCESS) {
        ERROR_LOG("Init input failed");
        return FAILED;
    }

    // 2. 获取模型输入的 stride 和维度信息（复用原逻辑）
    size_t bufSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims dims;
    if (GetInputStrideParam(0, bufSize, stride, dims) != SUCCESS) {  // 假设单输入
        ERROR_LOG("Get input param failed");
        return FAILED;
    }

    // 3. 分配设备内存并复制数据（替代原文件读取步骤）
    void* device_buf = nullptr;
    svp_acl_error ret = svp_acl_rt_malloc(&device_buf, bufSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("Malloc device buffer failed");
        return FAILED;
    }

    // 初始化设备内存并复制输入数据
    Utils::InitData(static_cast<int8_t*>(device_buf), bufSize);
    memcpy(device_buf, data, data_size);  // 将内存数据复制到设备缓冲区

    // 4. 创建模型输入缓冲区
    return CreateInput(device_buf, bufSize, stride);
}

Result ModelProcess::GetPackedOutputData(
    size_t index,
    std::vector<uint8_t>& packed,
    std::vector<int64_t>& dims,
    uint32_t& elem_type) const
{
    packed.clear();
    dims.clear();
    elem_type = WORKER_ELEM_UNKNOWN;

    if (output_ == nullptr || modelDesc_ == nullptr) {
        ERROR_LOG("output dataset is null");
        return FAILED;
    }
    if (index >= svp_acl_mdl_get_dataset_num_buffers(output_)) {
        ERROR_LOG("output index %zu out of range", index);
        return FAILED;
    }

    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, index);
    if (dataBuffer == nullptr) {
        ERROR_LOG("output[%zu] dataBuffer nullptr invalid", index);
        return FAILED;
    }

    uint8_t* outData = static_cast<uint8_t*>(svp_acl_get_data_buffer_addr(dataBuffer));
    size_t outSize = svp_acl_get_data_buffer_size(dataBuffer);
    if (outData == nullptr || outSize == 0) {
        ERROR_LOG("output[%zu] data invalid, size=%zu", index, outSize);
        return FAILED;
    }

    size_t bufSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims ioDims;
    if (GetOutputStrideParam(static_cast<int>(index), bufSize, stride, ioDims) != SUCCESS) {
        ERROR_LOG("Get output %zu stride param failed", index);
        return FAILED;
    }
    if (ioDims.dim_count == 0) {
        ERROR_LOG("output[%zu] dims invalid", index);
        return FAILED;
    }

    svp_acl_data_type dataType = svp_acl_mdl_get_output_data_type(modelDesc_, index);
    const size_t elemByteSize = svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
    if (elemByteSize == 0) {
        ERROR_LOG("output[%zu] elem byte size invalid", index);
        return FAILED;
    }

    elem_type = MapElementType(dataType);
    if (elem_type == WORKER_ELEM_UNKNOWN) {
        ERROR_LOG("output[%zu] data type unsupported", index);
        return FAILED;
    }

    dims.reserve(ioDims.dim_count);
    uint64_t rowCount = 1;
    for (size_t i = 0; i < ioDims.dim_count; ++i) {
        dims.push_back(ioDims.dims[i]);
        if (i + 1 < ioDims.dim_count) {
            rowCount *= static_cast<uint64_t>(std::max<int64_t>(ioDims.dims[i], 1));
        }
    }

    const uint64_t lastDim = static_cast<uint64_t>(std::max<int64_t>(ioDims.dims[ioDims.dim_count - 1], 0));
    const uint64_t rowBytes = lastDim * elemByteSize;
    const uint64_t totalBytes = rowCount * rowBytes;
    if (totalBytes == 0) {
        packed.clear();
        return SUCCESS;
    }
    if (stride < rowBytes) {
        ERROR_LOG("output[%zu] stride(%zu) smaller than rowBytes(%llu)", index, stride,
            static_cast<unsigned long long>(rowBytes));
        return FAILED;
    }

    packed.resize(static_cast<size_t>(totalBytes));
    for (uint64_t row = 0; row < rowCount; ++row) {
        const size_t srcOffset = static_cast<size_t>(row * stride);
        const size_t dstOffset = static_cast<size_t>(row * rowBytes);
        if (srcOffset + static_cast<size_t>(rowBytes) > outSize || dstOffset + static_cast<size_t>(rowBytes) > packed.size()) {
            ERROR_LOG("output[%zu] row copy overflow detected", index);
            packed.clear();
            dims.clear();
            elem_type = WORKER_ELEM_UNKNOWN;
            return FAILED;
        }
        std::memcpy(&packed[dstOffset], outData + srcOffset, static_cast<size_t>(rowBytes));
    }

    return SUCCESS;
}

void ModelProcess::DumpModelOutputResult() const
{
    stringstream ss;
    size_t outputNum = svp_acl_mdl_get_dataset_num_buffers(output_);
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << executeNum_ << "_" << i << ".bin";
        string outputFileName = ss.str();
        WriteOutput(outputFileName, i);
        ss.str("");
    }
    INFO_LOG("dump data success");
}

void ModelProcess::OutputModelResult() const {
    if (output_ == nullptr || modelDesc_ == nullptr) {
        ERROR_LOG("Output dataset is null, cannot output result");
        return;
    }

    INFO_LOG("Total output count: %zu", GetOutputNum());
    for (size_t i = 0; i < GetOutputNum(); ++i) {
        std::vector<uint8_t> packed;
        std::vector<int64_t> dims;
        uint32_t elemType = WORKER_ELEM_UNKNOWN;
        if (GetPackedOutputData(i, packed, dims, elemType) != SUCCESS) {
            continue;
        }
        std::ostringstream oss;
        oss << "Output[" << i << "] elem_type=" << elemType << " dims=[";
        for (size_t dimIdx = 0; dimIdx < dims.size(); ++dimIdx) {
            if (dimIdx != 0) {
                oss << ", ";
            }
            oss << dims[dimIdx];
        }
        oss << "] bytes=" << packed.size();
        INFO_LOG("%s", oss.str().c_str());
    }
}

void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(output_); ++i) {
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, i);
        void* data = svp_acl_get_data_buffer_addr(dataBuffer);
        (void)svp_acl_rt_free(data);
        (void)svp_acl_destroy_data_buffer(dataBuffer);
    }

    (void)svp_acl_mdl_destroy_dataset(output_);
    output_ = nullptr;
}

Result ModelProcess::Execute()
{
    svp_acl_error ret = svp_acl_mdl_execute(modelId_, input_, output_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }
    executeNum_++;
    return SUCCESS;
}

Result ModelProcess::CreateBuf(int index)
{
    void *bufPtr = nullptr;
    size_t bufSize = 0;
    size_t bufStride = 0;
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = GetInputStrideParam(index, bufSize, bufStride, inDims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetInputStrideParam failed");
        return FAILED;
    }

    ret = svp_acl_rt_malloc(&bufPtr, bufSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu", bufSize);
        return FAILED;
    }
    Utils::InitData(static_cast<int8_t*>(bufPtr), bufSize);

    ret = CreateInput(bufPtr, bufSize, bufStride);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        svp_acl_rt_free(bufPtr);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInputBuf(const string& filePath)
{
    size_t devSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims inputDims;
    // only support single input model
    Result ret = GetInputStrideParam(0, devSize, stride, inputDims);
    if (ret != SUCCESS) {
        ERROR_LOG("GetStrideParam error");
        return FAILED;
    }
    size_t dataSize = GetInputDataSize(0);
    if (dataSize == 0) {
        ERROR_LOG("GetInputDataSize == 0 error");
        return FAILED;
    }
    void *picDevBuffer = Utils::GetDeviceBufferOfFile(filePath, inputDims, stride, dataSize);
    if (picDevBuffer == nullptr) {
        ERROR_LOG("get pic device buffer failed");
        return FAILED;
    }

    ret = InitInput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute InitInput failed");
        svp_acl_rt_free(picDevBuffer);
        return FAILED;
    }

    ret = CreateInput(picDevBuffer, devSize, stride);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        svp_acl_rt_free(picDevBuffer);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateTaskBufAndWorkBuf()
{
    // 2 is stand taskbuf and workbuf
    if (svp_acl_mdl_get_num_inputs(modelDesc_) <= 2) {
        ERROR_LOG("input dataset Num is error.");
        return FAILED;
    }
    size_t datasetSize = svp_acl_mdl_get_dataset_num_buffers(input_);
    if (datasetSize == 0) {
        ERROR_LOG("input dataset Num is 0.");
        return FAILED;
    }
    for (size_t loop = datasetSize; loop < svp_acl_mdl_get_num_inputs(modelDesc_); loop++) {
        Result ret = CreateBuf(loop);
        if (ret != SUCCESS) {
            ERROR_LOG("execute Create taskBuffer and workBuffer failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

void ModelProcess::Unload()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    svp_acl_error ret = svp_acl_mdl_unload(modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelMemPtr_ != nullptr) {
        svp_acl_rt_free(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        svp_acl_rt_free(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}
