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

#include <map>
#include <sstream>
#include <fstream>
#include <cstring> 
#include "utils.h"

using namespace std;

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit
static const size_t FIXED_STRIDE = 256; // 固定stride为256字节
static const size_t LAST_DIM = 8;       // 每个对齐块的有效float数
static const size_t TOTAL_EFFECTIVE_FLOAT = 800; // 总有效float数

ModelProcess::ModelProcess()
{
}

ModelProcess::~ModelProcess()
{
    Unload();
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
}

void ModelProcess::DestroyResource()
{
    Unload();
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
}

Result ModelProcess::LoadModelFromFileWithMem(const std::string& modelPath)
{
    uint32_t fileSize = 0;
    modelMemPtr_ = Utils::ReadBinFile(modelPath, fileSize);
    modelMemSize_ = fileSize;
    aclError ret = aclmdlLoadFromMem(static_cast<uint8_t* >(modelMemPtr_), modelMemSize_, &modelId_);
    if (ret != ACL_SUCCESS) {
        aclrtFree(modelMemPtr_);
        ERROR_LOG("load model from file failed, model file is %s", modelPath.c_str());
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath.c_str());
    return SUCCESS;
}

Result ModelProcess::CreateDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::InitInput()
{
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize, int)
{
    aclDataBuffer* inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed");
        aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }

    return SUCCESS;
}

size_t ModelProcess::GetInputDataSize(int index) const
{
    aclDataType dataType = aclmdlGetInputDataType(modelDesc_, index);
    return aclDataTypeSize(dataType) / BYTE_BIT_NUM;
}

size_t ModelProcess::GetOutputDataSize(int index) const
{
    aclDataType dataType = aclmdlGetOutputDataType(modelDesc_, index);
    return aclDataTypeSize(dataType) / BYTE_BIT_NUM;
}

Result ModelProcess::GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, aclmdlIODims& dims) const
{
    aclError ret = aclmdlGetOutputDims(modelDesc_, index, &dims);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlGetOutputDims error!");
        return FAILED;
    }

    // 固定stride为256
    stride = FIXED_STRIDE;
    
    bufSize = aclmdlGetOutputSizeByIndex(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("aclmdlGetOutputSizeByIndex error!");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateOutput()
{
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t stride = FIXED_STRIDE;

        if (stride == 0) {
            ERROR_LOG("Error, output default stride is %lu.", stride);
            return FAILED;
        }
        
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        if (bufferSize == 0) {
            ERROR_LOG("Error, output size is %lu.", bufferSize);
            return FAILED;
        }

        void *outputBuffer = nullptr;
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", bufferSize);
            return FAILED;
        }
        Utils::InitData(static_cast<int8_t*>(outputBuffer), bufferSize);

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            aclrtFree(outputBuffer);
            return FAILED;
        }
        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(outputBuffer);
            aclDestroyDataBuffer(outputData);
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
    aclmdlIODims dims;
    aclError ret = GetOutputStrideParam(index, bufSize, bufStride, dims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetOutputStrideParam failed");
        return FAILED;
    }
    if ((bufSize == 0) || (bufStride == 0)) {
        ERROR_LOG("Error, bufSize(%zu) bufStride(%zu) invalid", bufSize, bufStride);
        return FAILED;
    }
    if ((dims.dimCount == 0) || (dims.dims[dims.dimCount - 1] <= 0)) {
        ERROR_LOG("Error, dims para invalid");
        return FAILED;
    }
    int64_t lastDim = dims.dims[dims.dimCount - 1];

    size_t dataSize = GetOutputDataSize(index);
    if (dataSize == 0) {
        ERROR_LOG("Error, dataSize == 0 invalid");
        return FAILED;
    }
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = bufSize / bufStride;
    size_t invalidSize = bufStride - lastDimSize;
    if (invalidSize == 0) {
        return SUCCESS;
    }

    for (size_t i = 0; i < loopNum; ++i) {
        size_t offset = bufStride * i + lastDimSize;
        int8_t* ptr = &buffer[offset];
        for (size_t idx = 0; idx < invalidSize; idx++) {
            ptr[idx] = 0;
        }
    }
    return SUCCESS;
}

void ModelProcess::WriteOutput(const string& outputFileName, size_t index) const
{
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, index);
    if (dataBuffer == nullptr) {
        ERROR_LOG("output[%zu] dataBuffer nullptr invalid", index);
        return;
    }
    int8_t* outData = (int8_t*)aclGetDataBufferAddr(dataBuffer);
    size_t outSize = aclGetDataBufferSize(dataBuffer);
    if (outData == nullptr || outSize == 0) {
        ERROR_LOG("output[%zu] data or size(%zu) invalid", index, outSize);
        return;
    }

    std::vector<float> effectiveData;
    effectiveData.reserve(TOTAL_EFFECTIVE_FLOAT);
    float* floatData = reinterpret_cast<float*>(outData);
    const size_t BLOCK_STEP = FIXED_STRIDE / sizeof(float); // 每个块跳64个float位置

    size_t validCount = 0;
    size_t blockIndex = 0;
    while (validCount < TOTAL_EFFECTIVE_FLOAT) {
        size_t blockStart = blockIndex * BLOCK_STEP;
        for (size_t j = 0; j < LAST_DIM && validCount < TOTAL_EFFECTIVE_FLOAT; ++j) {
            effectiveData.push_back(floatData[blockStart + j]);
            validCount++;
        }
        blockIndex++;
    }

    ofstream fout(outputFileName, ios::out|ios::binary);
    if (fout.good() == false) {
        ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
        return;
    }
    fout.write((char*)&effectiveData[0], effectiveData.size() * sizeof(float));
    fout.close();
    INFO_LOG("Write %zu effective float to %s", effectiveData.size(), outputFileName.c_str());
    return;
}

// 新增：获取模型输入个数
size_t ModelProcess::GetInputNum() const {
    if (modelDesc_ == nullptr) return 0;
    return aclmdlGetNumInputs(modelDesc_);
}

Result ModelProcess::GetInputStrideParam(int index, size_t& buf_size, size_t& stride, aclmdlIODims& dims) const {
    if (modelDesc_ == nullptr || index < 0 || static_cast<size_t>(index) >= GetInputNum()) {
        ERROR_LOG("Invalid input index or model desc");
        return FAILED;
    }

    // 获取输入维度
    aclError ret = aclmdlGetInputDims(modelDesc_, index, &dims);
    if (ret != ACL_SUCCESS) { 
        ERROR_LOG("Get input dims failed"); 
        return FAILED; 
    }

    // 获取输入大小（单batch）
    buf_size = aclmdlGetInputSizeByIndex(modelDesc_, index);
    
    // 固定stride为256
    stride = FIXED_STRIDE;

    return SUCCESS;
}

Result ModelProcess::CreateInputFromData(const std::vector<const void*>& input_datas, 
                                         const std::vector<size_t>& input_sizes) {
    // 初始化输入数据集
    if (input_ != nullptr) { DestroyInput(); }
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) { ERROR_LOG("Create input dataset failed"); return FAILED; }

    // 为每个输入创建缓冲区并绑定数据
    for (size_t i = 0; i < input_datas.size(); ++i) {
        size_t buf_size = 0;
        size_t stride = 0;
        aclmdlIODims dims;

        // 获取当前输入的参数
        if (GetInputStrideParam(i, buf_size, stride, dims) != SUCCESS) {
            ERROR_LOG("Get input %zu param failed", i);
            return FAILED;
        }

        aclDataBuffer* input_buf = aclCreateDataBuffer(
            const_cast<void*>(input_datas[i]), buf_size);
        if (input_buf == nullptr) {
            ERROR_LOG("Create input %zu buffer failed", i);
            return FAILED;
        }

        // 将缓冲区添加到输入数据集
        if (aclmdlAddDatasetBuffer(input_, input_buf) != ACL_SUCCESS) {
            ERROR_LOG("Add input %zu buffer to dataset failed", i);
            aclDestroyDataBuffer(input_buf);
            return FAILED;
        }
    }

    return SUCCESS;
}

// 修改：销毁输入（适配多输入缓冲区释放）
void ModelProcess::DestroyInput() {
    if (input_ == nullptr) return;

    // 遍历所有输入缓冲区，释放内存并销毁缓冲区对象
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        aclDataBuffer* buf = aclmdlGetDatasetBuffer(input_, i);
        if (buf != nullptr) {
            void* data_addr = aclGetDataBufferAddr(buf);
            if (data_addr != nullptr) {
                aclrtFree(data_addr); // 释放输入数据内存
            }
            aclDestroyDataBuffer(buf); // 销毁缓冲区对象
        }
    }

    aclmdlDestroyDataset(input_);
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
    aclmdlIODims dims;
    if (GetInputStrideParam(0, bufSize, stride, dims) != SUCCESS) {  // 假设单输入
        ERROR_LOG("Get input param failed");
        return FAILED;
    }

    // 3. 分配设备内存并复制数据（替代原文件读取步骤）
    void* device_buf = nullptr;
    aclError ret = aclrtMalloc(&device_buf, bufSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("Malloc device buffer failed");
        return FAILED;
    }

    // 初始化设备内存并复制输入数据
    Utils::InitData(static_cast<int8_t*>(device_buf), bufSize);
    memcpy(device_buf, data, data_size);  // 将内存数据复制到设备缓冲区

    return CreateInput(device_buf, bufSize, 0);
}

void ModelProcess::DumpModelOutputResult() const
{
    stringstream ss;
    size_t outputNum = aclmdlGetDatasetNumBuffers(output_);
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << executeNum_ << "_" << i << ".bin";
        string outputFileName = ss.str();
        WriteOutput(outputFileName, i);
        ss.str("");
    }
    INFO_LOG("dump data success");
}

// 核心修改：按SVP逻辑读取800个有效float并打印
void ModelProcess::OutputModelResult() const {
    if (output_ == nullptr) {
        ERROR_LOG("Output dataset is null, cannot output result");
        return;
    }

    // 获取输出数量
    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    INFO_LOG("Total output count: %zu", outputNum);

    // 遍历每个输出
    for (size_t i = 1; i < outputNum; ++i) {
        // 获取当前输出缓冲区
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        if (dataBuffer == nullptr) {
            ERROR_LOG("Output[%zu] buffer is null", i);
            continue;
        }

        // 获取输出数据地址
        int8_t* outputData = static_cast<int8_t*>(aclGetDataBufferAddr(dataBuffer));
        if (outputData == nullptr) {
            ERROR_LOG("Output[%zu] data is invalid", i);
            continue;
        }

        // 打印当前输出的基本信息
        INFO_LOG("\nOutput[%zu] (SVP style, only 800 valid float):", i);
        INFO_LOG("----------------------------------------");

        // 转换为float指针
        float* floatData = reinterpret_cast<float*>(outputData);
        // 每个对齐块跳64个float位置（256字节 ÷ 4字节/float）
        const size_t BLOCK_STEP = FIXED_STRIDE / sizeof(float);
        size_t validCount = 0;
        size_t blockIndex = 0;
        std::cout << "FLOAT_OUTPUT_START " << i << " " << TOTAL_EFFECTIVE_FLOAT << std::endl;
        
        while (validCount < TOTAL_EFFECTIVE_FLOAT) {
            // 计算当前块的起始位置
            size_t blockStart = blockIndex * BLOCK_STEP;
            // 读取当前块的前8个有效float
            for (size_t j = 0; j < LAST_DIM && validCount < TOTAL_EFFECTIVE_FLOAT; ++j) {
                std::cout << floatData[blockStart + j] << " ";
                validCount++;
            }
            // 跳到下一个块
            blockIndex++;
        }

        std::cout << std::endl << "FLOAT_OUTPUT_END " << i << std::endl;
    }
}

void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
}

Result ModelProcess::Execute()
{
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }
    executeNum_++;
    INFO_LOG("model execute success");
    return SUCCESS;
}

Result ModelProcess::CreateBuf(int index)
{
    void *bufPtr = nullptr;
    size_t bufSize = 0;
    size_t bufStride = 0;
    aclmdlIODims inDims;
    aclError ret = GetInputStrideParam(index, bufSize, bufStride, inDims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetInputStrideParam failed");
        return FAILED;
    }

    ret = aclrtMalloc(&bufPtr, bufSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu", bufSize);
        return FAILED;
    }
    Utils::InitData(static_cast<int8_t*>(bufPtr), bufSize);

    ret = CreateInput(bufPtr, bufSize, 0);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        aclrtFree(bufPtr);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInputBuf(const string& filePath)
{
    size_t devSize = 0;
    size_t stride = 0;
    aclmdlIODims inputDims;
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
        aclrtFree(picDevBuffer);
        return FAILED;
    }

    // stride不传入 aclCreateDataBuffer
    ret = CreateInput(picDevBuffer, devSize, 0);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        aclrtFree(picDevBuffer);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateTaskBufAndWorkBuf()
{
    // 2 is stand taskbuf and workbuf
    if (aclmdlGetNumInputs(modelDesc_) <= 2) {
        ERROR_LOG("input dataset Num is error.");
        return FAILED;
    }
    size_t datasetSize = aclmdlGetDatasetNumBuffers(input_);
    if (datasetSize == 0) {
        ERROR_LOG("input dataset Num is 0.");
        return FAILED;
    }
    for (size_t loop = datasetSize; loop < aclmdlGetNumInputs(modelDesc_); loop++) {
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

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}
