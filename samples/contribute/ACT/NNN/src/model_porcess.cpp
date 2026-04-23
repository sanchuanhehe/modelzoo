/**
* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

// 析构函数：对象销毁时自动调用，释放模型相关所有资源
ModelProcess::~ModelProcess()
{
    Unload();          // 卸载模型
    DestroyDesc();     // 销毁模型描述信息
    DestroyInput();    // 销毁模型输入数据集
    DestroyOutput();   // 销毁模型输出数据集
}

// 手动销毁模型资源接口，功能与析构函数一致
void ModelProcess::DestroyResource()
{
    Unload();          // 卸载模型
    DestroyDesc();     // 销毁模型描述信息
    DestroyInput();    // 销毁模型输入数据集
    DestroyOutput();   // 销毁模型输出数据集
}

// 从文件加载模型到内存，并加载到昇腾AI处理器
// modelPath：模型文件路径
Result ModelProcess::LoadModelFromFileWithMem(const std::string& modelPath)
{
    uint32_t fileSize = 0;                      // 定义变量存储模型文件大小
    modelMemPtr_ = Utils::ReadBinFile(modelPath, fileSize);  // 读取二进制模型文件到内存
    modelMemSize_ = fileSize;                    // 记录模型内存大小

    // 从内存中加载模型到AI处理器，获取模型ID
    aclError ret = aclmdlLoadFromMem(static_cast<uint8_t* >(modelMemPtr_), modelMemSize_, &modelId_);
    if (ret != ACL_SUCCESS) {                    // 判断模型加载是否失败
        aclrtFree(modelMemPtr_);                 // 加载失败，释放已申请的模型内存
        ERROR_LOG("load model from file failed, model file is %s", modelPath.c_str());  // 打印错误日志
        return FAILED;                           // 返回失败状态
    }

    loadFlag_ = true;                            // 模型加载成功，置位加载标志
    INFO_LOG("load model %s success", modelPath.c_str());  // 打印成功日志
    return SUCCESS;                              // 返回成功状态
}

// 创建模型描述信息，用于获取模型输入输出等属性
Result ModelProcess::CreateDesc()
{
    // 创建模型描述句柄
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {                 // 判断创建是否失败
        ERROR_LOG("create model description failed");  // 打印错误日志
        return FAILED;                           // 返回失败状态
    }

    // 根据模型ID，获取模型详细描述信息
    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {                    // 判断获取描述信息是否失败
        ERROR_LOG("get model description failed");  // 打印错误日志
        return FAILED;                           // 返回失败状态
    }

    INFO_LOG("create model description success");  // 打印成功日志
    return SUCCESS;                              // 返回成功状态
}

// 销毁模型描述信息，释放资源
void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {                // 判断模型描述句柄是否存在
        (void)aclmdlDestroyDesc(modelDesc_);     // 销毁模型描述
        modelDesc_ = nullptr;                    // 指针置空，防止野指针
    }
}

// 初始化模型输入数据集（只创建空数据集，不添加数据）
Result ModelProcess::InitInput()
{
    // 创建ACL数据集，用于存放模型输入
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {                     // 判断创建是否失败
        ERROR_LOG("can't create dataset, create input failed");  // 打印错误日志
        return FAILED;                           // 返回失败状态
    }
    return SUCCESS;                              // 返回成功状态
}

// 为模型输入数据集添加具体数据
// inputDataBuffer：输入数据内存地址
// bufferSize：输入数据大小
Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize, int)
{
    // 根据输入数据和大小，创建ACL数据缓冲
    aclDataBuffer* inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {                  // 判断创建缓冲是否失败
        ERROR_LOG("can't create data buffer, create input failed");  // 打印错误日志
        return FAILED;                           // 返回失败状态
    }

    // 将数据缓冲添加到输入数据集
    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {                    // 判断添加是否失败
        ERROR_LOG("add input dataset buffer failed");  // 打印错误日志
        aclDestroyDataBuffer(inputData);         // 销毁已创建的数据缓冲
        inputData = nullptr;                     // 指针置空
        return FAILED;                           // 返回失败状态
    }

    return SUCCESS;                              // 返回成功状态
}

// 获取模型指定输入端口的数据类型大小（单位：字节）
// index：输入端口索引
size_t ModelProcess::GetInputDataSize(int index) const
{
    // 获取模型输入数据类型
    aclDataType dataType = aclmdlGetInputDataType(modelDesc_, index);
    // 计算数据类型占用字节数并返回
    return aclDataTypeSize(dataType) / BYTE_BIT_NUM;
}

// 获取模型指定输出端口的数据类型大小（单位：字节）
// index：输出端口索引
size_t ModelProcess::GetOutputDataSize(int index) const
{
    // 获取模型输出数据类型
    aclDataType dataType = aclmdlGetOutputDataType(modelDesc_, index);
    // 计算数据类型占用字节数并返回
    return aclDataTypeSize(dataType) / BYTE_BIT_NUM;
}

// 获取模型输出的步长、缓冲区大小、维度信息
// index：输出端口索引
// bufSize：输出参数，存储输出缓冲区大小
// stride：输出参数，存储输出步长
// dims：输出参数，存储输出维度信息
Result ModelProcess::GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, aclmdlIODims& dims) const
{
    // 获取模型指定输出端口的维度信息
    aclError ret = aclmdlGetOutputDims(modelDesc_, index, &dims);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("aclmdlGetOutputDims error!");
        return FAILED;
    }

    // 固定stride为256
    stride = FIXED_STRIDE;
    
    // 获取模型指定输出端口的缓冲区大小
    bufSize = aclmdlGetOutputSizeByIndex(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("aclmdlGetOutputSizeByIndex error!");
        return FAILED;
    }
    return SUCCESS;
}

// 创建模型输出数据集，并为每个输出分配内存
Result ModelProcess::CreateOutput()
{
    // 创建输出数据集
    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    // 获取模型输出端口数量
    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    // 遍历所有输出端口，依次创建输出内存和数据缓冲
    for (size_t i = 0; i < outputSize; ++i) {
        size_t stride = FIXED_STRIDE;  // 使用固定步长

        if (stride == 0) {
            ERROR_LOG("Error, output default stride is %lu.", stride);
            return FAILED;
        }
        
        // 获取当前输出端口需要的内存大小
        size_t bufferSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        if (bufferSize == 0) {
            ERROR_LOG("Error, output size is %lu.", bufferSize);
            return FAILED;
        }

        void *outputBuffer = nullptr;
        // 在设备上申请输出内存
        aclError ret = aclrtMalloc(&outputBuffer, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", bufferSize);
            return FAILED;
        }
        // 初始化输出内存数据
        Utils::InitData(static_cast<int8_t*>(outputBuffer), bufferSize);

        // 根据输出内存创建数据缓冲
        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, bufferSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            aclrtFree(outputBuffer);  // 创建失败，释放已申请内存
            return FAILED;
        }

        // 将输出数据缓冲添加到输出数据集
        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed");
            aclrtFree(outputBuffer);        // 释放内存
            aclDestroyDataBuffer(outputData);  // 销毁数据缓冲
            return FAILED;
        }
    }

    INFO_LOG("create model output success");  // 输出创建成功日志
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
    // 获取指定输入索引对应的输入数据大小（单batch大小）
    buf_size = aclmdlGetInputSizeByIndex(modelDesc_, index);
    
    // 业务自定义配置：将内存对齐步长固定为256字节
    stride = FIXED_STRIDE;

    // 函数执行成功，返回成功状态
    return SUCCESS;
}

// 从外部输入数据数组中，创建并填充模型的输入数据集
// input_datas：外部输入数据的指针数组，每个元素对应一个输入端口的数据地址
// input_sizes：外部输入数据的大小数组，每个元素对应一个输入端口的数据字节大小
Result ModelProcess::CreateInputFromData(const std::vector<const void*>& input_datas, 
                                         const std::vector<size_t>& input_sizes) {
    // 若当前已存在输入数据集，先销毁旧的输入资源，避免内存泄漏和重复创建
    if (input_ != nullptr) { DestroyInput(); }
    
    // 创建ACL框架的模型输入数据集，用于承载所有模型输入数据
    input_ = aclmdlCreateDataset();
    
    // 判断输入数据集是否创建失败
    if (input_ == nullptr) { 
        // 打印错误日志
        ERROR_LOG("Create input dataset failed"); 
        // 返回失败状态
        return FAILED; 
    }
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
    // 3. 分配设备内存并复制数据
    // 定义设备端内存指针，初始化为空
    void* device_buf = nullptr;
    // 调用ACL接口申请设备内存，指定内存大小与分配类型
    aclError ret = aclrtMalloc(&device_buf, bufSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    // 判断设备内存申请是否失败
    if (ret != ACL_SUCCESS) {
        // 打印错误日志
        ERROR_LOG("Malloc device buffer failed");
        // 返回失败状态
        return FAILED;
    }

    // 初始化设备内存数据，将内存置为初始值
    Utils::InitData(static_cast<int8_t*>(device_buf), bufSize);
    // 将主机端的输入数据拷贝到已申请的设备内存中
    memcpy(device_buf, data, data_size);  // 将内存数据复制到设备缓冲区

    // 调用CreateInput接口，将设备内存封装为模型输入，完成输入创建
    return CreateInput(device_buf, bufSize, 0);
}

// 转储模型推理输出结果到本地二进制文件
void ModelProcess::DumpModelOutputResult() const
{
    // 定义字符串流，用于拼接输出文件名
    stringstream ss;
    // 获取模型输出数据集的缓冲区数量
    size_t outputNum = aclmdlGetDatasetNumBuffers(output_);
    // 遍历所有输出缓冲区
    for (size_t i = 0; i < outputNum; ++i) {
        // 拼接输出文件名：output + 推理次数 + 输出索引
        ss << "output" << executeNum_ << "_" << i << ".bin";
        // 将字符串流转换为字符串文件名
        string outputFileName = ss.str();
        // 调用WriteOutput将当前输出数据写入文件
        WriteOutput(outputFileName, i);
        // 清空字符串流，准备下一次拼接
        ss.str("");
    }
    // 打印数据转储成功日志
    INFO_LOG("dump data success");
}

// 读取800个有效float数据并打印
void ModelProcess::OutputModelResult() const {
    // 判断输出数据集是否为空，为空则无法处理
    if (output_ == nullptr) {
        // 打印错误日志
        ERROR_LOG("Output dataset is null, cannot output result");
        // 直接返回
        return;
    }

    // 从模型描述中获取模型总输出端口数量
    size_t outputNum = aclmdlGetNumOutputs(modelDesc_);
    // 打印总输出端口数量日志
    INFO_LOG("Total output count: %zu", outputNum);

    // 遍历模型的每个输出端口（从索引1开始）
    for (size_t i = 1; i < outputNum; ++i) {
        // 从输出数据集中获取当前索引对应的输出数据缓冲区
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        // 判断当前输出缓冲区是否为空
        if (dataBuffer == nullptr) {
            // 打印索引对应缓冲区为空的错误日志
            ERROR_LOG("Output[%zu] buffer is null", i);
            // 跳过当前无效输出，继续处理下一个
            continue;
        }

        // 获取输出数据缓冲区的设备内存地址，并转换为int8_t类型指针
        int8_t* outputData = static_cast<int8_t*>(aclGetDataBufferAddr(dataBuffer));
        // 判断输出数据指针是否有效
        if (outputData == nullptr) {
            // 打印数据无效错误日志
            ERROR_LOG("Output[%zu] data is invalid", i);
            // 跳过当前无效输出，继续处理下一个
            continue;
        }

        // 打印当前输出的基本信息
        INFO_LOG("\nOutput[%zu] (only 800 valid float):", i);
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
// 销毁模型输出相关资源（内存、数据集、数据缓冲）
void ModelProcess::DestroyOutput()
{
    // 如果输出数据集为空，直接返回，无需销毁
    if (output_ == nullptr) {
        return;
    }

    // 遍历输出数据集中所有的数据缓冲区，逐个释放资源
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        // 获取当前索引对应的输出数据缓冲区
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        // 获取数据缓冲区对应的设备内存地址
        void* data = aclGetDataBufferAddr(dataBuffer);
        // 释放设备内存（忽略返回值）
        (void)aclrtFree(data);
        // 销毁数据缓冲区（忽略返回值）
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    // 销毁整个输出数据集（忽略返回值）
    (void)aclmdlDestroyDataset(output_);
    // 指针置空，防止野指针
    output_ = nullptr;
}

// 执行模型推理
Result ModelProcess::Execute()
{
    // 调用ACL接口执行模型推理，传入模型ID、输入数据集、输出数据集
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    // 判断推理是否执行失败
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }
    // 推理成功，推理次数自增
    executeNum_++;
    INFO_LOG("model execute success");
    return SUCCESS;
}

// 根据输入索引创建设备内存缓冲区，并封装为模型输入
// index：模型输入端口索引
Result ModelProcess::CreateBuf(int index)
{
    // 定义设备内存指针
    void *bufPtr = nullptr;
    // 缓冲区大小
    size_t bufSize = 0;
    // 内存对齐步长
    size_t bufStride = 0;
    // 输入维度信息
    aclmdlIODims inDims;
    // 获取输入的步长、大小、维度信息
    aclError ret = GetInputStrideParam(index, bufSize, bufStride, inDims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetInputStrideParam failed");
        return FAILED;
    }

    // 为输入数据申请设备内存
    ret = aclrtMalloc(&bufPtr, bufSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu", bufSize);
        return FAILED;
    }
    // 初始化设备内存数据
    Utils::InitData(static_cast<int8_t*>(bufPtr), bufSize);

    // 将申请的设备内存封装为模型输入
    ret = CreateInput(bufPtr, bufSize, 0);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        // 创建输入失败，释放已申请的设备内存
        aclrtFree(bufPtr);
        return FAILED;
    }
    return SUCCESS;
}

// 根据图片文件路径创建模型输入缓冲区
// filePath：输入图片文件路径
Result ModelProcess::CreateInputBuf(const string& filePath)
{
    // 设备内存大小
    size_t devSize = 0;
    // 内存对齐步长
    size_t stride = 0;
    // 模型输入维度
    aclmdlIODims inputDims;
    // 仅支持单输入模型，获取0号输入的参数信息
    Result ret = GetInputStrideParam(0, devSize, stride, inputDims);
    if (ret != SUCCESS) {
        ERROR_LOG("GetStrideParam error");
        return FAILED;
    }
    // 获取输入数据类型大小
    size_t dataSize = GetInputDataSize(0);
    if (dataSize == 0) {
        ERROR_LOG("GetInputDataSize == 0 error");
        return FAILED;
    }
    // 读取文件并加载到设备内存，返回设备内存指针
    void *picDevBuffer = Utils::GetDeviceBufferOfFile(filePath, inputDims, stride, dataSize);
    if (picDevBuffer == nullptr) {
        ERROR_LOG("get pic device buffer failed");
        return FAILED;
    }

    // 初始化输入数据集
    ret = InitInput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute InitInput failed");
        // 初始化失败，释放设备内存
        aclrtFree(picDevBuffer);
        return FAILED;
    }

    // 创建模型输入，stride参数不传入
    ret = CreateInput(picDevBuffer, devSize, 0);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        // 创建失败，释放设备内存
        aclrtFree(picDevBuffer);
        return FAILED;
    }
    return SUCCESS;
}

// 创建任务缓冲区和工作缓冲区（模型额外需要的辅助输入）
Result ModelProcess::CreateTaskBufAndWorkBuf()
{
    // 模型输入数量必须大于2，2代表taskbuf和workbuf之外的业务输入
    if (aclmdlGetNumInputs(modelDesc_) <= 2) {
        ERROR_LOG("input dataset Num is error.");
        return FAILED;
    }
    // 获取当前已创建的输入缓冲区数量
    size_t datasetSize = aclmdlGetDatasetNumBuffers(input_);
    if (datasetSize == 0) {
        ERROR_LOG("input dataset Num is 0.");
        return FAILED;
    }
    // 遍历未创建的额外输入，依次创建任务缓冲区和工作缓冲区
    for (size_t loop = datasetSize; loop < aclmdlGetNumInputs(modelDesc_); loop++) {
        Result ret = CreateBuf(loop);
        if (ret != SUCCESS) {
            ERROR_LOG("execute Create taskBuffer and workBuffer failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

// 卸载模型，释放模型相关所有资源
void ModelProcess::Unload()
{
    // 如果未加载过模型，直接提示并返回
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    // 卸载模型
    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    // 销毁模型描述信息
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    // 释放模型文件内存
    if (modelMemPtr_ != nullptr) {
        aclrtFree(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    // 释放模型权重内存
    if (modelWeightPtr_ != nullptr) {
        aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    // 清除模型加载标志
    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}

