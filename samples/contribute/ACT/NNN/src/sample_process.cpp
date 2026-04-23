/**
* @file sample_process.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "sample_process.h"
#include "model_process.h"
#include "acl/acl.h"
#include "utils.h"
#include <chrono> 

using namespace std;

SampleProcess::SampleProcess()
{
}

SampleProcess::~SampleProcess()
{
    // 销毁模型资源
    modelProcess_.DestroyResource(); 
    DestroyResource();
}
Result SampleProcess::InitResource()
{
    // ACL init
    const char* aclConfigPath = "../src/acl.json";
    INFO_LOG("===== Start InitResource: aclInit with config: %s =====", aclConfigPath);
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl init failed, ret = %d", ret);
        return FAILED;
    }
    INFO_LOG("acl init success");

    // set device
    INFO_LOG("===== Set device: %d =====", deviceId_);
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl open device %d failed, ret = %d", deviceId_, ret);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);
    // create context (set current)
    INFO_LOG("===== Create context for device: %d =====", deviceId_);
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create context failed, ret = %d", ret);
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    INFO_LOG("===== Create stream =====");
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed, ret = %d", ret);
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    INFO_LOG("===== Get run mode =====");
    aclrtRunMode runMode;
    ret = aclrtGetRunMode(&runMode);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed, ret = %d", ret);
        return FAILED;
    }
    INFO_LOG("run mode get success, runMode = %d (ACL_DEVICE=0, ACL_HOST=1)", runMode);
    if (runMode != ACL_DEVICE) {
        ERROR_LOG("acl run mode failed, expect ACL_DEVICE(0), got %d", runMode);
        return FAILED;
    }
    isInited_ = true;
    INFO_LOG("===== InitResource success =====");
    return SUCCESS;
}

// 新增模型加载方法，只执行一次
Result SampleProcess::LoadModel() {
    INFO_LOG("===== Start LoadModel, isModelLoaded_ = %d =====", isModelLoaded_);
    if (isModelLoaded_) {
        INFO_LOG("Model already loaded, skip");
        return SUCCESS;
    }

    const string omModelPath = "../model/act_model.om";
    INFO_LOG("Load model from path: %s", omModelPath.c_str());
    Result ret = modelProcess_.LoadModelFromFileWithMem(omModelPath.c_str());
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed, ret = %d (SUCCESS=0, FAILED=1)", ret);
        return FAILED;
    }
    INFO_LOG("LoadModelFromFileWithMem success");

    INFO_LOG("Create model desc");
    ret = modelProcess_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed, ret = %d", ret);
        return FAILED;
    }
    INFO_LOG("CreateDesc success");

    INFO_LOG("Create model output buffer");
    ret = modelProcess_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed, ret = %d", ret);
        return FAILED;
    }
    INFO_LOG("CreateOutput success");

    isModelLoaded_ = true;
    INFO_LOG("===== LoadModel success, isModelLoaded_ = %d =====", isModelLoaded_);
    return SUCCESS;
}

// 修改Process方法，只处理单次推理
Result SampleProcess::Process() {
    if (!isInited_ || !isModelLoaded_) {
        ERROR_LOG("Resource or model not initialized");
        return FAILED;
    }

    // 创建输入（使用已加载的模型）
    Result ret = modelProcess_.CreateInputFromData(input_datas_, input_sizes_);
    if (ret != SUCCESS) { 
        ERROR_LOG("Create multi-input failed"); 
        return FAILED; 
    }

    ret = modelProcess_.CreateTaskBufAndWorkBuf();
    if (ret != SUCCESS) {
        ERROR_LOG("CreateTaskBufAndWorkBuf failed");
        return FAILED;
    }

    // 记录推理开始时间
    auto start = std::chrono::high_resolution_clock::now();

    ret = modelProcess_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("execute inference failed");
        modelProcess_.DestroyInput();
        return FAILED;
    }

    // 记录推理结束时间
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "INFERENCE_TIME:" << elapsed_ms << std::endl;

    // 输出结果
    modelProcess_.OutputModelResult();
    modelProcess_.DumpModelOutputResult();

    // 释放当前输入缓冲区（保留模型资源）
    modelProcess_.DestroyInput();

    return SUCCESS;
}

// 新增：保存输入文件路径
void SampleProcess::SetInputPath(const std::string& path) {
    this->input_path_ = path;  // 在类中新增私有成员变量input_path_
}

void SampleProcess::SetInputDatas(const std::vector<const void*>& input_datas, 
                                   const std::vector<size_t>& input_sizes) {
    input_datas_ = input_datas;
    input_sizes_ = input_sizes;
}

void SampleProcess::DestroyResource()
{
    aclError ret;
    // 1. 先销毁流
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    // 2. 再销毁上下文
    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    // 3. 重置设备（确保流和上下文已销毁）
    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device %d", deviceId_);

    // 4. 最后执行finalize（全局只执行一次）
    static bool isFinalized = false;
    if (!isFinalized) {
        ret = aclFinalize();
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("finalize acl failed");
        } else {
            isFinalized = true;
        }
    }
    INFO_LOG("end to finalize acl");
}
