/**
* @file sample_process.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "sample_process.h"
#include "model_process.h"
#include "acl/svp_acl.h"
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
    svp_acl_error ret = svp_acl_init(aclConfigPath);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // set device
    ret = svp_acl_rt_set_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // set no timeout
    ret = svp_acl_rt_set_op_wait_timeout(0);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl set op wait time failed");
        return FAILED;
    }
    INFO_LOG("set op wait time success");

    // create context (set current)
    ret = svp_acl_rt_create_context(&context_, deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = svp_acl_rt_create_stream(&stream_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    svp_acl_rt_run_mode runMode;
    ret = svp_acl_rt_get_run_mode(&runMode);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    if (runMode != SVP_ACL_DEVICE) {
        ERROR_LOG("acl run mode failed");
        return FAILED;
    }
    INFO_LOG("get run mode success");
    isInited_ = true;
    return SUCCESS;
}

// 新增模型加载方法，只执行一次
Result SampleProcess::LoadModel() {
    if (isModelLoaded_) {
        return SUCCESS;
    }

    const string omModelPath = "../model/act_distill_fp32_for_mindcmd_simp_release.om";
    Result ret = modelProcess_.LoadModelFromFileWithMem(omModelPath.c_str());
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    ret = modelProcess_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    ret = modelProcess_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    isModelLoaded_ = true;
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
    svp_acl_error ret;
    // 1. 先销毁流
    if (stream_ != nullptr) {
        ret = svp_acl_rt_destroy_stream(stream_);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    // 2. 再销毁上下文
    if (context_ != nullptr) {
        ret = svp_acl_rt_destroy_context(context_);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    // 3. 重置设备（确保流和上下文已销毁）
    ret = svp_acl_rt_reset_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device %d", deviceId_);

    // 4. 最后执行finalize（全局只执行一次）
    static bool isFinalized = false;
    if (!isFinalized) {
        ret = svp_acl_finalize();
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("finalize acl failed");
        } else {
            isFinalized = true;
        }
    }
    INFO_LOG("end to finalize acl");
}
