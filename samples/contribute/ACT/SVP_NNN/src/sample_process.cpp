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

#include "sample_process.h"
#include "model_process.h"
#include "acl/svp_acl.h"
#include "utils.h"
#include <chrono> 
#include <cstdlib>
#include <sstream>
#include <unistd.h>

using namespace std;

namespace {

static const size_t kExpectedInputCount = 3;
static const size_t kActionOutputIndex = 2;

}  // namespace

SampleProcess::SampleProcess()
{
}

SampleProcess::~SampleProcess()
{
    DestroyResource();
}

Result SampleProcess::InitResource()
{
    // ACL init
    const char* aclConfigPath = std::getenv("SVP_ACL_CONFIG_PATH");
    if (!(aclConfigPath != nullptr && aclConfigPath[0] != '\0' && access(aclConfigPath, R_OK) == 0)) {
        aclConfigPath = nullptr;
    }
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

    const auto load_start = std::chrono::high_resolution_clock::now();

    const char* modelPathEnv = std::getenv("SVP_MODEL_PATH");
    const std::string omModelPath = (modelPathEnv != nullptr && modelPathEnv[0] != '\0')
        ? std::string(modelPathEnv)
        : "../model/act_distill_fp32_for_mindcmd_simp_release.om";
    Result ret = modelProcess_.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed, model path: %s", omModelPath.c_str());
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

    const auto load_end = std::chrono::high_resolution_clock::now();
    const double model_load_ms = std::chrono::duration_cast<std::chrono::microseconds>(
        load_end - load_start).count() / 1000.0;
    INFO_LOG("[PERF] model_load_ms=%.3f model_path=%s", model_load_ms, omModelPath.c_str());

    isModelLoaded_ = true;
    return SUCCESS;
}

// 修改Process方法，只处理单次推理
Result SampleProcess::Process() {
    InferenceResponse response = ProcessOnce(input_datas_, input_sizes_, 0U);
    return response.success ? SUCCESS : FAILED;
}

InferenceResponse SampleProcess::ProcessOnce(
    const std::vector<const void*>& input_datas,
    const std::vector<size_t>& input_sizes,
    uint32_t request_id)
{
    if (!isInited_) {
        ERROR_LOG("resource not initialized");
        return MakeErrorResponse(request_id, WORKER_ERROR_ACL_NOT_READY, "resource not initialized");
    }
    if (!isModelLoaded_) {
        ERROR_LOG("model not initialized");
        return MakeErrorResponse(request_id, WORKER_ERROR_MODEL_NOT_READY, "model not initialized");
    }
    if (input_datas.size() != kExpectedInputCount || input_sizes.size() != kExpectedInputCount) {
        std::ostringstream oss;
        oss << "expected " << kExpectedInputCount << " inputs but got " << input_datas.size();
        ERROR_LOG("%s", oss.str().c_str());
        return MakeErrorResponse(request_id, WORKER_ERROR_INPUT_COUNT, oss.str());
    }

    Result ret = modelProcess_.CreateInputFromData(input_datas, input_sizes);
    if (ret != SUCCESS) {
        ERROR_LOG("Create multi-input failed");
        return MakeErrorResponse(request_id, WORKER_ERROR_INPUT_SIZE, "Create multi-input failed");
    }

    ret = modelProcess_.CreateTaskBufAndWorkBuf();
    if (ret != SUCCESS) {
        ERROR_LOG("CreateTaskBufAndWorkBuf failed");
        modelProcess_.DestroyInput();
        return MakeErrorResponse(request_id, WORKER_ERROR_INFERENCE_FAILED, "CreateTaskBufAndWorkBuf failed");
    }

    auto start = std::chrono::high_resolution_clock::now();

    ret = modelProcess_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("execute inference failed");
        modelProcess_.DestroyInput();
        return MakeErrorResponse(request_id, WORKER_ERROR_INFERENCE_FAILED, "execute inference failed");
    }

    auto end = std::chrono::high_resolution_clock::now();
    uint32_t elapsed_us = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    INFO_LOG("[PERF] request_id=%u model_infer_ms=%.3f", request_id, elapsed_us / 1000.0);

    WorkerTensor actionTensor;
    ret = modelProcess_.GetPackedOutputData(
        kActionOutputIndex,
        actionTensor.data,
        actionTensor.dims,
        actionTensor.elem_type);
    actionTensor.output_index = static_cast<uint32_t>(kActionOutputIndex);

    modelProcess_.DestroyInput();

    if (ret != SUCCESS) {
        ERROR_LOG("extract output failed");
        return MakeErrorResponse(request_id, WORKER_ERROR_OUTPUT_PARSE_FAILED, "extract output failed");
    }

    InferenceResponse response;
    response.request_id = request_id;
    response.success = true;
    response.latency_us = elapsed_us;
    response.error_code = WORKER_ERROR_NONE;
    response.outputs.push_back(actionTensor);
    return response;
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
    if (isModelLoaded_) {
        modelProcess_.DestroyResource();
        isModelLoaded_ = false;
    }

    if (!isInited_) {
        return;
    }

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

    isInited_ = false;
}
