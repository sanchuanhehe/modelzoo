/**
* @file main.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <iostream>
#include <string>
#include <vector>
#include "acl/svp_acl.h"
#include "protocol.h"
#include "sample_process.h"
#include "utils.h"

using namespace std;

namespace {

static const uint16_t kExpectedInputCount = 3;

void FreeInputBuffers(const vector<const void*>& input_datas)
{
    for (size_t i = 0; i < input_datas.size(); ++i) {
        if (input_datas[i] != nullptr) {
            svp_acl_rt_free(const_cast<void*>(input_datas[i]));
        }
    }
}

}  // namespace

int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 初始化推理环境（只执行一次）
    SampleProcess sample;
    if (sample.InitResource() != SUCCESS) {
        ERROR_LOG("Init resource failed");
        return -1;
    }

    // 加载模型（只执行一次）
    if (sample.LoadModel() != SUCCESS) {
        ERROR_LOG("Load model failed");
        return -1;
    }

    // 循环处理多次请求
    while (true) {
        RequestHeader requestHeader;
        string protocolError;
        const ReadHeaderStatus headerStatus = ReadRequestHeader(cin, requestHeader, protocolError);
        if (headerStatus == READ_HEADER_EOF) {
            break;
        }
        if (headerStatus == READ_HEADER_ERROR) {
            ERROR_LOG("Read request header failed: %s", protocolError.c_str());
            break;
        }

        if (requestHeader.magic != WORKER_PROTOCOL_MAGIC
            || requestHeader.version != WORKER_PROTOCOL_VERSION) {
            ERROR_LOG(
                "Invalid protocol header: magic=0x%x version=%u",
                requestHeader.magic,
                requestHeader.version);
            break;
        }

        vector<const void*> input_datas(kExpectedInputCount, nullptr);
        vector<size_t> input_sizes(kExpectedInputCount, 0U);
        bool requestOk = true;

        if (requestHeader.input_count != kExpectedInputCount) {
            requestOk = false;
            protocolError = "unexpected input_count";
        }

        for (uint16_t i = 0; i < requestHeader.input_count; ++i) {
            InputEntryHeader inputHeader;
            if (!ReadInputEntryHeader(cin, inputHeader, protocolError)) {
                ERROR_LOG("Read input header failed: %s", protocolError.c_str());
                requestOk = false;
                break;
            }

            if (inputHeader.input_index >= kExpectedInputCount || input_datas[inputHeader.input_index] != nullptr) {
                requestOk = false;
                protocolError = "invalid or duplicated input_index";
                break;
            }

            void* data = nullptr;
            if (inputHeader.byte_size > 0) {
                svp_acl_error ret = svp_acl_rt_malloc(&data, inputHeader.byte_size, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
                if (ret != SVP_ACL_SUCCESS || data == nullptr) {
                    protocolError = "failed to allocate device buffer for input";
                    requestOk = false;
                    break;
                }

                if (!ReadExact(cin, reinterpret_cast<char*>(data), inputHeader.byte_size, protocolError)) {
                    svp_acl_rt_free(data);
                    data = nullptr;
                    requestOk = false;
                    break;
                }
            }

            input_datas[inputHeader.input_index] = data;
            input_sizes[inputHeader.input_index] = inputHeader.byte_size;
        }

        if (!requestOk) {
            FreeInputBuffers(input_datas);

            InferenceResponse response = MakeErrorResponse(
                requestHeader.request_id,
                WORKER_ERROR_BAD_REQUEST,
                protocolError.empty() ? "bad request frame" : protocolError);
            string writeError;
            if (!WriteResponseFrame(cout, response, writeError)) {
                ERROR_LOG("Write error response failed: %s", writeError.c_str());
                break;
            }
            continue;
        }

        InferenceResponse response = sample.ProcessOnce(
            input_datas,
            input_sizes,
            requestHeader.request_id);

        string writeError;
        if (!WriteResponseFrame(cout, response, writeError)) {
            ERROR_LOG("Write response failed: %s", writeError.c_str());
            break;
        }
    }
    return 0;
}
