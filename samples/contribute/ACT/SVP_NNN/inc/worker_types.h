#ifndef WORKER_TYPES_H
#define WORKER_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

static const uint32_t WORKER_PROTOCOL_MAGIC = 0x53565031U;  // "SVP1"
static const uint16_t WORKER_PROTOCOL_VERSION = 1U;

enum WorkerStatusCode {
    WORKER_STATUS_OK = 0,
    WORKER_STATUS_ERROR = 1,
};

enum WorkerErrorCode {
    WORKER_ERROR_NONE = 0,
    WORKER_ERROR_BAD_REQUEST = 1,
    WORKER_ERROR_INPUT_COUNT = 2,
    WORKER_ERROR_INPUT_SIZE = 3,
    WORKER_ERROR_ACL_NOT_READY = 4,
    WORKER_ERROR_MODEL_NOT_READY = 5,
    WORKER_ERROR_INFERENCE_FAILED = 6,
    WORKER_ERROR_OUTPUT_PARSE_FAILED = 7,
    WORKER_ERROR_ALLOC_FAILED = 8,
    WORKER_ERROR_PROTOCOL_IO = 9,
};

enum WorkerElementType {
    WORKER_ELEM_UNKNOWN = 0,
    WORKER_ELEM_FLOAT32 = 1,
    WORKER_ELEM_FLOAT16 = 2,
    WORKER_ELEM_INT8 = 3,
    WORKER_ELEM_UINT8 = 4,
    WORKER_ELEM_INT32 = 5,
    WORKER_ELEM_INT64 = 6,
};

struct WorkerTensor {
    uint32_t output_index = 0;
    uint32_t elem_type = WORKER_ELEM_UNKNOWN;
    std::vector<int64_t> dims;
    std::vector<uint8_t> data;
};

struct InferenceResponse {
    uint32_t request_id = 0;
    bool success = false;
    uint32_t latency_us = 0;
    int32_t error_code = WORKER_ERROR_NONE;
    std::string error_msg;
    std::vector<WorkerTensor> outputs;
};

inline InferenceResponse MakeErrorResponse(
    uint32_t request_id,
    int32_t error_code,
    const std::string& error_msg)
{
    InferenceResponse response;
    response.request_id = request_id;
    response.success = false;
    response.error_code = error_code;
    response.error_msg = error_msg;
    return response;
}

#endif
