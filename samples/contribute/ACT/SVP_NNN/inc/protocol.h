#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>

#include "worker_types.h"

struct RequestHeader {
    uint32_t magic = 0;
    uint16_t version = 0;
    uint16_t input_count = 0;
    uint32_t request_id = 0;
    uint32_t flags = 0;
};

struct InputEntryHeader {
    uint32_t input_index = 0;
    uint32_t byte_size = 0;
    uint32_t reserved = 0;
};

enum ReadHeaderStatus {
    READ_HEADER_OK = 0,
    READ_HEADER_EOF = 1,
    READ_HEADER_ERROR = 2,
};

ReadHeaderStatus ReadRequestHeader(std::istream& in, RequestHeader& header, std::string& error_msg);
bool ReadInputEntryHeader(std::istream& in, InputEntryHeader& header, std::string& error_msg);
bool ReadExact(std::istream& in, char* data, size_t size, std::string& error_msg);
bool WriteResponseFrame(std::ostream& out, const InferenceResponse& response, std::string& error_msg);

#endif
