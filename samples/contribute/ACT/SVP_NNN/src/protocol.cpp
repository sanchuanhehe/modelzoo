#include "protocol.h"

#include <cstring>
#include <istream>
#include <limits>
#include <ostream>

namespace {

bool ReadBytes(std::istream& in, char* data, size_t size, std::string& error_msg)
{
    if (size == 0) {
        return true;
    }
    in.read(data, static_cast<std::streamsize>(size));
    if (!in.good()) {
        error_msg = "failed to read bytes from worker input stream";
        return false;
    }
    return true;
}

bool WriteBytes(std::ostream& out, const char* data, size_t size, std::string& error_msg)
{
    if (size == 0) {
        return true;
    }
    out.write(data, static_cast<std::streamsize>(size));
    if (!out.good()) {
        error_msg = "failed to write bytes to worker output stream";
        return false;
    }
    return true;
}

bool ReadUint16LE(std::istream& in, uint16_t& value, std::string& error_msg)
{
    unsigned char buf[2];
    if (!ReadBytes(in, reinterpret_cast<char*>(buf), sizeof(buf), error_msg)) {
        return false;
    }
    value = static_cast<uint16_t>(buf[0])
        | (static_cast<uint16_t>(buf[1]) << 8);
    return true;
}

bool ReadUint32LE(std::istream& in, uint32_t& value, std::string& error_msg)
{
    unsigned char buf[4];
    if (!ReadBytes(in, reinterpret_cast<char*>(buf), sizeof(buf), error_msg)) {
        return false;
    }
    value = static_cast<uint32_t>(buf[0])
        | (static_cast<uint32_t>(buf[1]) << 8)
        | (static_cast<uint32_t>(buf[2]) << 16)
        | (static_cast<uint32_t>(buf[3]) << 24);
    return true;
}

bool WriteUint16LE(std::ostream& out, uint16_t value, std::string& error_msg)
{
    unsigned char buf[2];
    buf[0] = static_cast<unsigned char>(value & 0xFFU);
    buf[1] = static_cast<unsigned char>((value >> 8) & 0xFFU);
    return WriteBytes(out, reinterpret_cast<const char*>(buf), sizeof(buf), error_msg);
}

bool WriteUint32LE(std::ostream& out, uint32_t value, std::string& error_msg)
{
    unsigned char buf[4];
    buf[0] = static_cast<unsigned char>(value & 0xFFU);
    buf[1] = static_cast<unsigned char>((value >> 8) & 0xFFU);
    buf[2] = static_cast<unsigned char>((value >> 16) & 0xFFU);
    buf[3] = static_cast<unsigned char>((value >> 24) & 0xFFU);
    return WriteBytes(out, reinterpret_cast<const char*>(buf), sizeof(buf), error_msg);
}

bool WriteInt32LE(std::ostream& out, int32_t value, std::string& error_msg)
{
    return WriteUint32LE(out, static_cast<uint32_t>(value), error_msg);
}

bool WriteUint64LE(std::ostream& out, uint64_t value, std::string& error_msg)
{
    unsigned char buf[8];
    buf[0] = static_cast<unsigned char>(value & 0xFFULL);
    buf[1] = static_cast<unsigned char>((value >> 8) & 0xFFULL);
    buf[2] = static_cast<unsigned char>((value >> 16) & 0xFFULL);
    buf[3] = static_cast<unsigned char>((value >> 24) & 0xFFULL);
    buf[4] = static_cast<unsigned char>((value >> 32) & 0xFFULL);
    buf[5] = static_cast<unsigned char>((value >> 40) & 0xFFULL);
    buf[6] = static_cast<unsigned char>((value >> 48) & 0xFFULL);
    buf[7] = static_cast<unsigned char>((value >> 56) & 0xFFULL);
    return WriteBytes(out, reinterpret_cast<const char*>(buf), sizeof(buf), error_msg);
}

}  // namespace

ReadHeaderStatus ReadRequestHeader(std::istream& in, RequestHeader& header, std::string& error_msg)
{
    header = RequestHeader();

    const int first = in.peek();
    if (first == std::char_traits<char>::eof()) {
        if (in.eof()) {
            return READ_HEADER_EOF;
        }
        error_msg = "failed to peek worker input stream";
        return READ_HEADER_ERROR;
    }

    if (!ReadUint32LE(in, header.magic, error_msg)) {
        return READ_HEADER_ERROR;
    }
    if (!ReadUint16LE(in, header.version, error_msg)) {
        return READ_HEADER_ERROR;
    }
    if (!ReadUint16LE(in, header.input_count, error_msg)) {
        return READ_HEADER_ERROR;
    }
    if (!ReadUint32LE(in, header.request_id, error_msg)) {
        return READ_HEADER_ERROR;
    }
    if (!ReadUint32LE(in, header.flags, error_msg)) {
        return READ_HEADER_ERROR;
    }

    return READ_HEADER_OK;
}

bool ReadInputEntryHeader(std::istream& in, InputEntryHeader& header, std::string& error_msg)
{
    header = InputEntryHeader();
    return ReadUint32LE(in, header.input_index, error_msg)
        && ReadUint32LE(in, header.byte_size, error_msg)
        && ReadUint32LE(in, header.reserved, error_msg);
}

bool ReadExact(std::istream& in, char* data, size_t size, std::string& error_msg)
{
    return ReadBytes(in, data, size, error_msg);
}

bool WriteResponseFrame(std::ostream& out, const InferenceResponse& response, std::string& error_msg)
{
    const uint16_t status = response.success ? WORKER_STATUS_OK : WORKER_STATUS_ERROR;
    const uint32_t output_count = static_cast<uint32_t>(response.outputs.size());
    const uint32_t error_msg_size = static_cast<uint32_t>(response.error_msg.size());

    if (!WriteUint32LE(out, WORKER_PROTOCOL_MAGIC, error_msg)
        || !WriteUint16LE(out, WORKER_PROTOCOL_VERSION, error_msg)
        || !WriteUint16LE(out, status, error_msg)
        || !WriteUint32LE(out, response.request_id, error_msg)
        || !WriteUint32LE(out, output_count, error_msg)
        || !WriteUint32LE(out, response.latency_us, error_msg)
        || !WriteInt32LE(out, response.error_code, error_msg)
        || !WriteUint32LE(out, error_msg_size, error_msg)) {
        return false;
    }

    for (size_t i = 0; i < response.outputs.size(); ++i) {
        const WorkerTensor& tensor = response.outputs[i];
        const uint32_t elem_count = tensor.data.empty() ? 0U : static_cast<uint32_t>(
            tensor.data.size() / (tensor.elem_type == WORKER_ELEM_FLOAT16 ? 2U :
                tensor.elem_type == WORKER_ELEM_FLOAT32 ? 4U :
                tensor.elem_type == WORKER_ELEM_INT32 ? 4U :
                tensor.elem_type == WORKER_ELEM_INT64 ? 8U : 1U));
        const uint32_t byte_size = static_cast<uint32_t>(tensor.data.size());
        const uint32_t dim_count = static_cast<uint32_t>(tensor.dims.size());

        if (!WriteUint32LE(out, tensor.output_index, error_msg)
            || !WriteUint32LE(out, tensor.elem_type, error_msg)
            || !WriteUint32LE(out, elem_count, error_msg)
            || !WriteUint32LE(out, byte_size, error_msg)
            || !WriteUint32LE(out, dim_count, error_msg)
            || !WriteUint32LE(out, 0U, error_msg)) {
            return false;
        }

        for (size_t dim_idx = 0; dim_idx < tensor.dims.size(); ++dim_idx) {
            if (!WriteUint64LE(out, static_cast<uint64_t>(tensor.dims[dim_idx]), error_msg)) {
                return false;
            }
        }

        if (!WriteBytes(
                out,
                reinterpret_cast<const char*>(tensor.data.data()),
                tensor.data.size(),
                error_msg)) {
            return false;
        }
    }

    if (!WriteBytes(out, response.error_msg.data(), response.error_msg.size(), error_msg)) {
        return false;
    }

    out.flush();
    if (!out.good()) {
        error_msg = "failed to flush worker output stream";
        return false;
    }
    return true;
}
