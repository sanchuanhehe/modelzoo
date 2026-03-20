#include "utils.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include "acl/acl.h"

void Utils::InitData(int8_t* data, size_t dataSize)
{
    for (size_t i = 0; i < dataSize; i++) {
        data[i] = 0;
    }
}

Result Utils::GetFileSize(const std::string& fileName, uint32_t& fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return FAILED;
    }
    binFile.seekg(0, binFile.end);
    int binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return FAILED;
    }
    fileSize = static_cast<uint32_t>(binFileBufferLen);
    binFile.close();
    return SUCCESS;
}

void* Utils::ReadBinFile(const std::string& fileName, uint32_t &fileSize)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", fileName.c_str());
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }
    binFile.seekg(0, binFile.end);
    int binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }
    binFile.seekg(0, binFile.beg);
    void* binFileBufferData = nullptr;
    aclError ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
        binFile.close();
        return nullptr;
    }
    InitData(static_cast<int8_t*>(binFileBufferData), binFileBufferLen);

    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = static_cast<uint32_t>(binFileBufferLen);
    return binFileBufferData;
}

void* Utils::ReadBinFileWithStride(const std::string& fileName, const aclmdlIODims& dims,
    size_t stride, size_t dataSize)
{
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", fileName.c_str());
        return nullptr;
    }

    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }

    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }
    binFile.seekg(0, binFile.end);
    int binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }
    binFile.seekg(0, binFile.beg);
    void* binFileBufferData = nullptr;
    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < dims.dimCount - 1; loop++) {
        loopTimes *= dims.dims[loop];
    }
    size_t bufferSize = loopTimes * stride;
    aclError ret = aclrtMalloc(&binFileBufferData, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
        binFile.close();
        return nullptr;
    }
    InitData(static_cast<int8_t*>(binFileBufferData), bufferSize);

    int64_t dimValue = dims.dims[dims.dimCount - 1];
    size_t lineSize = dimValue * dataSize;
    for (int64_t loop = 0; loop < loopTimes; loop++) {
        binFile.read((static_cast<char *>(binFileBufferData) + loop * stride), lineSize);
    }

    binFile.close();
    return binFileBufferData;
}

void* Utils::GetDeviceBufferOfFile(const std::string& fileName, const aclmdlIODims& dims,
    size_t stride, size_t dataSize)
{
    return Utils::ReadBinFileWithStride(fileName, dims, stride, dataSize);
}
