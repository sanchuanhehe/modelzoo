/**
* @file utils.h
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include "acl/svp_acl.h"
#include "acl/svp_acl_mdl.h"

#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef _WIN32
#define S_ISREG(m) (((m) & 0170000) == (0100000))
#endif

typedef enum Result {
    SUCCESS = 0,
    FAILED = 1
} Result;

class Utils {
public:
    /**
    * @brief create device buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return device buffer of file
    */
    static void* GetDeviceBufferOfFile(const std::string& fileName, const svp_acl_mdl_io_dims& dims,
        size_t stride, size_t dataSize);

    /**
    * @brief create buffer of file
    * @param [in] fileName: file name
    * @param [out] fileSize: size of file
    * @return buffer of pic
    */
    static void* ReadBinFile(const std::string& fileName, uint32_t& fileSize);

    static Result GetFileSize(const std::string& fileName, uint32_t& fileSize);

    static void* ReadBinFileWithStride(const std::string& fileName, const svp_acl_mdl_io_dims& dims,
        size_t stride, size_t dataSize);

    static void InitData(int8_t* data, size_t dataSize);
};

#endif
