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
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <string>
#include "acl/acl.h"
#include "acl/acl_mdl.h"

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
    * Function: Allocate and initialize device-side buffer using data from a specified file
    * Input: fileName - full path/name of the input file
    * Input: dims - dimension information of the model I/O data
    * Input: stride - step size for data alignment
    * Input: dataSize - total size of the data to be read
    * Return: Pointer to the allocated device buffer (nullptr if failed)
    */
    static void* GetDeviceBufferOfFile(const std::string& fileName, const aclmdlIODims& dims,
        size_t stride, size_t dataSize);

    /**
    * Function: Read binary file content into a memory buffer
    * Input: fileName - path and name of the binary file to read
    * Output: fileSize - total byte size of the read file (output parameter)
    * Return: Pointer to the host-side buffer containing file data (nullptr if failed)
    */
    static void* ReadBinFile(const std::string& fileName, uint32_t& fileSize);

    /**
    * Function: Get the total byte size of a specified file
    * Input: fileName - path and name of the target file
    * Output: fileSize - byte size of the file (output parameter)
    * Return: Result code (success/failure status)
    */
    static Result GetFileSize(const std::string& fileName, uint32_t& fileSize);

    /**
    * Function: Read binary file with stride alignment and fill to specified dimension
    * Input: fileName - path and name of the binary file
    * Input: dims - dimension configuration of the model I/O
    * Input: stride - alignment step size for data processing
    * Input: dataSize - expected total size of the output buffer
    * Return: Pointer to the aligned data buffer (nullptr if failed)
    */
    static void* ReadBinFileWithStride(const std::string& fileName, const aclmdlIODims& dims,
        size_t stride, size_t dataSize);

    /**
    * Function: Initialize int8_t type data buffer with default values
    * Input: data - pointer to the int8_t data buffer to initialize
    * Input: dataSize - total size of the data buffer in bytes
    * Note: This function will overwrite all existing data in the buffer
    */
    static void InitData(int8_t* data, size_t dataSize);
};

#endif
