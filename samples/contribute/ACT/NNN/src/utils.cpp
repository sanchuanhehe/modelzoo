/**
* Copyright (C) 2020. Huawei Technologies Co.; Ltd. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "utils.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include "acl/acl.h"

// 功能：数据内存初始化，将指定内存区域全部置为0
// data：待初始化的数据指针
// dataSize：需要初始化的内存大小
void Utils::InitData(int8_t* data, size_t dataSize)
{
    // 循环遍历内存，逐字节置0
    for (size_t i = 0; i < dataSize; i++) {
        data[i] = 0;
    }
}

// 功能：获取二进制文件的大小
// fileName：文件路径
// fileSize：输出参数，用于存储获取到的文件大小
// 返回值：成功/失败
Result Utils::GetFileSize(const std::string& fileName, uint32_t& fileSize)
{
    // 以二进制只读方式打开文件
    std::ifstream binFile(fileName, std::ifstream::binary);
    // 判断文件是否打开成功
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return FAILED;
    }
    // 将文件读取指针移动到文件末尾
    binFile.seekg(0, binFile.end);
    // 获取文件读取指针当前位置，即文件总大小
    int binFileBufferLen = binFile.tellg();
    // 判断文件是否为空
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return FAILED;
    }
    // 将文件大小赋值给输出参数
    fileSize = static_cast<uint32_t>(binFileBufferLen);
    // 关闭文件
    binFile.close();
    return SUCCESS;
}

// 功能：读取二进制文件内容，并直接加载到昇腾设备内存
// fileName：文件路径
// fileSize：输出参数，返回读取的文件大小
// 返回值：存储文件内容的设备内存指针
void* Utils::ReadBinFile(const std::string& fileName, uint32_t &fileSize)
{
    // 定义文件状态结构体
    struct stat sBuf;
    // 获取文件状态信息
    int fileStatus = stat(fileName.data(), &sBuf);
    // 判断获取文件状态是否失败
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", fileName.c_str());
        return nullptr;
    }
    // 判断是否为常规文件
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }
    // 以二进制方式打开文件
    std::ifstream binFile(fileName, std::ifstream::binary);
    // 判断文件是否打开失败
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }
    // 移动文件指针到末尾
    binFile.seekg(0, binFile.end);
    // 获取文件大小
    int binFileBufferLen = binFile.tellg();
    // 判断文件是否为空
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }
    // 移动文件指针到文件开头
    binFile.seekg(0, binFile.beg);

    // 定义设备内存指针
    void* binFileBufferData = nullptr;
    // 在昇腾AI芯片上申请设备内存
    aclError ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
    // 判断内存申请是否失败
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
        binFile.close();
        return nullptr;
    }
    // 初始化设备内存，全部置0
    InitData(static_cast<int8_t*>(binFileBufferData), binFileBufferLen);

    // 将文件内容读取到设备内存中
    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    // 关闭文件
    binFile.close();
    // 设置文件大小输出参数
    fileSize = static_cast<uint32_t>(binFileBufferLen);
    // 返回设备内存指针
    return binFileBufferData;
}

// 功能：按照指定步长（stride）读取文件到设备内存，适配模型输入格式
// fileName：文件路径
// dims：模型输入维度信息
// stride：内存对齐步长
// dataSize：单个数据元素的字节大小
// 返回值：加载完成的设备内存指针
void* Utils::ReadBinFileWithStride(const std::string& fileName, const aclmdlIODims& dims,
    size_t stride, size_t dataSize)
{
    // 定义文件状态结构体
    struct stat sBuf;
    // 获取文件状态
    int fileStatus = stat(fileName.data(), &sBuf);
    // 判断获取文件状态失败
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", fileName.c_str());
        return nullptr;
    }

    // 判断是否为常规文件
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }

    // 以二进制方式打开文件
    std::ifstream binFile(fileName, std::ifstream::binary);
    // 判断文件打开失败
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }
    // 移动指针到文件末尾获取大小
    binFile.seekg(0, binFile.end);
    int binFileBufferLen = binFile.tellg();
    // 判断文件为空
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }
    // 移动指针到文件开头
    binFile.seekg(0, binFile.beg);

    // 设备内存指针初始化
    void* binFileBufferData = nullptr;
    // 计算循环次数：维度前N-1维相乘
    int64_t loopTimes = 1;
    for (size_t loop = 0; loop < dims.dimCount - 1; loop++) {
        loopTimes *= dims.dims[loop];
    }

    // 计算需要申请的总缓冲区大小
    size_t bufferSize = loopTimes * stride;
    // 申请设备内存
    aclError ret = aclrtMalloc(&binFileBufferData, bufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    // 判断内存申请失败
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
        binFile.close();
        return nullptr;
    }
    // 初始化内存为0
    InitData(static_cast<int8_t*>(binFileBufferData), bufferSize);

    // 获取最后一维的维度值
    int64_t dimValue = dims.dims[dims.dimCount - 1];
    // 计算单行数据大小
    size_t lineSize = dimValue * dataSize;
    // 按行读取数据，按stride间隔存储
    for (int64_t loop = 0; loop < loopTimes; loop++) {
        binFile.read((static_cast<char *>(binFileBufferData) + loop * stride), lineSize);
    }

    // 关闭文件
    binFile.close();
    // 返回设备内存指针
    return binFileBufferData;
}

// 功能：对外提供的接口，根据文件路径获取带步长格式的设备内存
// fileName：文件路径
// dims：模型输入维度
// stride：内存对齐步长
// dataSize：单个数据元素大小
// 返回值：设备内存指针
void* Utils::GetDeviceBufferOfFile(const std::string& fileName, const aclmdlIODims& dims,
    size_t stride, size_t dataSize)
{
    // 调用带步长的文件读取函数
    return Utils::ReadBinFileWithStride(fileName, dims, stride, dataSize);
}
