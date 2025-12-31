#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <log.h>
#include <getopt.h>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <errno.h>
#include <cstring>

namespace Infer {
using namespace std;

// FP32 (单精度浮点数) 标准定义常量
constexpr uint32_t FP32_SIGN_SHIFT = 31;          // 符号位偏移（第31位）
constexpr uint32_t FP32_SIGN_MASK = 0x1;          // 符号位掩码（仅最低1位）
constexpr uint32_t FP32_EXP_SHIFT = 23;           // 指数位偏移（第23-30位）
constexpr uint32_t FP32_EXP_MASK = 0xFF;          // 指数位掩码（8位）
constexpr uint32_t FP32_MANTISSA_MASK = 0x7FFFFF; // 尾数位掩码（23位）
constexpr int32_t FP32_EXP_BIAS = 127;            // 指数偏移量（FP32标准定义）

// FP16 (半精度浮点数) 标准定义常量
constexpr uint16_t FP16_SIGN_SHIFT = 15;          // 符号位偏移（第15位）
constexpr uint16_t FP16_SIGN_MASK = 0x1;          // 符号位掩码（仅最低1位）
constexpr uint16_t FP16_EXP_SHIFT = 10;           // 指数位偏移（第10-14位）
constexpr uint16_t FP16_EXP_MASK = 0x1F;          // 指数位掩码（5位）
constexpr uint16_t FP16_MANTISSA_MASK = 0x3FF;    // 尾数位掩码（10位）
constexpr int32_t FP16_EXP_BIAS = 15;             // 指数偏移量（FP16标准定义）
constexpr int32_t FP16_EXP_MAX = 31;              // 指数位最大值（5位无符号数上限）
constexpr int32_t FP16_EXP_MIN = 0;               // 指数位最小值（5位无符号数下限）
constexpr uint32_t FP16_MANTISSA_EXTEND_SHIFT = 13; // FP16尾数转FP32时的左移位数（23-10=13）
constexpr uint32_t FP32_MANTISSA_TRUNC_SHIFT = 13; // FP32尾数转FP16时的右移位数（23-10=13）

bool PathToRealPath(const std::string &path, std::string &realPath)
{
    if (path.empty()) {
        return false;
    }
    if (path.length() > PATH_MAX) {
        return false;
    }
    char tmpPath[PATH_MAX] = {};
    if (realpath(path.c_str(), tmpPath) == nullptr) {
        return false;
    }
    realPath = tmpPath;
    return true;
}

bool ParseParamFromCmd(int argc, char *argv[], InferParam &inferParam)
{
    int opt;
    const char *optstring = "hm:a:i:l:";
    struct option longOptions[] = {
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {0, 0, 0, 0}
    };
    while ((opt = getopt_long(argc, argv, optstring, longOptions, NULL)) != -1) {
        switch (opt) {
            case 'm':
                if (!PathToRealPath(optarg, inferParam.omModelPath)) {
                    LOG(ERROR) << "parse model path error";
                    return false;
                }
                break;
            case 'a':
                if (!PathToRealPath(optarg, inferParam.aclConfigPath)) {
                    LOG(ERROR) << "parse acl config path error";
                    return false;
                }
                break;
            case 'i':
                if (!PathToRealPath(optarg, inferParam.imglistPath)) {
                    LOG(ERROR) << "parse image dir error";
                    return false;
                }
                break;
            case '?':
                LOG(ERROR) << "incorrect config";
                return false;
            default:
                return false;
        }
    }
    return true;
}

std::unordered_map<std::string, std::string> ReadCfgFile(const std::string& cfgPath) {
    std::unordered_map<std::string, std::string> params;
    std::ifstream file(cfgPath);
    if (!file.is_open()) return params;

    std::string line;
    while (std::getline(file, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#') continue;
        size_t eqPos = line.find('=');
        if (eqPos == std::string::npos) continue;
        std::string key = line.substr(0, eqPos);
        std::string value = line.substr(eqPos + 1);
        // 去除引号和空格
        value.erase(remove_if(value.begin(), value.end(), [](char c) {
            return c == '"' || c == ' ';
        }), value.end());
        params[key] = value;
    }
    return params;
}

bool CreateDirectoryRecursive(const std::string& path) {
    mode_t mode = 0755;  // 权限设置为755
    size_t pos = 0;
    std::string currentPath;

    // 跳过路径开头的根目录符号（如"/"）
    if (!path.empty() && path[0] == '/') {
        currentPath += "/";
        pos = 1;
    }

    while (pos < path.size()) {
        // 查找下一个路径分隔符
        size_t next = path.find('/', pos);
        if (next == std::string::npos) {
            next = path.size();  // 处理最后一级目录
        }

        // 提取当前级目录名
        std::string dir = path.substr(pos, next - pos);
        if (dir.empty()) {  // 跳过连续的"/"（如"a//b"）
            pos = next + 1;
            continue;
        }

        currentPath += dir;  // 拼接当前级目录

        // 尝试创建目录
        if (mkdir(currentPath.c_str(), mode) != 0) {
            // 若目录已存在，忽略错误；否则返回失败
            if (errno != EEXIST) {
                return false;
            }
        }

        currentPath += "/";  // 为下一级目录拼接分隔符
        pos = next + 1;
    }

    return true;
}

/**
 * @brief FP32 转 FP16（用 memcpy 避免严格别名警告）
 */
uint16_t FloatToHalf(float value) {
    uint32_t fp32Bits;
    // 字节级拷贝：将 float 的二进制内容拷贝到 uint32_t（无类型转换冲突）
    std::memcpy(&fp32Bits, &value, sizeof(float));

    // 提取 FP32 符号位、指数位、尾数位（逻辑不变）
    uint32_t sign = (fp32Bits >> FP32_SIGN_SHIFT) & FP32_SIGN_MASK;
    uint32_t exp = (fp32Bits >> FP32_EXP_SHIFT) & FP32_EXP_MASK;
    uint32_t mantissa = fp32Bits & FP32_MANTISSA_MASK;

    int32_t fp16Exp = static_cast<int32_t>(exp) - FP32_EXP_BIAS + FP16_EXP_BIAS;
    if (fp16Exp > FP16_EXP_MAX) {
        fp16Exp = FP16_EXP_MAX;
        mantissa = 0;
    } else if (fp16Exp < FP16_EXP_MIN) {
        fp16Exp = FP16_EXP_MIN;
        mantissa = 0;
    }

    return static_cast<uint16_t>(
        (sign << FP16_SIGN_SHIFT) |
        (static_cast<uint16_t>(fp16Exp) << FP16_EXP_SHIFT) |
        (static_cast<uint16_t>(mantissa >> FP32_MANTISSA_TRUNC_SHIFT) & FP16_MANTISSA_MASK)
    );
}

/**
 * @brief FP16 转 FP32（用 memcpy 避免严格别名警告）
 */
float HalfToFloat(uint16_t value) {
    // 提取 FP16 符号位、指数位、尾数位（逻辑不变）
    uint32_t sign = (static_cast<uint32_t>(value) >> FP16_SIGN_SHIFT) & FP16_SIGN_MASK;
    uint32_t exp = (static_cast<uint32_t>(value) >> FP16_EXP_SHIFT) & FP16_EXP_MASK;
    uint32_t mantissa = (static_cast<uint32_t>(value) & FP16_MANTISSA_MASK);

    if (exp == 0 && mantissa == 0) {
        uint32_t fp32Zero = sign << FP32_SIGN_SHIFT;  
        float result;
        // 字节级拷贝：将 uint32_t 的二进制内容拷贝到 float
        std::memcpy(&result, &fp32Zero, sizeof(float));
        return result;
    }

    uint32_t fp32Exp = exp + FP32_EXP_BIAS - FP16_EXP_BIAS;  
    uint32_t fp32Bits =  
        (sign << FP32_SIGN_SHIFT) |
        (fp32Exp << FP32_EXP_SHIFT) |
        (mantissa << FP16_MANTISSA_EXTEND_SHIFT);

    float result;
    // 字节级拷贝：避免指针类型转换
    std::memcpy(&result, &fp32Bits, sizeof(float));
    return result;
}

}