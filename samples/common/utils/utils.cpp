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

namespace Infer {
using namespace std;
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

bool PadDataToTensorBuf(void* data, size_t size, TensorDesc& desc, TensorBuf& buf)
{
    size_t width = desc.dims[desc.dimCount - 1]; // dims last dim is width
    size_t lineSize = width * desc.typeSize / 8; // 1 byte = 8bits
    if (buf.stride == 0 || lineSize == buf.stride) {
        if (DevMemcpy(buf.GetRawPtr(), buf.size, data, size) != 0) {
            LOG(ERROR) << "fail to memcpy";
            return false;
        }
        return true;
    }
    size_t loopTimes = 1;
    for (size_t loop = 0; loop < desc.dimCount - 1; loop++) {
        loopTimes *= desc.dims[loop];
    }

    size_t remain = buf.size;
    for (size_t loop = 0; loop < loopTimes; loop++) {
        if (DevMemcpy(static_cast<uint8_t*>(buf.GetRawPtr()) + loop * buf.stride, remain, static_cast<uint8_t*>(data) + loop * lineSize, lineSize) != 0) {
            LOG(ERROR) << "fail to memcpy";
            return false;
        }
        remain = remain > buf.stride ? remain - buf.stride : 0;
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

}