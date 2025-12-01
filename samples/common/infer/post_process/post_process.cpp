/*
 * Copyright (c) ModelZoo. 2025-2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "post_process.h"
#include "log.h"
#include <map>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

namespace Infer {
using namespace std;
bool PrintTop5AndDumpResult(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    if (tensorBufs.size() != 1 || fileList.size() != 1) {
        return false;
    }

    size_t num = tensorBufs[0].size / sizeof(float);
    std::vector<std::pair<float, size_t>> result(num);

    float *outData = static_cast<float*>(tensorBufs[0].GetRawPtr());
    for (size_t j = 0; j < num; ++j) {
        result[j].first = *outData;
        result[j].second = j;
        outData++;
    }
    std::sort(result.begin(), result.end(), 
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });
    LOG(INFO) << fileList[0] << " inference result top 5:";
    int cnt = 0;
    for (auto it = result.begin(); it != result.end(); ++it) {
        if (++cnt > 5) {
            break;
        }
        LOG(INFO) << "top " << cnt << ": index[" << it->second << "] value[" << it->first << "]";
    }

    static std::string resultPath = "./result";
    static std::string txtPath = "./result/txt";
    struct stat info;
    if (stat(resultPath.c_str(), &info) != 0) {
        mkdir(resultPath.c_str(), 0777);
    }
    if (stat(txtPath.c_str(), &info) != 0) {
        mkdir(txtPath.c_str(), 0777);
    }
    size_t start = fileList[0].find_last_of("/");
    size_t end = fileList[0].find_last_of(".");
    std::string fileName = fileList[0].substr(start , end-start);
    std::string binFile = txtPath + fileName + "_0.txt";
    std::ofstream fout(binFile, std::ios::out | std::ios::trunc);
    if (fout.good() == false) {
        std::cout << "create output file failed " << binFile << std::endl;
        return false;
    }
    for (auto &t : result) {
        fout << t.second << "," << t.first << "\n";
    }
    fout.close();

    return true;
}
}