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

namespace Infer {
using namespace std;
bool PrintTop5(std::vector<std::string>& fileList, std::vector<TensorBuf>& tensorBufs, std::vector<TensorDesc>& tensorDescs)
{
    if (tensorBufs.size() != 1 || fileList.size() != 1) {
        return false;
    }
    float *outData = static_cast<float*>(tensorBufs[0].GetRawPtr());
    map<float, unsigned int, greater<float> > resultMap;
    for (unsigned int j = 0; j < tensorBufs[0].size / sizeof(float); ++j) {
        resultMap[*outData] = j;
        outData++;
    }
    LOG(INFO) << fileList[0] << " inference result top 5:";
    int cnt = 0;
    for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
        if (++cnt > 5) {
            break;
        }
        LOG(INFO) << "top " << cnt << ": index[" << it->second << "] value[" << it->first << "]";
    }
    return true;
}
}