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

#include "log.h"
#include "dev_interface_adapter.h"
#include <getopt.h>
#include <limits.h>
#include <stdio.h>

using namespace Infer;
using namespace std;

constexpr float MS2S = 1000.0f;

struct InferParam {
    string omModelPath;
    string aclConfigPath;
    size_t loop {1};
};

static bool PathToRealPath(const std::string &path, std::string &realPath)
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

static bool ParseCmd(int argc, char *argv[], InferParam &inferParam)
{
    int opt;
    const char *optstring = "m:a:l:";
    struct option longOptions[] = {
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"loop", required_argument, NULL, 'l'},
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
            case 'l':{
                char *endptr = nullptr;
                inferParam.loop = strtoull(optarg, &endptr, 0);
                if (*endptr != '\0') {
                    LOG(ERROR) << "incorrect input after -l/--loop, " << endptr;
                    return false;
                }
                break;
            }
            case '?':
                LOG(ERROR) << "incorrect config";
                return false;
            default:
                return false;
        }
    }
    return true;
}

static int ModelInfer(InferParam &inferParam)
{
    std::shared_ptr<MdlBase> model = MdlCreate();
    int ret = model->LoadModel(inferParam.omModelPath);
    if (ret != 0) {
        LOG(ERROR) << "load model failed, path : " << inferParam.omModelPath;
        return ret;
    }
    /* set in and out bufs, only support single input*/
    std::vector<TensorBuf> inBufs, outBufs;
    TensorDesc desc;
    size_t inputNum = model->GetInTensorNum();
    size_t outputNum = model->GetOutTensorNum();
    for (size_t i = 0; i < inputNum; i++) {
        model->GetInTensorDescByIdx(i, desc);
        inBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }
    for (size_t i = 0; i < outputNum; i++) {
        model->GetOutTensorDescByIdx(i, desc);
        outBufs.emplace_back(desc.defaultSize, desc.defaultStride);
    }
    inBufs.emplace_back(4, 0);
    std::chrono::microseconds dur(0);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < inferParam.loop; j++) {
        ret = model->Execute(inBufs, outBufs);
        if (ret != SUCCESS) {
            LOG(ERROR) << "execute inference failed";
            model->UnLoadModel();
            return ret;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    dur += std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float msDur = static_cast<float>(dur.count()) / (inferParam.loop  * MS2S);
    LOG(DEBUG) << std::fixed << std::setprecision(2) << "api execution time: " << msDur << "ms, frame rate: " << (MS2S / msDur) << "fps";
    msDur = *(static_cast<float*>(inBufs[inBufs.size() - 1].GetRawPtr())) / (inferParam.loop * MS2S);
    LOG(INFO) << std::fixed << std::setprecision(2) << "execution time: " << msDur << "ms, frame rate: " << (MS2S / msDur) << "fps";
    model->UnLoadModel();
    return 0;
}

int main(int argc, char *argv[])
{
    InferParam inferParam;
    if (!ParseCmd(argc, argv, inferParam)) {
        LOG(ERROR) << "fail to parse cmd";
        return -1;
    }
    int ret = DevInit(inferParam.aclConfigPath);
    if (ret != 0) {
        LOG(ERROR) << "dev init failed";
        return -1;
    }
    ret = ModelInfer(inferParam);
    if (ret != 0) {
        LOG(ERROR) << "model infer failed";
        return -1;
    }
    DevDeInit();
    return 0;
}