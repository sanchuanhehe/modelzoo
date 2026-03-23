/*
 * Copyright (c) ModelZoo. 2026-2026. All rights reserved.
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
#include "ClipInfer.h"
#include <getopt.h>

int main(int argc, char* argv[])
{
    ClipInfer model;
    
    if (!model.ExecuteParams(argc, argv)) {
        LOG(ERROR) << "Fail to parse cmd!";
        return -1;
    }
    EnvInit(model.acl_path_);
    model.LoadModel();
    std::vector<std::vector<float>> zeroshotWeights = model.InferTxt();
    for (size_t i = 0; i < model.imgFileList_.size(); ++i)
    {
        std::vector<float> imgResult = model.InferImageSingle(model.imgFileList_[i][0]);
        std::vector<float> tmp = model.ComputeModelLogits(imgResult, zeroshotWeights, 512, model.txtFileList_.size(), model.imgFileList_[i][0]);
    }
    model.UnLoadModel();
    EnvDeinit();
    return 0;
}