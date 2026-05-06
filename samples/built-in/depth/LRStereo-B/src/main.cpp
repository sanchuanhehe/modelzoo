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

#include <memory>
#include "model.h"
#include "log.h"
#include "utils.h"

using namespace Infer;

int main(int argc, char *argv[])
{
    InferParam inferParam;
    if (!ParseParamFromCmd(argc, argv, inferParam))
    {
        LOG(ERROR) << "fail to parse cmd";
        return -1;
    }
    EnvInit(inferParam.aclConfigPath);
    std::unique_ptr<Model> modelStereo = std::make_unique<Model>();
    if (modelStereo->Load(inferParam.omModelPath, LRStereo) != 0)
    {
        LOG(ERROR) << "fail to load model";
        return 0;
    }
    std::unique_ptr<Model> modelDis = std::make_unique<Model>();
    // 读取配置文件
    auto cfg = ReadCfgFile("../data/cfg.txt");
    std::string dis_path = cfg["dis_path"];
    bool type = std::stoi(cfg["type"]) == 1;
    if (type && modelDis->Load(dis_path, LRStereoDis) != 0)
    {
        LOG(ERROR) << "fail to load Dis model";
        return 0;
    }
    std::vector<std::vector<Tensor>> ret;
    std::vector<Tensor> result;
    if (type)
    {
        std::vector<std::vector<std::string>> fileLists = ParseFileList(inferParam.imglistPath);
        for (size_t i = 0; i < fileLists.size(); ++i)
        {
            std::string inputString = BuildInputString(fileLists[i]);
            ret = modelStereo->Infer(inputString, FileType::SingelImageFile);
            if (ret.size() == 0)
            {
                LOG(ERROR) << "fail to infer model";
                modelStereo->Unload();
                EnvDeinit();
                return 0;
            }
            
            result = modelDis->Infer(ret[0], inputString);
        }
    } else {
        ret = modelStereo->Infer(inferParam.imglistPath, FileType::JsonFile);
    }
    ret.clear();
    ret.shrink_to_fit();
    if (modelStereo->Unload() != 0)
    {
        LOG(ERROR) << "fail to unload model";
        EnvDeinit();
        return 0;
    }
    if (type) {
        result.clear();
        result.shrink_to_fit();
        if (modelDis->Unload() != 0)
        {
            LOG(ERROR) << "fail to unload model";
            EnvDeinit();
            return 0;
        }
    }
    EnvDeinit();
    return 0;
}