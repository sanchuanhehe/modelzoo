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
    if (!ParseParamFromCmd(argc, argv, inferParam)) {
        LOG(ERROR) << "fail to parse cmd";
        return -1;
    }
    EnvInit(inferParam.aclConfigPath);
    std::unique_ptr<Model> model = std::make_unique<Model>();
    if (model->Load(inferParam.omModelPath, Unet) != 0) {
        LOG(ERROR) << "fail to load model";
        return 0;
    }
    auto ret = model->Infer(inferParam.imglistPath);
    if (ret.size() == 0) {
        LOG(ERROR) << "fail to infer model";
        model->Unload();
        EnvDeinit();
        return 0;
    }
    ret.clear();
    ret.shrink_to_fit();
    if (model->Unload() != 0) {
        LOG(ERROR) << "fail to unload model";
        EnvDeinit();
        return 0;
    }
    EnvDeinit();
    return 0;
}
