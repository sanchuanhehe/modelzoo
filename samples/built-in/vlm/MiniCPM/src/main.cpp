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
#include "MiniCpmInfer.h"

int main(int argc, char* argv[])
{
    InferParam inferParam;
    if (!ParseParamFromCmd(argc, argv, inferParam)) {
        LOG(ERROR) << "Fail to parse cmd!";
        return -1;
    }
    EnvInit(inferParam.aclConfigPath);
    MiniCpmInfer model;
    ExecuteParam param;
    ParseInputJsonFile(inferParam.imglistPath, param);
    if (param.loop == ZERO)
        param.loop++;
    string result;
    for (size_t i = 0; i < param.loop; ++i) {
        for (const auto& prompt : param.fileLists) {
            if (prompt.size() != QUESTION_PAIR_ITEM_NUM) {
                LOG(ERROR) << "Invalid file_list.json format: questions must be defined in image-question pairs";
                continue;
            }
            auto imagePath = prompt[ID_0];
            auto text = prompt[ID_1];
            LOG(INFO) << "\n Current question: \n Text: " << text << "\n Image path: " << imagePath;
            result = model.InferSingle(imagePath, text);
            LOG(INFO) << "\n MiniCPM infer result: \n " << result;
            LOG(INFO) << "Avg TTFT:" << model.GetTTFT() << " ms; TPS: " << model.GetTPS() << " token/s";
        }
    }
    LOG(INFO) << "Avg TTFT:" << model.GetTTFT() << " ms; TPS: " << model.GetTPS() << " token/s";
    model.UnLoad();
    EnvDeinit();
    return 0;
}