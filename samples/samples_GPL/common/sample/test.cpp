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

#include <string>
#include <climits>
#include <iostream>
#include <getopt.h>
#include <memory>
#include "model.h"
#include "log.h"

using namespace Infer;

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

int main(int argc, char *argv[])
{
    int opt;
    const char *optstring = "hm:a:i:l:";
    struct option longOptions[] = {
        {"help", no_argument, NULL, 'h'},
        {"model", required_argument, NULL, 'm'},
        {"acl", required_argument, NULL, 'a'},
        {"input", required_argument, NULL, 'i'},
        {"loop", required_argument, NULL, 'l'},
        {0, 0, 0, 0}
    };
    std::string omModelPath;
    std::string aclConfigPath;
    std::string imglistPath;
    size_t loop = 1;

    while ((opt = getopt_long(argc, argv, optstring, longOptions, NULL)) != -1) {
        switch (opt) {
            case 'm':
                if (!PathToRealPath(optarg, omModelPath)) {
                    LOG(ERROR) << "parse model path error";
                    return 0;
                }
                break;
            case 'a':
                if (!PathToRealPath(optarg, aclConfigPath)) {
                    LOG(ERROR) << "parse acl config path error";
                    return 0;
                }
                break;
            case 'i':
                if (!PathToRealPath(optarg, imglistPath)) {
                    LOG(ERROR) << "parse image dir error";
                    return 0;
                }
                break;
            case 'l':{
                char *endptr = nullptr;
                loop = strtoull(optarg, &endptr, 0);
                if (*endptr != '\0') {
                    LOG(ERROR) << "incorrect input after -l/--loop, " << endptr;
                    return 0;
                }
                break;
            }
            case '?':
                return 0;
            default:
                return 0;
        }
    }

    EnvInit(aclConfigPath);
    std::unique_ptr<Model> model = std::make_unique<Model>();
    if (model->Load(omModelPath, Resnet50) != 0) {
        LOG(ERROR) << "model load failed";
        return 0;
    }
    for (size_t i = 0; i < loop; i++) {
        if (model->Infer(imglistPath).size() == 0) {
            LOG(ERROR) << "model infer failed";
            return 0;
        }
    }
    if (model->Unload() != 0) {
        EnvDeinit();
        return 0;
    }
    EnvDeinit();
    return 0;
}