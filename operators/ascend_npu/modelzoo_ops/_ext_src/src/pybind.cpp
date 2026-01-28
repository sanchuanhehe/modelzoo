/*
 * Copyright (c) ModelZoo. 2025-2026. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "functions.h"
#include <torch/extension.h>
#include <string>

using namespace std;

static once_flag initial_flag;
string g_opApiSoPath;

void op_api_lib_path_init(const string& op_api_lib_path)
{
    call_once(initial_flag, [&]() { g_opApiSoPath = op_api_lib_path; });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("op_api_lib_path_init", &op_api_lib_path_init);
    m.def("npu_group_points", &npu_group_points);
    m.def("npu_furthest_point_sampling", &npu_furthest_point_sampling);
    m.def("npu_cylinder_query", &npu_cylinder_query);
}
