# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import modelzoo_ops._ext

def env_init():
    modelzoo_ops_entry = os.path.dirname(os.path.abspath(__file__))
    modelzoo_ops_opp_path = os.path.join(modelzoo_ops_entry, "..", "..")

    modelzoo_ops_op_api_lib_path = os.path.join(modelzoo_ops_opp_path, "op_api", "lib", "libcust_opapi.so")
    modelzoo_ops._ext.op_api_lib_path_init(modelzoo_ops_op_api_lib_path)

env_init()
