# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
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

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 安装aarch64-mix210-linux-gcc

# 安装cann包
# export NPU_INCLUDE_PATH=/path/to/ascend-toolkit/latest/acllib/include/acl
# export NPU_LIB_PATH=/path/to/ascend-toolkit/latest/acllib/lib64/stub
# eg: export NPU_INCLUDE_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib/include/acl
# eg：export NPU_LIB_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib/lib64/stub

# cmake ../src -DSOC_VERSION=SS928V100 -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_linux.cmake

# 编译器
set(CMAKE_C_COMPILER aarch64-mix210-linux-gcc)
set(CMAKE_CXX_COMPILER aarch64-mix210-linux-gcc)

# 路径适配
set(INC_PATH $ENV{NPU_INCLUDE_PATH})
set(LIB_PATH $ENV{NPU_LIB_PATH})
set(OPENCV_LIB_PATH "../opensource/opencv/lib/aarch64_linux")

set(CXX_STDLIB stdc++)
set(LINUX_PLATFORM TRUE)