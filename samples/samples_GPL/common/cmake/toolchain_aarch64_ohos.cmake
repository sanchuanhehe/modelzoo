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

# 下载和解压clang和sysroot
# export PATH=/path/to/clang/ohos/linux-x86_64/llvm/bin:$PATH
# export SYSROOT_PATH=/path/to/sysroot
# eg: export PATH=/home/hispark/shared/sample/modelzoo-dev_ohos/oh/clang/ohos/linux-x86_64/llvm/bin:$PATH
# eg: export SYSROOT_PATH=/home/hispark/shared/sample/modelzoo-dev_ohos/oh/sysroot

# 下载SDK https://gitee.com/HiSpark/ss928v100_clang  
# export NPU_INCLUDE_PATH=/path/to/smp/a55_linux/mpp/out/include
# export NPU_LIB_PATH=/path/to/smp/a55_linux/mpp/out/lib
# eg：export NPU_INCLUDE_PATH=/home/hispark/shared/ss928v100inner/platform/ss928v100_clang/smp/a55_linux/mpp/out/include
# eg: export NPU_LIB_PATH=/home/hispark/shared/ss928v100inner/platform/ss928v100_clang/smp/a55_linux/mpp/out/lib

# cmake ../src -DSOC_VERSION=SS928V100 -DCMAKE_TOOLCHAIN_FILE=../../../../common/cmake/toolchain_aarch64_ohos.cmake

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

add_compile_options(-target aarch64-linux-ohos)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=lld -target aarch64-linux-ohos")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=lld -target aarch64-linux-ohos")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=lld -target aarch64-linux-ohos")

# 路径适配
if(DEFINED ENV{SYSROOT_PATH})
    set(CMAKE_SYSROOT $ENV{SYSROOT_PATH})
    message(STATUS "CMAKE_SYSROOT set from env SYSROOT_PATH: ${CMAKE_SYSROOT}")
else()
    message(FATAL_ERROR "SYSROOT_PATH is not defined! Please set it first.")
endif()


set(OPENCV_LIB_PATH "../opensource/opencv/lib/aarch64_ohos")
set(INC_PATH "$ENV{NPU_INCLUDE_PATH}/svp_npu")
message(STATUS "INC_PATH: ${INC_PATH}")
set(LIB_PATH $ENV{NPU_LIB_PATH} "$ENV{NPU_LIB_PATH}/svp_npu")

set(CXX_STDLIB c++)
set(OHOS_PLATFORM TRUE)